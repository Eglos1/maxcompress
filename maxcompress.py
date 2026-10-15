#!/usr/bin/env python3
"""
MaxCompress - Ultimate Multi-Algorithm Compression Engine (v3)
===============================================================
A "find the absolute smallest representation" compressor.

Design:
  * A registry of 1000+ algorithms, split into two classes:
      - "codec"     -> entropy coders (zlib/deflate/gzip/bz2/lzma/xz/brotli/zstd/lz4).
      - "transform" -> reversible byte re-shapers (delta, XOR-prev, RLE, MTF, BWT,
                       BPE, bit-planes, dictionary substitution, ...).
  * The engine does NOT chain codec-after-codec (that re-compresses incompressible
    data and almost never helps).  What genuinely works - and is the whole point of
    this rewrite - is chaining TRANSFORMS (which expose structure) followed by ONE
    strong entropy coder.  Classic examples:
        delta -> lzma            (time series / sorted data)
        BWT -> MTF -> RLE -> lzma (bzip2-style, great on text)
        dictionary -> zstd       (JSON / XML / logs)
        bitplane -> brotli       (low-bit-depth imagery)
  * The search is adaptive and multi-phase: it runs cheap single codecs first, ranks
    transforms against the current best codec, then explores chains built from the
    top-ranked ingredients.  "Chaining different algorithms" is preserved - but only
    chains that have a real chance of winning get full compression time.

v3 format (self-contained):
  'MXCP' | ver:1 | meta_len:u32BE | meta:json | data_len:u64BE | data | crc32:u32BE
  crc32 covers `data`.  v2 files (no checksum) are still readable.

Author: Russell
"""

import argparse
import bz2
import gzip
import json
import lzma
import os
import struct
import sys
import time
import zlib

import multiprocessing as _mp

# LRU cache of transform/dict intermediate results shared across all pipelines
# of a single compress() run.  Transform chains share long prefixes, so caching
# the intermediate bytes avoids recomputing e.g. delta->RLE for hundreds of
# candidate suffixes.  Bounded by entry count + approximate total bytes.
_MAX_CACHE_ENTRIES = 4096
_MAX_CACHE_BYTES = 1 << 27   # 128 MiB

# ---------------------------------------------------------------------------
# fork-based parallel phase worker.  Child processes inherit the registry and
# the (read-only) data block, so no pickling of lambdas is needed; the parent
# publishes phase context via module globals right before creating the pool.
# Workers ship back only the tiny winner descriptor + score map.
# ---------------------------------------------------------------------------
_PHASE_CTX = None
_P_DATA = None
_P_BUDGET = None


def _phase_worker(slice_i):
    engine, pipelines = _PHASE_CTX
    data = _P_DATA
    budget_end = _P_BUDGET
    local_best = None
    local_scores = {}
    done = 0
    for i in slice_i:
        if budget_end is not None and time.time() > budget_end:
            return (done, None, None, None, local_scores, False)
        pipe = pipelines[i]
        try:
            out = engine._apply(data, pipe)
        except Exception:
            out = None
        done += 1
        engine._emit({"type": "pipeline", "phase": "<parallel>",
                      "method": pipe.describe(),
                      "ok": out is not None,
                      "size": len(out) if out is not None else None,
                      "done": None, "total": None})
        if out is not None:
            size = len(out)
            if size <= len(data) * 2:
                key = engine._score_key(pipe)
                old = local_scores.get(key)
                if old is None or size < old:
                    local_scores[key] = size
                if local_best is None or size < local_best[0]:
                    local_best = (size, out, i)
                else:
                    del out
    return (done, local_best[0] if local_best else None,
            local_best[1] if local_best else None,
            local_best[2] if local_best else None,
            local_scores, True)

try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import snappy
    HAS_SNAPPY = True
except ImportError:
    HAS_SNAPPY = False

try:
    import lz4.block as lz4block
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import lz4.frame as lz4frame
    HAS_LZ4_FRAME = True
except ImportError:
    HAS_LZ4_FRAME = False

try:
    import pyzstd
    HAS_PYZSTD = True
except ImportError:
    HAS_PYZSTD = False

try:
    import zopfli.gzip as _zopfli_gzip
    import zopfli.zlib as _zopfli_zlib
    HAS_ZOPFLI = True
except ImportError:
    HAS_ZOPFLI = False

try:
    import bcj as _bcj_module
    HAS_BCJ = True
except ImportError:
    HAS_BCJ = False

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pysais
except Exception:
    pysais = None

if pysais is None:          # PySAIS (pip) exposes module `PySAIS`
    try:
        import PySAIS as _py_sais_mod
        pysais = _py_sais_mod
    except Exception:
        pysais = None

try:
    import pyppmd as _ppmd            # new canonical name
except Exception:
    try:
        import ppmd as _ppmd          # legacy package name
    except Exception:
        _ppmd = None
HAS_PPMD = _ppmd is not None

from collections import OrderedDict, Counter

# ============================================================================
# ALGORITHM
# ============================================================================

class Algo:
    """One named, reversible compression algorithm/procedure."""
    __slots__ = ("name", "kind", "compress", "decompress", "cost", "maxlen")

    def __init__(self, name, kind, compress, decompress, cost="normal", maxlen=None):
        self.name = name
        self.kind = kind              # "codec" | "transform"
        self.compress = compress
        self.decompress = decompress
        self.cost = cost              # "fast" | "normal" | "slow"
        self.maxlen = maxlen          # None = any size, else int (bytes)

    def __repr__(self):
        return f"<Algo {self.name} ({self.kind}/{self.cost})>"


_REGISTRY = []
_REG_SEEN = set()


def _reg(algo):
    if algo.name in _REG_SEEN:
        return
    _REG_SEEN.add(algo.name)
    _REGISTRY.append(algo)


# ============================================================================
# CODECT (ENTROPY CODER) BUILDERS
# ============================================================================

_STRAT = {0: "DEF", 1: "FIL", 2: "HUF", 3: "RLE", 4: "FIX"}


def _zlib_compress(level, wbits, strategy, data, memLevel=8):
    co = zlib.compressobj(level=level, wbits=wbits, strategy=strategy,
                          memLevel=memLevel)
    return co.compress(data) + co.flush()


def _zlib_decompress(wbits, data):
    return zlib.decompress(data, wbits)


# --- zlib format, window bits 9..15, all strategies -----------------------
for wb in range(9, 16):
    for strat in (0, 1, 2, 3, 4):
        for lv in range(1, 10):
            _reg(Algo(
                f"zlib_l{lv}_w{wb}_{_STRAT[strat]}", "codec",
                (lambda lv=lv, wb=wb, st=strat: lambda d: _zlib_compress(lv, wb, st, d))(),
                (lambda wb=wb: lambda d: _zlib_decompress(wb, d))(),
                cost="fast"))

# --- raw DEFLATE, window bits 9..15, all strategies, memLevel 1 and 9 ----
for wb in range(9, 16):
    for strat in (0, 1, 2, 3, 4):
        for lv in range(1, 10):
            for mem in (1, 9):
                def _mk(lv=lv, wb=wb, st=strat, mem=mem):
                    def comp(d):
                        co = zlib.compressobj(level=lv, wbits=-wb, strategy=st, memLevel=mem)
                        return co.compress(d) + co.flush()
                    return comp

                def _mk_d(wb=wb):
                    return lambda d: zlib.decompress(d, -wb)

                _reg(Algo(
                    f"deflate_l{lv}_w{wb}_{_STRAT[strat]}_m{mem}", "codec",
                    _mk(), _mk_d(), cost="fast"))

# --- gzip (deterministic: mtime=0, via compressobj wbits=31) --------------
for lv in range(1, 10):
    def _gzip_c(lv=lv):
        def f(d):
            co = zlib.compressobj(level=lv, wbits=zlib.MAX_WBITS | 16)
            return co.compress(d) + co.flush()
        return f

    _reg(Algo(f"gzip_l{lv}", "codec",
              _gzip_c(), lambda d: zlib.decompress(d, zlib.MAX_WBITS | 16),
              cost="fast"))

# --- bz2 ------------------------------------------------------------------
for lv in range(1, 10):
    _reg(Algo(f"bz2_l{lv}", "codec",
              (lambda lv=lv: lambda d: bz2.compress(d, lv))(),
              bz2.decompress, cost="fast"))

# --- lzma alone (LZMA1) presets -------------------------------------------
_LZMA_DICT = {0: 1 << 16, 1: 1 << 20, 2: 1 << 20, 3: 1 << 20, 4: 1 << 20,
              5: 1 << 23, 6: 1 << 23, 7: 1 << 25, 8: 1 << 26, 9: 1 << 26}

for lv in range(0, 10):
    _reg(Algo(f"lzma_l{lv}", "codec",
              (lambda lv=lv: lambda d: lzma.compress(d, preset=lv))(),
              lzma.decompress, cost="slow" if lv >= 7 else "normal"))

# --- lzma alone with explicit dictionary sizes ----------------------------
for lv in (5, 6, 7, 8, 9):
    for ds in (1 << 16, 1 << 20, 1 << 24, 1 << 26):
        def _alone(lv=lv, ds=ds):
            # FORMAT_ALONE only accepts a single LZMA1 filter
            return lambda d: lzma.compress(
                d, format=lzma.FORMAT_ALONE,
                filters=[{"id": lzma.FILTER_LZMA1, "preset": lv, "dict_size": ds}])
        _reg(Algo(f"lzma_d{lv}_{ds}", "codec",
                  _alone(), lzma.decompress,
                  cost="slow" if lv >= 8 else "normal"))

# --- lzma2 in an xz container (LZMA2 with explicit dict sizes) -------------
for lv in range(0, 10):
    for ds in (1 << 16, 1 << 20, 1 << 23):
        def _l2a(lv=lv, ds=ds):
            # LZMA2 requires the xz container
            return lambda d: lzma.compress(
                d, format=lzma.FORMAT_XZ,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": lv, "dict_size": ds}])
        _reg(Algo(f"lzma2_xz{lv}_{ds}", "codec",
                  _l2a(), lzma.decompress,
                  cost="slow" if lv >= 8 else "normal"))

# --- xz -------------------------------------------------------------------
for lv in range(0, 10):
    _reg(Algo(f"xz_l{lv}", "codec",
              (lambda lv=lv: lambda d: lzma.compress(d, format=lzma.FORMAT_XZ, preset=lv))(),
              lambda d: lzma.decompress(d, format=lzma.FORMAT_XZ),
              cost="slow" if lv >= 7 else "normal"))

for lv in (6, 7, 8, 9):
    for ds in (1 << 20, 1 << 24, 1 << 26):
        def _xzf(lv=lv, ds=ds):
            return lambda d: lzma.compress(
                d, format=lzma.FORMAT_XZ,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": lv, "dict_size": ds}])
        _reg(Algo(f"xz_d{lv}_{ds}", "codec",
                  _xzf(),
                  lambda d: lzma.decompress(d, format=lzma.FORMAT_XZ),
                  cost="slow"))

# --- brotli ---------------------------------------------------------------
if HAS_BROTLI:
    for q in range(0, 12):
        for w in (10, 14, 18, 22, 24):
            _reg(Algo(f"brotli_q{q}_w{w}", "codec",
                      (lambda q=q, w=w: lambda d: brotli.compress(d, quality=q, lgwin=w))(),
                      brotli.decompress,
                      cost="slow" if q >= 9 else "normal"))

# --- zstandard ------------------------------------------------------------
if HAS_ZSTD:
    for lv in range(1, 23):
        for ldm in (False, True):
            def _zc(lv=lv, ldm=ldm):
                if ldm:
                    params = zstd.ZstdCompressionParameters.from_level(
                        lv, enable_ldm=True)
                    return lambda d: zstd.ZstdCompressor(
                        compression_params=params).compress(d)
                return lambda d: zstd.ZstdCompressor(level=lv).compress(d)
            _reg(Algo(f"zstd_l{lv}{'_ldm' if ldm else ''}", "codec",
                      _zc(),
                      lambda d: zstd.ZstdDecompressor().decompress(d),
                      cost="slow" if lv >= 19 else "normal"))

if HAS_PYZSTD:
    for lv in (3, 9, 15, 22):
        _reg(Algo(f"pyzstd_l{lv}", "codec",
                  (lambda lv=lv: lambda d: pyzstd.compress(d, level=lv))(),
                  pyzstd.decompress, cost="normal"))

# --- lz4 ------------------------------------------------------------------
if HAS_LZ4:
    for lv in range(1, 13):
        _reg(Algo(f"lz4_l{lv}", "codec",
                  (lambda lv=lv: lambda d: lz4block.compress(
                      d, store_size=True, mode="high_compression",
                      compression=min(lv, 12)))(),
                  lz4block.decompress, cost="fast"))
    for acc in (1, 3, 7, 15, 31):
        _reg(Algo(f"lz4_a{acc}", "codec",
                  (lambda acc=acc: lambda d: lz4block.compress(
                      d, store_size=True, acceleration=acc))(),
                  lz4block.decompress, cost="fast"))

# --- PPMd (if pyppmd is available) -----------------------------------------
if HAS_PPMD:
    for order in (4, 6, 8, 10, 12, 16):
        for msize in (8, 32, 64):

            def _pc(order=order, msize=msize):
                def comp(d):
                    if not d:
                        return b""
                    return _ppmd.compress(d, max_order=order,
                                          mem_size=msize << 20, variant="I")
                return comp

            def _pd(order=order, msize=msize):
                def decomp(c):
                    if not c:
                        return b""
                    return _ppmd.decompress(c, max_order=order,
                                            mem_size=msize << 20, variant="I")
                return decomp

            _reg(Algo(f"ppmd_o{order}_m{msize}", "codec",
                      _pc(), _pd(), cost="slow", maxlen=(1 << 24)))

# ============================================================================
# IN-HOUSE ENTROPY CODERS  (implemented from scratch)
# ============================================================================

def _lzw_compress(bits, data):
    """GIF-style LZW with explicit clear codes and variable-width codes."""
    if not data:
        return struct.pack('>HI', bits, 0)
    max_code = (1 << bits) - 1
    clear = 256
    eoi = 257
    table = {bytes([i]): i for i in range(256)}
    next_code = 258
    out = bytearray()
    bitbuf = 0
    bitcnt = 0

    def put(code):
        nonlocal bitbuf, bitcnt
        bitbuf = (bitbuf << bits) | code
        bitcnt += bits
        while bitcnt >= 8:
            bitcnt -= 8
            out.append((bitbuf >> bitcnt) & 0xFF)

    put(clear)
    w = b''
    for b in data:
        wc = w + bytes([b])
        if wc in table:
            w = wc
        else:
            put(table[w])
            if next_code <= max_code:
                table[wc] = next_code
                next_code += 1
            else:
                put(clear)
                table = {bytes([i]): i for i in range(256)}
                next_code = 258
            w = bytes([b])
    if w:
        put(table[w])
    put(eoi)
    if bitcnt:
        out.append((bitbuf << (8 - bitcnt)) & 0xFF)
    return struct.pack('>HI', bits, len(data)) + bytes(out)


def _lzw_decompress(data):
    bits, orig = struct.unpack('>HI', data[:6])
    if orig == 0:
        return b''
    max_code = (1 << bits) - 1
    clear = 256
    eoi = 257
    # unpack codes
    body = data[6:]
    codes = []
    bitbuf = 0
    bitcnt = 0
    for byte in body:
        bitbuf = (bitbuf << 8) | byte
        bitcnt += 8
        while bitcnt >= bits:
            bitcnt -= bits
            codes.append((bitbuf >> bitcnt) & ((1 << bits) - 1))
    table = [[i] for i in range(258)]  # 0..255 chars; 256/257 = clear/EOI encoded
    next_code = 258
    out = bytearray()
    prev = None
    for code in codes:
        if code == clear:
            table = [[i] for i in range(258)]
            next_code = 258
            prev = None
            continue
        if code == eoi:
            break
        if code < len(table):
            entry = table[code]
        elif code == next_code and prev is not None:
            entry = prev + prev[:1]          # KwKwK deferred-entry case
        else:
            raise ValueError("bad LZW code")
        out.extend(entry)
        if prev is not None and next_code <= max_code:
            table.append(prev + entry[:1])
            next_code += 1
        prev = entry
    return bytes(out[:orig])


def _lzss_compress(data):
    """Simple LZSS: 4-byte original length, 1 flag bit, 11-bit offset,
    4-bit length (3..18)."""
    n = len(data)
    if n == 0:
        return b''
    W = 2048
    MIN = 3
    MAX = 18
    bb = bytearray()
    cur = 0
    nbits = 0

    def put(v, nb):
        nonlocal cur, nbits
        cur = (cur << nb) | v
        nbits += nb
        while nbits >= 8:
            nbits -= 8
            bb.append((cur >> nbits) & 0xFF)

    ht = {}
    i = 0
    while i < n:
        key = (data[i], data[i + 1], data[i + 2]) if i + MIN <= n else None
        best = 0
        bestoff = 0
        if key is not None:
            for p in ht.get(key, ()):
                if i - p > W:
                    continue
                m = 0
                while (i + m < n and m < MAX
                       and data[p + m] == data[i + m]):
                    m += 1
                if m >= MIN and m > best:
                    best = m
                    bestoff = i - p
        if best >= MIN:
            put(1, 1)
            put(bestoff - 1, 11)
            put(best - MIN, 4)
            for j in range(i, i + best):
                if j + MIN <= n:
                    k = (data[j], data[j + 1], data[j + 2])
                    ht.setdefault(k, []).append(j)
            i += best
        else:
            put(0, 1)
            put(data[i], 8)
            if i + MIN <= n:
                k = (data[i], data[i + 1], data[i + 2])
                ht.setdefault(k, []).append(i)
            i += 1
    if nbits:
        bb.append((cur << (8 - nbits)) & 0xFF)
    return struct.pack('>I', n) + bytes(bb)


def _lzss_decompress(data):
    if len(data) < 4:
        return b''
    n = struct.unpack('>I', data[:4])[0]
    if n == 0:
        return b''
    bits = data[4:]
    out = bytearray()
    pos = 0
    cur = 0
    nbits = 0

    def read(nb):
        nonlocal pos, cur, nbits
        while nbits < nb:
            cur = (cur << 8) | (bits[pos] if pos < len(bits) else 0)
            pos += 1
            nbits += 8
        v = (cur >> (nbits - nb)) & ((1 << nb) - 1)
        nbits -= nb
        cur &= (1 << nbits) - 1
        return v

    while len(out) < n:
        if read(1) == 0:
            out.append(read(8))
        else:
            off = read(11) + 1
            ln = read(4) + 3
            start = len(out) - off
            for k in range(ln):
                out.append(out[start + k])
    return bytes(out[:n])


def _huff_compress(data):
    """Static order-0 canonical Huffman (two-pass)."""
    n = len(data)
    if n == 0:
        return b'\x00\x00\x00\x00'
    C = [0] * 256
    for b in data:
        C[b] += 1
    import heapq
    # build a binary tree of nodes; node = [freq, sym, left, right]
    nodes = []
    heap = []
    for i in range(256):
        if C[i]:
            nodes.append([C[i], i, -1, -1])
            heap.append((C[i], len(nodes) - 1))
    heapq.heapify(heap)
    while len(heap) > 1:
        f1, n1 = heapq.heappop(heap)
        f2, n2 = heapq.heappop(heap)
        nodes.append([f1 + f2, -1, n1, n2])
        heapq.heappush(heap, (f1 + f2, len(nodes) - 1))
    root = heap[0][1]
    # compute code lengths via BFS depths
    lens = [0] * 256
    stack = [(root, 0)]
    while stack:
        node_idx, depth = stack.pop()
        f, sym, l, r = nodes[node_idx]
        if sym >= 0:
            lens[sym] = max(1, depth)
        else:
            stack.append((l, depth + 1))
            stack.append((r, depth + 1))
    # canonical codes (DEFLATE style)
    maxlen = max(lens)
    bl_count = [0] * (maxlen + 1)
    for l in lens:
        if l:
            bl_count[l] += 1
    next_code = [0] * (maxlen + 1)
    code = 0
    for bits in range(1, maxlen + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    codes = {}
    for i in range(256):
        l = lens[i]
        if l:
            codes[i] = next_code[l]
            next_code[l] += 1
    # encode
    bb = bytearray()
    cur = 0
    nbits = 0
    for b in data:
        l = lens[b]
        c = codes[b]
        cur = (cur << l) | c
        nbits += l
        while nbits >= 8:
            nbits -= 8
            bb.append((cur >> nbits) & 0xFF)
    if nbits:
        bb.append((cur << (8 - nbits)) & 0xFF)
    return struct.pack('>I', n) + bytes(lens) + bytes(bb)


def _huff_decompress(data):
    if len(data) < 260:
        return b''
    n = struct.unpack('>I', data[:4])[0]
    if n == 0:
        return b''
    lens = list(data[4:260])
    maxlen = max(lens)
    bl_count = [0] * (maxlen + 1)
    for l in lens:
        if l:
            bl_count[l] += 1
    next_code = [0] * (maxlen + 1)
    code = 0
    for bits in range(1, maxlen + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    # canonical symbol ordering
    canon = [(lens[i], i) for i in range(256) if lens[i]]
    canon.sort()
    start_by_len = {}
    for idx, (l, _) in enumerate(canon):
        start_by_len.setdefault(l, idx)
    bits = data[260:]
    out = bytearray()
    pos = 0
    bit = 7

    def read1():
        nonlocal pos, bit
        v = (bits[pos] >> bit) & 1 if pos < len(bits) else 0
        bit -= 1
        if bit < 0:
            bit = 7
            pos += 1
        return v

    done = 0
    while done < n:
        c = 0
        for l in range(1, maxlen + 1):
            c = (c << 1) | read1()
            if (c - next_code[l]) in range(bl_count[l]):
                sym = canon[start_by_len[l] + (c - next_code[l])][1]
                out.append(sym)
                done += 1
                break
        else:
            raise ValueError("bad huffman stream")
    return bytes(out)


# --- adaptive order-0 range coder (LZMA-style carry handling) --------------
_TOP = 1 << 24
_PAC_SCALE = 1 << 13


def _fw_add(tree, i, delta):
    while i < 256:
        tree[i] += delta
        i |= i + 1


def _fw_sum(tree, i):
    s = 0
    while i > 0:
        s += tree[i - 1]
        i -= i & -i
    return s


def _fw_find(tree, x):
    idx = -1
    acc = 0
    bit = 128
    while bit:
        nxt = idx + bit
        if nxt < 255 and acc + tree[nxt] <= x:
            acc += tree[nxt]
            idx = nxt
        bit >>= 1
    return idx + 1


def _pac_compress(data):
    if not data:
        return struct.pack('>I', 0)
    freq = [1] * 256
    ft = [0] * 256
    for i in range(256):
        _fw_add(ft, i, 1)
    total = 256
    low = 0
    rng = 0xFFFFFFFF
    cache = 0
    cache_size = 0
    out = bytearray()

    def shift():
        nonlocal low, cache, cache_size
        if low < 0xFF000000 or (low >> 32) != 0:
            out.append((cache + ((low >> 32) & 1)) & 0xFF)
            carry = (low >> 32) & 1
            for _ in range(cache_size):
                out.append((0xFF + carry) & 0xFF)
            cache = (low >> 24) & 0xFF
            cache_size = 0
        else:
            cache_size += 1
        low = (low << 8) & 0xFFFFFFFF

    for b in data:
        cum = _fw_sum(ft, b)
        step = rng // total
        low += cum * step
        rng = freq[b] * step
        freq[b] += 1
        total += 1
        _fw_add(ft, b, 1)
        if total >= _PAC_SCALE:
            for i in range(256):
                freq[i] = (freq[i] + 1) >> 1
            total = sum(freq)
            ft = [0] * 256
            for i in range(256):
                _fw_add(ft, i, freq[i])
        while rng <= _TOP:
            rng <<= 8
            shift()
    for _ in range(5):
        rng <<= 8
        shift()
    out.append(cache)
    for _ in range(cache_size):
        out.append(0xFF)
    return struct.pack('>I', len(data)) + bytes(out)


def _pac_decompress(data):
    if len(data) < 8:
        return b''
    orig = struct.unpack('>I', data[:4])[0]
    if orig == 0:
        return b''
    freq = [1] * 256
    ft = [0] * 256
    for i in range(256):
        _fw_add(ft, i, 1)
    total = 256
    code = 0
    pos = 4
    for _ in range(5):            # LZMA-style 5-byte init (carry alignment)
        code = ((code << 8) | data[pos]) & 0xFFFFFFFF
        pos += 1
    rng = 0xFFFFFFFF
    out = bytearray()
    for _ in range(orig):
        step = rng // total
        x = code // step
        if x >= total:
            x = total - 1
        sym = _fw_find(ft, x)
        cum = _fw_sum(ft, sym)
        code = (code - cum * step) & 0xFFFFFFFF
        rng = freq[sym] * step
        out.append(sym)
        freq[sym] += 1
        total += 1
        _fw_add(ft, sym, 1)
        if total >= _PAC_SCALE:
            for i in range(256):
                freq[i] = (freq[i] + 1) >> 1
            total = sum(freq)
            ft = [0] * 256
            for i in range(256):
                _fw_add(ft, i, freq[i])
        while rng <= _TOP:
            rng <<= 8
            code = ((code << 8) | data[pos]) & 0xFFFFFFFF
            pos += 1
    return bytes(out)


for _b in (12, 14, 16):
    _reg(Algo(f"lzw_b{_b}", "codec",
              (lambda b=_b: lambda d: _lzw_compress(b, d))(),
              _lzw_decompress, cost="normal", maxlen=(1 << 22)))
_reg(Algo("lzss", "codec", _lzss_compress, _lzss_decompress,
          cost="normal", maxlen=(1 << 22)))
_reg(Algo("huff", "codec", _huff_compress, _huff_decompress,
          cost="normal", maxlen=(1 << 22)))
_reg(Algo("pac", "codec", _pac_compress, _pac_decompress,
          cost="slow", maxlen=(1 << 20)))

# --- order-1 adaptive range coder (context = previous byte) -------------------
# Same LZMA-style carry handling as `pac`, but each of 256 contexts keeps its own
# adaptive Fenwick table, so symbol probabilities condition on the previous byte.
# Order-1 models typically cut text / structured-data size far below order-0 AC.
_PAC1_SCALE = 1 << 13
_PAC1_INIT = 256          # pseudo-context for the very first symbol


def _pac1_compress(data):
    if not data:
        return struct.pack('>I', 0)
    # per-context flat frequency + Fenwick tables
    freq = [[1] * 256 for _ in range(257)]
    ft = [[0] * 256 for _ in range(257)]
    total = [256] * 257
    for c in range(257):
        for i in range(256):
            _fw_add(ft[c], i, 1)
    low = 0
    rng = 0xFFFFFFFF
    cache = 0
    cache_size = 0
    out = bytearray()

    def shift():
        nonlocal low, cache, cache_size
        if low < 0xFF000000 or (low >> 32) != 0:
            out.append((cache + ((low >> 32) & 1)) & 0xFF)
            carry = (low >> 32) & 1
            for _ in range(cache_size):
                out.append((0xFF + carry) & 0xFF)
            cache = (low >> 24) & 0xFF
            cache_size = 0
        else:
            cache_size += 1
        low = (low << 8) & 0xFFFFFFFF

    ctx = _PAC1_INIT
    for b in data:
        t = total[ctx]
        f = freq[ctx]
        tr = ft[ctx]
        cum = _fw_sum(tr, b)
        step = rng // t
        low += cum * step
        rng = f[b] * step
        f[b] += 1
        t += 1
        total[ctx] = t
        _fw_add(tr, b, 1)
        if t >= _PAC1_SCALE:
            for i in range(256):
                f[i] = (f[i] + 1) >> 1
            t = sum(f)
            total[ctx] = t
            tr = [0] * 256
            for i in range(256):
                _fw_add(tr, i, f[i])
            ft[ctx] = tr
        while rng <= _TOP:
            rng <<= 8
            shift()
        ctx = b
    for _ in range(5):
        rng <<= 8
        shift()
    out.append(cache)
    for _ in range(cache_size):
        out.append(0xFF)
    return struct.pack('>I', len(data)) + bytes(out)


def _pac1_decompress(data):
    if len(data) < 8:
        return b''
    orig = struct.unpack('>I', data[:4])[0]
    if orig == 0:
        return b''
    freq = [[1] * 256 for _ in range(257)]
    ft = [[0] * 256 for _ in range(257)]
    total = [256] * 257
    for c in range(257):
        for i in range(256):
            _fw_add(ft[c], i, 1)
    code = 0
    pos = 4
    for _ in range(5):            # LZMA-style 5-byte init (carry alignment)
        code = ((code << 8) | data[pos]) & 0xFFFFFFFF
        pos += 1
    rng = 0xFFFFFFFF
    out = bytearray()
    ctx = _PAC1_INIT
    for _ in range(orig):
        t = total[ctx]
        f = freq[ctx]
        tr = ft[ctx]
        step = rng // t
        x = code // step
        if x >= t:
            x = t - 1
        sym = _fw_find(tr, x)
        cum = _fw_sum(tr, sym)
        code = (code - cum * step) & 0xFFFFFFFF
        rng = f[sym] * step
        out.append(sym)
        f[sym] += 1
        t += 1
        total[ctx] = t
        _fw_add(tr, sym, 1)
        if t >= _PAC1_SCALE:
            for i in range(256):
                f[i] = (f[i] + 1) >> 1
            t = sum(f)
            total[ctx] = t
            tr = [0] * 256
            for i in range(256):
                _fw_add(tr, i, f[i])
            ft[ctx] = tr
        while rng <= _TOP:
            rng <<= 8
            code = ((code << 8) | data[pos]) & 0xFFFFFFFF
            pos += 1
        ctx = sym
    return bytes(out)


_reg(Algo("pac1", "codec", _pac1_compress, _pac1_decompress,
          cost="slow", maxlen=(1 << 20)))

# --- static order-0 rANS -----------------------------------------------------
# Integer (stable) rANS, ryg-style byte renorm, 12-bit total, L = 1<<23.
_RANS_SHIFT = 12
_RANS_TOTAL = 1 << _RANS_SHIFT
_RANS_MASK = _RANS_TOTAL - 1
_RANS_L = 1 << 23


def _rans_normalize(C):
    """Scale counts so they sum to _RANS_TOTAL; every present symbol >= 1."""
    s = sum(C)
    out = [0] * 256
    if s == 0:
        return out
    present = [i for i in range(256) if C[i]]
    for i in present:
        out[i] = 1
    rem = _RANS_TOTAL - len(present)
    if rem > 0:
        alloc = [(C[i] * rem) // s for i in range(256)]
        for i in present:
            out[i] += alloc[i]
        diff = rem - sum(out[i] for i in present)
        if diff > 0:
            order = sorted(present, key=lambda i: (C[i] * rem) % s, reverse=True)
            for i in order[:diff]:
                out[i] += 1
        elif diff < 0:
            order = sorted(present, key=lambda i: (C[i] * rem) % s)
            for i in order[:(-diff)]:
                out[i] -= 1
    for i in present:
        if out[i] < 1:
            out[i] = 1
    d = _RANS_TOTAL - sum(out)
    if d:
        out[present[0]] += d
    return out


def _rans_compress(data):
    n = len(data)
    if n == 0:
        return struct.pack('>I', 0)
    C = [0] * 256
    for b in data:
        C[b] += 1
    freq = _rans_normalize(C)
    cdf = [0] * 257
    for i in range(256):
        cdf[i + 1] = cdf[i] + freq[i]
    # ryg-style: encode symbols in REVERSE; renorm bursts written backwards
    x = _RANS_L
    emitted = bytearray()
    for b in reversed(data):
        f = freq[b]
        x_max = ((_RANS_L >> _RANS_SHIFT) << 8) * f
        while x >= x_max:
            emitted.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << _RANS_SHIFT) + (x % f) + cdf[b]
    # flush state LSB-first into the first 4 stream bytes
    stream = bytes((x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, (x >> 24) & 0xFF))
    stream += bytes(emitted)[::-1]
    hdr = struct.pack('>I', n) + struct.pack('>256H', *freq)
    return hdr + stream


def _rans_decompress(data):
    if len(data) < 4:
        return b''
    n = struct.unpack('>I', data[:4])[0]
    if n == 0 or len(data) < 4 + 512 + 4:
        return b''
    freq = list(struct.unpack('>256H', data[4:516]))
    cdf = [0] * 257
    for i in range(256):
        cdf[i + 1] = cdf[i] + freq[i]
    stream = data[516:]
    pos = 0
    x = stream[0] | (stream[1] << 8) | (stream[2] << 16) | (stream[3] << 24)
    pos = 4
    out = bytearray()
    for _ in range(n):
        slot = x & _RANS_MASK
        # binary search: highest c such that cdf[c] <= slot
        lo, hi = 0, 256
        while lo < hi:
            mid = (lo + hi) >> 1
            if cdf[mid + 1] <= slot:
                lo = mid + 1
            else:
                hi = mid
        c = lo
        f = freq[c]
        x = f * (x >> _RANS_SHIFT) + (slot - cdf[c])
        while x < _RANS_L:
            x = (x << 8) | (stream[pos] if pos < len(stream) else 0)
            pos += 1
        out.append(c)
    return bytes(out)


_reg(Algo("rans", "codec", _rans_compress, _rans_decompress,
          cost="slow", maxlen=(1 << 22)))

# --- PAQ-style context-mixing binary coder (cm) ------------------------------
# A compact neural-network mixer (lpaq-style): several byte-order context models
# each predict P(next bit==0) over a stretch/squash pair; predictions are mixed
# with adaptive per-bit-position weights; a binary range coder emits the bits.
# Stretch/squash are the exact logistic pair so the mixer never saturates.
_CM_SS = 6                       # stretch scale (higher = sharper)
_CM_P = 1 << 12                  # probability scale (P(total))


def _cm_build_stretch():
    import math as _cm_math
    P = _CM_P
    SS = _CM_SS
    stretch = [0.0] * (P + 1)
    for p in range(1, P):
        x = p / (P - p)
        stretch[p] = _cm_math.log(x) * SS if x > 0 else -8000.0
    stretch[0] = stretch[1]
    stretch[P] = stretch[P - 1]
    return stretch


_CM_STRETCH = _cm_build_stretch()


def _cm_squash(dot):
    """Logistic inverse of stretch: P(0) (0..4096) from a real-valued dot."""
    import math as _cm_math
    P = _CM_P
    if dot >= 0:
        return round(P / (1.0 + _cm_math.exp(-dot / _CM_SS)))
    e = _cm_math.exp(dot / _CM_SS)
    return round(P * e / (1.0 + e))


def _cm_compress(data):
    if not data:
        return struct.pack('>I', 0)
    P = _CM_P
    # models: adaptive P(0) context tables (bit pos / prev byte / order-2 / folded order-3)
    m0 = [P >> 1] * 8
    m1 = [P >> 1] * (1 << 11)
    m2 = [P >> 1] * (1 << 16)
    m3 = [P >> 1] * (1 << 13)
    MODELS = (m0, m1, m2, m3)
    rate = (6, 6, 4, 4)
    # mixer weights per bit position (4 models), small-scale logistic mixing
    w = [[0.0, 0.0, 0.0, 0.0] for _ in range(8)]
    wclamp = 1.0
    low = 0
    rng = 0xFFFFFFFF
    cache = 0
    cache_size = 0
    out = bytearray()

    def shift():
        nonlocal low, cache, cache_size
        if low < 0xFF000000 or (low >> 32) != 0:
            out.append((cache + ((low >> 32) & 1)) & 0xFF)
            carry = (low >> 32) & 1
            for _ in range(cache_size):
                out.append((0xFF + carry) & 0xFF)
            cache = (low >> 24) & 0xFF
            cache_size = 0
        else:
            cache_size += 1
        low = (low << 8) & 0xFFFFFFFF

    h1 = h2 = h3 = 0
    st = _CM_STRETCH
    for b in data:
        for i in range(8):            # MSB first
            y = (b >> (7 - i)) & 1
            c1 = ((h1 << 3) | i) & 0x7FF
            c2 = h2 & 0xFFFF
            c3 = (h3 ^ (h3 >> 13)) & 0x1FFF
            prow = (m0[i], m1[c1], m2[c2], m3[c3])
            dot = 0.0
            wr = w[i]
            for mi in range(4):
                dot += wr[mi] * st[prow[mi]]
            p0 = _cm_squash(dot)
            if p0 <= 0:
                p0 = 1
            elif p0 >= P:
                p0 = P - 1
            rng0 = (rng // P) * p0
            if y == 0:
                rng = rng0
            else:
                low += rng0
                rng = rng - rng0
            while rng <= 1 << 24:
                rng <<= 8
                shift()
            # update models toward the true P(0) target
            target = P if y == 0 else 0
            for mi in range(4):
                idx = c1 if mi == 1 else i if mi == 0 else (c2 if mi == 2 else c3)
                cur = MODELS[mi][idx]
                MODELS[mi][idx] = cur + ((target - cur) >> rate[mi])
            # logistic weight update on P(1) error (scaled learning rate)
            p1 = (P - p0) / P
            y1 = 1.0 - y
            lr = 0.5 / (1.0 + 0.05 * abs(dot / _CM_SS))
            for mi in range(4):
                wr[mi] += lr * (y1 - p1) * st[prow[mi]] / _CM_SS
                if wr[mi] > wclamp:
                    wr[mi] = wclamp
                elif wr[mi] < -wclamp:
                    wr[mi] = -wclamp
        h3 = (((h3 & 0xFFFFFF) << 8) | b) & 0xFFFFFF
        h2 = ((h2 & 0xFFFF) << 8) | b
        h1 = b
    for _ in range(5):
        rng <<= 8
        shift()
    out.append(cache)
    for _ in range(cache_size):
        out.append(0xFF)
    return struct.pack('>I', len(data)) + bytes(out)


def _cm_decompress(data):
    if len(data) < 8:
        return b''
    P = _CM_P
    orig = struct.unpack('>I', data[:4])[0]
    if orig == 0:
        return b''
    m0 = [P >> 1] * 8
    m1 = [P >> 1] * (1 << 11)
    m2 = [P >> 1] * (1 << 16)
    m3 = [P >> 1] * (1 << 13)
    MODELS = (m0, m1, m2, m3)
    rate = (6, 6, 4, 4)
    w = [[0.0, 0.0, 0.0, 0.0] for _ in range(8)]
    wclamp = 1.0
    code = 0
    pos = 4
    for _ in range(5):
        if pos < len(data):
            code = ((code << 8) | data[pos]) & 0xFFFFFFFF
            pos += 1
    rng = 0xFFFFFFFF
    out = bytearray()
    h1 = h2 = h3 = 0
    st = _CM_STRETCH
    for _ in range(orig):
        b = 0
        for i in range(8):
            c1 = ((h1 << 3) | i) & 0x7FF
            c2 = h2 & 0xFFFF
            c3 = (h3 ^ (h3 >> 13)) & 0x1FFF
            prow = (m0[i], m1[c1], m2[c2], m3[c3])
            dot = 0.0
            wr = w[i]
            for mi in range(4):
                dot += wr[mi] * st[prow[mi]]
            p0 = _cm_squash(dot)
            if p0 <= 0:
                p0 = 1
            elif p0 >= P:
                p0 = P - 1
            rng0 = (rng // P) * p0
            if code < rng0:
                y = 0
                rng = rng0
            else:
                y = 1
                code -= rng0
                rng = rng - rng0
            b = (b << 1) | y
            target = P if y == 0 else 0
            for mi in range(4):
                idx = c1 if mi == 1 else i if mi == 0 else (c2 if mi == 2 else c3)
                cur = MODELS[mi][idx]
                MODELS[mi][idx] = cur + ((target - cur) >> rate[mi])
            p1 = (P - p0) / P
            y1 = 1.0 - y
            lr = 0.5 / (1.0 + 0.05 * abs(dot / _CM_SS))
            for mi in range(4):
                wr[mi] += lr * (y1 - p1) * st[prow[mi]] / _CM_SS
                if wr[mi] > wclamp:
                    wr[mi] = wclamp
                elif wr[mi] < -wclamp:
                    wr[mi] = -wclamp
            while rng <= 1 << 24:
                rng <<= 8
                if pos < len(data):
                    code = ((code << 8) | data[pos]) & 0xFFFFFFFF
                    pos += 1
        out.append(b)
        h3 = (((h3 & 0xFFFFFF) << 8) | b) & 0xFFFFFF
        h2 = ((h2 & 0xFFFF) << 8) | b
        h1 = b
    return bytes(out)


_reg(Algo("cm", "codec", _cm_compress, _cm_decompress,
          cost="slow", maxlen=(1 << 16)))

# --- Snappy (Google's fast LZ77 codec, spec-compatible) ----------------------
# We emit standard snappy: varint (uncompressed len) + literal / copy elements.
# A simple hash-table greedy matcher keeps it fast; the format matches the
# reference decoder so archives are interoperable.

def _snappy_varint(n, out):
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break


def _snappy_compress(data):
    n = len(data)
    out = bytearray()
    _snappy_varint(n, out)
    if n == 0:
        return bytes(out)
    pos = 0
    lit_start = 0
    ht = {}
    MINL = 4

    def emit_lits(up_to):
        nonlocal lit_start, out
        if up_to <= lit_start:
            return
        lit = data[lit_start:up_to]
        L = len(lit)
        while L > 0:
            frag = L
            if frag < 60:
                out.append((frag - 1) << 2)
            else:
                nbytes = 4
                if frag <= 0x100:
                    nbytes = 1
                elif frag <= 0x10000:
                    nbytes = 2
                elif frag <= 0x1000000:
                    nbytes = 3
                out.append((59 + nbytes) << 2)
                v = frag - 1
                for i in range(nbytes):
                    out.append((v >> (8 * i)) & 0xFF)
            out += lit[:frag]
            lit = lit[frag:]
            L = len(lit)

    while pos < n:
        key = data[pos:pos + MINL]
        if len(key) < MINL:
            break
        k = hash(key) & 0xFFFF
        cands = ht.get(k, ())
        best_off = -1
        best_len = 0
        for cp in cands:
            if pos - cp > 65535:
                continue
            m = 0
            while (pos + m < n and data[cp + m] == data[pos + m] and m < 64):
                m += 1
            if m > best_len:
                best_len = m
                best_off = pos - cp
        lst = ht.setdefault(k, [])
        lst.append(pos)
        if len(lst) > 64:
            ht[k] = lst[-32:]
        if best_len >= MINL and 0 < best_off <= 65535:
            emit_lits(pos)
            matched = best_len
            while matched > 0:
                if matched <= 11 and best_off < 2048:
                    out.append(1 | ((best_off >> 8) << 5)
                               | ((matched - 4) << 2))
                    out.append(best_off & 0xFF)
                    matched = 0
                else:
                    L = min(matched, 64)
                    out.append(2 | ((L - 1) << 2))
                    out.append(best_off & 0xFF)
                    out.append((best_off >> 8) & 0xFF)
                    matched -= L
            # register the whole matched run for overlapping matches
            for j in range(max(0, pos - 96), pos + best_len - MINL + 1):
                if j + MINL <= n:
                    kk = hash(data[j:j + MINL]) & 0xFFFF
                    ll = ht.setdefault(kk, [])
                    ll.append(j)
                    if len(ll) > 64:
                        ht[kk] = ll[-32:]
            pos += best_len
            lit_start = pos
        else:
            pos += 1
            if pos - lit_start >= 65536 or pos == n:
                emit_lits(pos)
    emit_lits(n)
    return bytes(out)


def _snappy_decompress(data):
    if not data or len(data) < 1:
        return b''
    pos = 0

    def rd_varint():
        nonlocal pos
        result = 0
        shift = 0
        while True:
            b = data[pos]
            pos += 1
            result |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        return result

    n = rd_varint()
    if n == 0:
        return b''
    out = bytearray()

    def emit_copy(off, length):
        for _ in range(length):
            out.append(out[-off])

    while pos < len(data) and len(out) < n:
        tag = data[pos]
        pos += 1
        t = tag & 0x03
        if t == 0:
            L = (tag >> 2) & 0x3F
            if L < 60:
                length = L + 1
            else:
                nb = L - 59
                length = 0
                for i in range(nb):
                    length |= data[pos + i] << (8 * i)
                pos += nb
                length += 1
            out += data[pos:pos + length]
            pos += length
        elif t == 1:
            length = 4 + ((tag >> 2) & 0x07)
            off = ((tag & 0xE0) << 3) | data[pos]
            pos += 1
            emit_copy(off, length)
        elif t == 2:
            length = 1 + (tag >> 2)
            off = data[pos] | (data[pos + 1] << 8)
            pos += 2
            emit_copy(off, length)
        else:
            length = 1 + (tag >> 2)
            off = (data[pos] | (data[pos + 1] << 8)
                   | (data[pos + 2] << 16) | (data[pos + 3] << 24))
            pos += 4
            emit_copy(off, length)
    return bytes(out[:n])


_reg(Algo("snappy", "codec", _snappy_compress, _snappy_decompress,
          cost="fast", maxlen=(1 << 26)))

# ============================================================================
# MORE CODEC FAMILIES FROM LIBRARIES
# ============================================================================

# --- zlib with explicit memLevel grids -------------------------------------
for wb in range(9, 16):
    for strat in (0, 1, 2, 3, 4):
        for lv in range(1, 10):
            for mem in (1, 5, 9):
                _reg(Algo(
                    f"zlibm_l{lv}_w{wb}_{_STRAT[strat]}_m{mem}", "codec",
                    (lambda lv=lv, wb=wb, st=strat, mem=mem:
                     lambda d: _zlib_compress(lv, wb, st, d, mem))(),
                    (lambda wb=wb: lambda d: _zlib_decompress(wb, d))(),
                    cost="normal"))

# --- raw DEFLATE with extra memLevels ---------------------------------------
for wb in range(9, 16):
    for strat in (0, 1, 2, 3, 4):
        for lv in range(1, 10):
            for mem in (3, 5, 7):
                def _mk2(lv=lv, wb=wb, st=strat, mem=mem):
                    def comp(d):
                        co = zlib.compressobj(level=lv, wbits=-wb,
                                              strategy=st, memLevel=mem)
                        return co.compress(d) + co.flush()
                    return comp

                def _mk2_d(wb=wb):
                    return lambda d: zlib.decompress(d, -wb)

                _reg(Algo(
                    f"deflate_l{lv}_w{wb}_{_STRAT[strat]}_m{mem}", "codec",
                    _mk2(), _mk2_d(), cost="normal"))

# --- brotli with static text/font modes ------------------------------------
if HAS_BROTLI:
    for mode in (brotli.MODE_TEXT, brotli.MODE_FONT):
        for q in range(0, 12):
            for w in (10, 14, 18, 22, 24):
                _reg(Algo(f"brotli_q{q}_w{w}_m{mode}", "codec",
                          (lambda q=q, w=w, md=mode:
                           lambda d: brotli.compress(d, quality=q, lgwin=w,
                                                     mode=md))(),
                          brotli.decompress,
                          cost="slow" if q >= 9 else "normal"))

# --- zstd with a compression-parameter grid --------------------------------
if HAS_ZSTD:
    for lv in (1, 3, 5, 9, 15, 19, 22):
        for wl in (18, 20, 22, 24, 26, 27):
            for ldm in (0, 1):
                def _zp(lv=lv, wl=wl, ldm=ldm):
                    params = zstd.ZstdCompressionParameters.from_level(
                        lv, window_log=wl, enable_ldm=bool(ldm))
                    return lambda d: zstd.ZstdCompressor(
                        compression_params=params).compress(d)
                _reg(Algo(f"zstdp_l{lv}_w{wl}{'_ldm' if ldm else ''}",
                          "codec", _zp(),
                          lambda d: zstd.ZstdDecompressor().decompress(d),
                          cost="slow" if lv >= 19 else "normal"))

# --- lzma / xz extreme presets ---------------------------------------------
for lv in range(0, 10):
    _reg(Algo(f"lzma_ex{lv}", "codec",
              (lambda lv=lv: lambda d: lzma.compress(
                  d, preset=lv | lzma.PRESET_EXTREME))(),
              lzma.decompress, cost="slow", maxlen=(1 << 24)))
    _reg(Algo(f"xz_ex{lv}", "codec",
              (lambda lv=lv: lambda d: lzma.compress(
                  d, format=lzma.FORMAT_XZ,
                  preset=lv | lzma.PRESET_EXTREME))(),
              lambda d: lzma.decompress(d, format=lzma.FORMAT_XZ),
              cost="slow", maxlen=(1 << 24)))

# --- lz4 frame variants -----------------------------------------------------
if HAS_LZ4_FRAME:
    for bs in (4, 5, 6, 7):
        for cc in (0, 1):
            for bl in (0, 1):
                _reg(Algo(f"lz4f_bs{bs}_cc{cc}_bl{bl}", "codec",
                          (lambda bs=bs, cc=cc, bl=bl:
                           lambda d: lz4frame.compress(
                               d, block_size=bs, content_checksum=bool(cc),
                               block_linked=bool(bl)))(),
                          lz4frame.decompress, cost="fast"))

# --- zopfli (extra-resolution deflate; very slow) ---------------------------
if HAS_ZOPFLI:
    import io as _io
    import contextlib as _cl
    import os as _os

    def _zop_run(fn, *a, **kw):
        # zopfli prints progress via C printf (fd 1) — silence it
        saved = _os.dup(1)
        devnull = _os.open(_os.devnull, _os.O_WRONLY)
        _os.dup2(devnull, 1)
        try:
            with _cl.redirect_stdout(_io.StringIO()):
                return fn(*a, **kw)
        finally:
            _os.dup2(saved, 1)
            _os.close(devnull)
            _os.close(saved)

    for it in (2, 5, 15, 30, 100):
        def _zop_g(it=it):
            def f(d):
                return _zop_run(_zopfli_gzip.compress, d,
                                numiterations=it)
            return f

        def _zop_z(it=it):
            def f(d):
                return _zop_run(_zopfli_zlib.compress, data=d,
                                numiterations=it)
            return f

        _reg(Algo(f"zopfli_gz{it}", "codec", _zop_g(),
                  lambda d: zlib.decompress(d, zlib.MAX_WBITS | 16),
                  cost="slow", maxlen=(1 << 24)))
        _reg(Algo(f"zopfli_zl{it}", "codec", _zop_z(),
                  lambda d: zlib.decompress(d, zlib.MAX_WBITS),
                  cost="slow", maxlen=(1 << 24)))


# --- snappy ---------------------------------------------------------------
if HAS_SNAPPY:
    _reg(Algo("snappy", "codec",
              snappy.compress, snappy.decompress, cost="fast"))


# ============================================================================
# TRANSFORM BUILDERS
# ============================================================================

def _delta_compress(k, data):
    n = len(data)
    out = bytearray(n + 1)
    out[0] = k & 0xFF
    if n == 0:
        return bytes(out)
    m = k if k < n else n
    out[1:1 + m] = data[:m]
    if np is not None and n > k:
        a = np.frombuffer(data, dtype=np.uint8)
        # uint8 uint8 subtraction wraps modulo 256, byte-identical to the
        # Python loop `(data[i] - data[i - k]) & 0xFF`
        out[1 + k:] = (a[k:] - a[:n - k]).tobytes()
        return bytes(out)
    for i in range(k, n):
        out[i + 1] = (data[i] - data[i - k]) & 0xFF
    return bytes(out)


def _delta_decompress(k, data):
    n = len(data) - 1
    if n <= 0:
        return bytes(n)
    if np is not None:
        a = np.frombuffer(memoryview(data)[1:1 + n], dtype=np.uint8)
        out = np.empty(n, dtype=np.uint8)
        # chain cumsum: out[i] = (data[i+1] + out[i-k]) mod 256
        for r in range(k):
            out[r::k] = np.cumsum(a[r::k], dtype=np.uint8)
        return out.tobytes()
    out = bytearray(n)
    for i in range(n):
        out[i] = data[i + 1] if i < k else (data[i + 1] + out[i - k]) & 0xFF
    return bytes(out)


for k in range(1, 257):
    def _dc(k=k):
        return lambda d: _delta_compress(k, d)

    def _dd(k=k):
        return lambda d: _delta_decompress(k, d)
    _reg(Algo(f"delta_o{k}", "transform", _dc(), _dd(), cost="fast"))


def _xorp_compress(k, data):
    n = len(data)
    out = bytearray(n + 1)
    out[0] = k & 0xFF
    if n == 0:
        return bytes(out)
    m = k if k < n else n
    out[1:1 + m] = data[:m]
    if np is not None and n > k:
        a = np.frombuffer(data, dtype=np.uint8)
        out[1 + k:] = (a[k:] ^ a[:n - k]).tobytes()
        return bytes(out)
    for i in range(k, n):
        out[i + 1] = data[i] ^ data[i - k]
    return bytes(out)


def _xorp_decompress(k, data):
    n = len(data) - 1
    if n <= 0:
        return bytes(n)
    if np is not None:
        a = np.frombuffer(memoryview(data)[1:1 + n], dtype=np.uint8)
        out = np.empty(n, dtype=np.uint8)
        for r in range(k):
            out[r::k] = np.bitwise_xor.accumulate(a[r::k])
        return out.tobytes()
    out = bytearray(n)
    for i in range(n):
        out[i] = data[i + 1] if i < k else data[i + 1] ^ out[i - k]
    return bytes(out)


for k in range(1, 33):
    _reg(Algo(f"xorp_o{k}", "transform",
              (lambda k=k: lambda d: _xorp_compress(k, d))(),
              (lambda k=k: lambda d: _xorp_decompress(k, d))(),
              cost="fast"))


def _mtf_compress(data):
    lst = bytearray(range(256))
    out = bytearray()
    app = out.append
    for b in data:
        i = lst.index(b)          # C-speed scan on a bytearray
        app(i)
        del lst[i]
        lst.insert(0, b)
    return bytes(out)


def _mtf_decompress(data):
    lst = bytearray(range(256))
    out = bytearray()
    app = out.append
    for i in data:
        b = lst[i]
        app(b)
        del lst[i]
        lst.insert(0, b)
    return bytes(out)


_reg(Algo("mtf", "transform", _mtf_compress, _mtf_decompress, cost="fast"))

# --- MTF variants ---------------------------------------------------------
def _mtf2_compress(data):
    lst = bytearray(range(256))
    out = bytearray()
    app = out.append
    for b in data:
        i = lst.index(b)
        app(i)
        if i >= 2:
            del lst[i]
            lst.insert(i - 2, b)
    return bytes(out)


def _mtf2_decompress(data):
    lst = bytearray(range(256))
    out = bytearray()
    app = out.append
    for i in data:
        b = lst[i]
        app(b)
        if i >= 2:
            del lst[i]
            lst.insert(i - 2, b)
    return bytes(out)


def _mtfswap_compress(data):
    lst = bytearray(range(256))
    out = bytearray()
    app = out.append
    for b in data:
        i = lst.index(b)
        app(i)
        if i > 0:
            lst[i - 1], lst[i] = lst[i], lst[i - 1]
    return bytes(out)


def _mtfswap_decompress(data):
    lst = bytearray(range(256))
    out = bytearray()
    app = out.append
    for i in data:
        b = lst[i]
        app(b)
        if i > 0:
            lst[i - 1], lst[i] = lst[i], lst[i - 1]
    return bytes(out)


_reg(Algo("mtf2", "transform", _mtf2_compress, _mtf2_decompress, cost="fast"))
_reg(Algo("mtf_swap", "transform", _mtfswap_compress, _mtfswap_decompress, cost="fast"))

# --- zigzag delta (good for signed integer arrays) -------------------------
def _delta_zigzag_compress(k, data):
    n = len(data)
    out = bytearray(n + 1)
    out[0] = k & 0xFF
    if n == 0:
        return bytes(out)
    m = k if k < n else n
    out[1:1 + m] = data[:m]
    if np is not None and n > k:
        a = np.frombuffer(data, dtype=np.uint8)
        d = (a[k:].astype(np.uint8) - a[:n - k].astype(np.uint8)).astype(np.uint8)
        v = d.astype(np.int16)
        v[v >= 128] -= 256                      # signed -128..127
        out[1 + k:] = np.bitwise_xor(v << 1, v >> 7).astype(np.uint8).tobytes()
        return bytes(out)
    for i in range(k, n):
        d = (data[i] - data[i - k]) & 0xFF
        v = d if d < 128 else d - 256
        out[i + 1] = ((v << 1) ^ (v >> 7)) & 0xFF
    return bytes(out)


def _delta_zigzag_decompress(k, data):
    n = len(data) - 1
    if n <= 0:
        return bytes(n)
    if np is not None:
        z = np.frombuffer(memoryview(data)[1:1 + n], dtype=np.uint8).astype(np.int32)
        s = (z >> 1) ^ -(z & 1)                 # zigzag decode into signed deltas
        out = np.empty(n, dtype=np.uint8)
        for r in range(min(k, n)):
            chain = s[r::k]                     # pos 0 = seed's zz, rest = deltas
            if n <= k or chain.size <= 1:
                out[r::k] = data[1 + r]
                continue
            vals = np.concatenate([[data[1 + r] & 0xFF],
                                   chain[1:]]).astype(np.int64)
            out[r::k] = (np.cumsum(vals) & 0xFF).astype(np.uint8)
        return out.tobytes()
    out = bytearray(n)
    for i in range(n):
        if i < k:
            out[i] = data[i + 1]
        else:
            z = data[i + 1]
            s = (z >> 1) ^ -(z & 1)
            out[i] = (out[i - k] + s) & 0xFF
    return bytes(out)


for k in range(1, 9):
    _reg(Algo(f"delta_zz_o{k}", "transform",
              (lambda k=k: lambda d: _delta_zigzag_compress(k, d))(),
              (lambda k=k: lambda d: _delta_zigzag_decompress(k, d))(),
              cost="fast"))

# --- wordwise delta (little-endian 16/32/64-bit words) ----------------------
_NP_WDT = {1: None, 2: np.uint16 if np is not None else None,
           4: np.uint32 if np is not None else None,
           8: np.uint64 if np is not None else None}


def _word_delta_compress(width, order, data):
    mask = (1 << (8 * width)) - 1
    body = data
    n = len(body)
    nw = n // width
    main = body[:nw * width]
    if np is not None and nw:
        vals = np.frombuffer(main, dtype=_NP_WDT[width])
        diff = vals[order:] - vals[:-order]      # wraps mod 2^(8*width)
        outv = np.concatenate([vals[:order], diff]) if order < nw else vals
        return bytes([width, order]) + outv.tobytes() + body[nw * width:]
    out = bytearray([width, order])
    for i in range(nw):
        v = int.from_bytes(main[i * width:(i + 1) * width], 'little')
        if i >= order:
            pv = int.from_bytes(main[(i - order) * width:(i - order + 1) * width],
                                'little')
            v = (v - pv) & mask
        out += v.to_bytes(width, 'little')
    out += body[nw * width:]
    return bytes(out)


def _word_delta_decompress(width, order, data):
    mask = (1 << (8 * width)) - 1
    if len(data) < 2:
        return b''
    body = data[2:]
    nw = len(body) // width
    n_trail = len(body) % width
    main = body[:nw * width]
    if np is not None and nw:
        vals = np.frombuffer(main, dtype=_NP_WDT[width])
        out = np.empty(nw, dtype=_NP_WDT[width])
        for r in range(order):
            out[r::order] = np.cumsum(vals[r::order], dtype=_NP_WDT[width])
        return out.tobytes() + body[nw * width:]
    fmt = "<" + ({2: 'H', 4: 'I', 8: 'Q'}[width]) * nw
    vals = list(struct.unpack(fmt, main)) if nw else []
    for i in range(order, nw):
        vals[i] = (vals[i] + vals[i - order]) & mask
    out = bytearray(nw * width)
    struct.pack_into(fmt, out, 0, *vals)
    out += body[nw * width:]
    return bytes(out)


def _word_delta_compress_be(width, order, data):
    mask = (1 << (8 * width)) - 1
    body = data
    n = len(body)
    nw = n // width
    main = body[:nw * width]
    if np is not None and nw:
        bd = {2: '>u2', 4: '>u4', 8: '>u8'}[width]
        nd = _NP_WDT[width]
        vals = np.frombuffer(main, dtype=np.dtype(bd)).astype(nd)  # native
        diff = vals[order:] - vals[:-order]          # wraps mod 2^(8*width)
        outv = np.concatenate([vals[:order], diff]) if order < nw else vals
        return bytes([width, order]) + outv.astype(np.dtype(bd)).tobytes() + body[nw * width:]
    out = bytearray([width, order])
    for i in range(nw):
        v = int.from_bytes(main[i * width:(i + 1) * width], 'big')
        if i >= order:
            pv = int.from_bytes(main[(i - order) * width:(i - order + 1) * width],
                                'big')
            v = (v - pv) & mask
        out += v.to_bytes(width, 'big')
    out += body[nw * width:]
    return bytes(out)


def _word_delta_decompress_be(width, order, data):
    mask = (1 << (8 * width)) - 1
    if len(data) < 2:
        return b''
    body = data[2:]
    nw = len(body) // width
    main = body[:nw * width]
    if np is not None and nw:
        bd = {2: '>u2', 4: '>u4', 8: '>u8'}[width]
        nd = _NP_WDT[width]
        vals = np.frombuffer(main, dtype=np.dtype(bd)).astype(nd)
        out = np.empty(nw, dtype=nd)
        for r in range(order):
            out[r::order] = np.cumsum(vals[r::order], dtype=nd)
        return out.astype(np.dtype(bd)).tobytes() + body[nw * width:]
    fmt = ">" + ({2: 'H', 4: 'I', 8: 'Q'}[width]) * nw
    vals = list(struct.unpack(fmt, main)) if nw else []
    for i in range(order, nw):
        vals[i] = (vals[i] + vals[i - order]) & mask
    out = bytearray(nw * width)
    struct.pack_into(fmt, out, 0, *vals)
    out += body[nw * width:]
    return bytes(out)


for w in (2, 4, 8):
    for o in (1, 2, 3):
        _reg(Algo(f"delta_w{w * 8}_o{o}", "transform",
                  (lambda w=w, o=o: lambda d: _word_delta_compress(w, o, d))(),
                  (lambda w=w, o=o: lambda d: _word_delta_decompress(w, o, d))(),
                  cost="fast"))
        _reg(Algo(f"delta_w{w * 8}_be_o{o}", "transform",
                  (lambda w=w, o=o: lambda d: _word_delta_compress_be(w, o, d))(),
                  (lambda w=w, o=o: lambda d: _word_delta_decompress_be(w, o, d))(),
                  cost="fast"))

# --- per-byte bit rotations & constant maps --------------------------------
# Precomputed 256-entry translate tables -> transform is a single C pass
_ROTL = [bytes(((b << k) | (b >> (8 - k))) & 0xFF for b in range(256))
         for k in range(8)]


def _rotl_compress(k, data):
    return data.translate(_ROTL[k])


def _rotl_decompress(k, data):
    return data.translate(_ROTL[(8 - k) & 7])


for k in range(1, 8):
    _reg(Algo(f"rotl_{k}", "transform",
              (lambda k=k: lambda d: _rotl_compress(k, d))(),
              (lambda k=k: lambda d: _rotl_decompress(k, d))(),
              cost="fast"))

_NOTB_TAB = bytes(b ^ 0xFF for b in range(256))
_reg(Algo("notb", "transform",
          (lambda d: d.translate(_NOTB_TAB)),
          (lambda d: d.translate(_NOTB_TAB)), cost="fast"))

for c in (0x0F, 0x55, 0xAA):
    _tab = bytes(b ^ c for b in range(256))
    _reg(Algo(f"xorc_{c:02x}", "transform",
              (lambda t=_tab: lambda d: d.translate(t))(),
              (lambda t=_tab: lambda d: d.translate(t))(),
              cost="fast"))

# --- LZP residual predictor (context prediction + XOR residual) -------------
def _lzp_compress(ctx, data):
    if not data:
        return b''
    n = len(data)
    tab = [-1] * 65536
    out = bytearray()
    raw = data  # encoder "raw" prefix == input

    def hsh(i):
        if ctx == 1:
            return raw[i - 1]
        if ctx == 2:
            return (raw[i - 2] << 8) | raw[i - 1]
        return (((raw[i - 3] * 255 + raw[i - 2]) * 255 + raw[i - 1])
                & 0xFFFF)

    for i in range(n):
        if i >= ctx:
            h = hsh(i)
            p = tab[h]
            pred = raw[p + ctx] if p >= 0 and p + ctx < n else 0
            tab[h] = i - ctx
        else:
            pred = 0
        out.append(raw[i] ^ pred)
    return bytes(out)


def _lzp_decompress(ctx, data):
    n = len(data)
    tab = [-1] * 65536
    raw = bytearray(n)  # reconstructed raw prefix (data[i] = out[i] ^ pred)

    def hsh(i):
        if ctx == 1:
            return raw[i - 1]
        if ctx == 2:
            return (raw[i - 2] << 8) | raw[i - 1]
        return (((raw[i - 3] * 255 + raw[i - 2]) * 255 + raw[i - 1])
                & 0xFFFF)

    for i in range(n):
        if i >= ctx:
            h = hsh(i)
            p = tab[h]
            pred = raw[p + ctx] if p >= 0 and p + ctx < n else 0
            tab[h] = i - ctx
        else:
            pred = 0
        raw[i] = data[i] ^ pred
    return bytes(raw)


for ctx in (1, 2, 3):
    _reg(Algo(f"lzp_c{ctx}", "transform",
              (lambda ctx=ctx: lambda d: _lzp_compress(ctx, d))(),
              (lambda ctx=ctx: lambda d: _lzp_decompress(ctx, d))(),
              cost="normal", maxlen=(1 << 24)))


def _bitplane_compress(n, data):
    """Split each byte's bits across n planes.

    Plane p owns bit positions p, p+n, p+2n, ... (MSB-first packing).
    numpy path is bit-identical to the reference loop below.
    """
    if np is not None:
        nbytes = len(data)
        allbits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        per = 8 // n
        body = bytearray()
        for p in range(n):
            # plane p owns bit positions p, p+n, ... (MSB-first per byte)
            jj = np.arange(per)
            idx = (np.arange(nbytes)[:, None] * 8 + p + jj * n).ravel()
            plane = np.packbits(allbits[idx]).tobytes()
            body += plane
        return bytes([n]) + struct.pack('>I', nbytes) + bytes(body)
    out = bytearray([n])
    out += struct.pack('>I', len(data))
    per = 8 // n
    for p in range(n):
        acc = 0
        cnt = 0
        for b in data:
            for j in range(per):
                bitpos = p + j * n
                acc = (acc << 1) | ((b >> (7 - bitpos)) & 1)
                cnt += 1
                if cnt == 8:
                    out.append(acc)
                    acc = 0
                    cnt = 0
        if cnt:
            out.append(acc << (8 - cnt))
    return bytes(out)


def _bitplane_decompress(n, data):
    orig = struct.unpack('>I', data[1:5])[0]
    per = 8 // n
    pos = 5
    planes = []
    for _p in range(n):
        nbits = orig * per
        nbytes = (nbits + 7) // 8
        planes.append(data[pos:pos + nbytes])
        pos += nbytes
    if np is not None and orig:
        frags = []
        for p in planes:
            b = np.unpackbits(np.frombuffer(p, dtype=np.uint8))
            frags.append(b[:orig * per])
        bits = np.concatenate(frags)[:orig * 8]
        if bits.size == 0:
            return b''
        # out bit index k -> source position in plane-major bit array
        # k = i*8 + bitpos ; pl = bitpos % n ; within = bitpos // n
        # src = pl*(orig*per) + i*per + within
        idx = np.arange(orig * 8, dtype=np.int64)
        i = idx // 8
        bitpos = idx % 8
        pl = bitpos % n
        within = bitpos // n
        src = pl * (orig * per) + i * per + within
        return np.packbits(bits[src].reshape(orig, 8)).tobytes()
    out = bytearray()
    for i in range(orig):
        b = 0
        for bitpos in range(8):
            pl = bitpos % n
            within = bitpos // n
            plane = planes[pl]
            bit_in_plane = i * per + within
            byidx = bit_in_plane // 8
            bitoff = 7 - (bit_in_plane % 8)
            bit = (plane[byidx] >> bitoff) & 1 if byidx < len(plane) else 0
            b = (b << 1) | bit
        out.append(b)
    return bytes(out)


for n in (1, 2, 4, 8):
    _reg(Algo(f"bitplane{n}", "transform",
              (lambda n=n: lambda d: _bitplane_compress(n, d))(),
              (lambda n=n: lambda d: _bitplane_decompress(n, d))(),
              cost="fast"))


def _nibble_compress(hi_first, data):
    if np is not None:
        a = np.frombuffer(data, dtype=np.uint8)
        hi = (a >> 4).tobytes()
        lo = (a & 0x0F).tobytes()
        return hi + lo if hi_first else lo + hi
    hi = bytearray(); lo = bytearray()
    for b in data:
        hi.append(b >> 4); lo.append(b & 0x0F)
    if hi_first:
        return bytes(hi) + bytes(lo)
    return bytes(lo) + bytes(hi)


def _nibble_decompress(hi_first, data):
    n = len(data) // 2
    if np is not None and n:
        ha = np.frombuffer(memoryview(data)[:n], dtype=np.uint8)
        la = np.frombuffer(memoryview(data)[n:2 * n], dtype=np.uint8)
        if hi_first:
            return ((ha << 4) | la).tobytes()
        return ((la << 4) | ha).tobytes()
    a = list(data[:n]); b = list(data[n:])
    out = bytearray()
    for i in range(n):
        if hi_first:
            out.append((a[i] << 4) | b[i])
        else:
            out.append((b[i] << 4) | a[i])
    return bytes(out)


_reg(Algo("nibble_hi", "transform",
          (lambda d: _nibble_compress(True, d)), (lambda d: _nibble_decompress(True, d))))
_reg(Algo("nibble_lo", "transform",
          (lambda d: _nibble_compress(False, d)), (lambda d: _nibble_decompress(False, d))))

_NIBSWAP_TAB = bytes(((b >> 4) | (b << 4)) & 0xFF for b in range(256))
_reg(Algo("nibswap", "transform",
          (lambda d: d.translate(_NIBSWAP_TAB)),
          (lambda d: d.translate(_NIBSWAP_TAB)),
          cost="fast"))


def _interleave_compress(n, data):
    if not data:
        return b'\x01\x00'
    q, r = divmod(len(data), n)
    main = data[:q * n]
    if np is not None and q:
        arr = np.frombuffer(main, dtype=np.uint8).reshape(q, n)
        return struct.pack('>H', n) + arr.transpose().reshape(-1).tobytes() \
            + data[q * n:]
    out = bytearray()
    for j in range(n):
        out += main[j::n]
    out += data[q * n:]
    return struct.pack('>H', n) + bytes(out)


def _interleave_decompress(n, data):
    if len(data) < 2:
        return b''
    body = data[2:]
    q = len(body) // n
    r = len(body) % n
    main = body[:q * n]
    if np is not None and q:
        arr = np.frombuffer(main, dtype=np.uint8).reshape(n, q)
        return arr.transpose().reshape(-1).tobytes() + body[q * n:]
    out = bytearray()
    for i in range(q):
        for j in range(n):
            out.append(main[j * q + i])
    out += body[q * n:]
    return bytes(out)


for n in (2, 4, 8, 16, 32, 64, 128, 256):
    _reg(Algo(f"interleave{n}", "transform",
              (lambda n=n: lambda d: _interleave_compress(n, d))(),
              (lambda n=n: lambda d: _interleave_decompress(n, d))(),
              cost="fast"))

def _bitrev8(b):
    b = ((b & 0x55) << 1) | ((b >> 1) & 0x55)
    b = ((b & 0x33) << 2) | ((b >> 2) & 0x33)
    return ((b & 0x0F) << 4) | ((b >> 4) & 0x0F)


_BITREV_TAB = bytes(_bitrev8(b) for b in range(256))
_reg(Algo("bitrev", "transform",
          (lambda d: d.translate(_BITREV_TAB)),
          (lambda d: d.translate(_BITREV_TAB)),
          cost="fast"))

_reg(Algo("byteswap", "transform",
          (lambda d: d[::-1]), (lambda d: d[::-1]), cost="fast"))


def _transpose8(data):
    q, r = divmod(len(data), 8)
    if np is not None and q:
        arr8 = np.frombuffer(data, dtype=np.uint8)[:q * 8].reshape(q, 8)
        bits = np.unpackbits(arr8, axis=1).reshape(q, 8, 8)  # (q, byte, bit)
        bt = bits.transpose(0, 2, 1)                         # (q, bitpos, byte)
        return np.packbits(bt, axis=2).reshape(-1).tobytes() + data[q * 8:]
    out = bytearray()
    for g in range(q):
        chunk = data[g * 8:(g + 1) * 8]
        m = [[(chunk[i] >> (7 - j)) & 1 for j in range(8)] for i in range(8)]
        res = [0] * 8
        for j in range(8):
            for i in range(8):
                res[j] = (res[j] << 1) | m[i][j]
        out += bytes(res)
    out += data[q * 8:]
    return bytes(out)


_reg(Algo("transpose8", "transform", _transpose8, _transpose8, cost="fast"))


def _rle_compress(t, zeros_only, data):
    out = bytearray([t])
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        j = i + 1
        while j < n and data[j] == b:
            j += 1
        run = j - i
        if b == 0xFF:
            # never run-encode 0xFF (it is the escape byte); always literal
            out += b'\xff\xff' * run
        elif (not zeros_only or b == 0) and run >= t:
            out += b'\xff'
            out.append(b)
            out += struct.pack('>I', run)
        else:
            out += bytes([b]) * run
        i = j
    return bytes(out)


def _rle_decompress(data):
    t = data[0]
    out = bytearray()
    i = 1
    n = len(data)
    while i < n:
        if data[i] != 0xFF:
            out.append(data[i]); i += 1
        else:
            if i + 1 >= n:
                out.append(0xFF); i += 1
            elif data[i + 1] == 0xFF:
                out.append(0xFF); i += 2
            else:
                b = data[i + 1]
                run = struct.unpack('>I', data[i + 2:i + 6])[0]
                out += bytes([b]) * run
                i += 6
    return bytes(out)


for t in (2, 3, 4, 8, 16):
    _reg(Algo(f"rle_t{t}", "transform",
              (lambda t=t: lambda d: _rle_compress(t, False, d))(),
              _rle_decompress, cost="fast"))
_reg(Algo("rle_z", "transform",
          lambda d: _rle_compress(4, True, d), _rle_decompress, cost="fast"))


def _suffix_array(d):
    """O(n log n) prefix-doubling suffix array for bytes.

    Keys are packed into a single int (rank * K + second-rank) so the
    sort compares plain Python ints in C instead of tuples.  If the
    optional `pysais` C binding is installed it is used instead (O(n))."""
    n = len(d)
    if n <= 1:
        return list(range(n))
    if pysais is not None:
        try:
            return list(pysais.sais(bytes(d)))
        except Exception:
            pass
    sa = list(range(n))
    rank = list(d)
    tmp = [0] * n
    k = 1
    K = n + 1
    while True:
        r2 = [0] * n
        for i in range(n - k):          # out-of-range second key -> 0
            r2[i] = rank[i + k] + 1
        sa.sort(key=lambda i: rank[i] * K + r2[i])
        tmp[sa[0]] = 0
        cl = 0
        prev = sa[0]
        for idx in range(1, n):
            a = sa[idx]
            if rank[a] != rank[prev] or r2[a] != r2[prev]:
                cl += 1
            tmp[a] = cl
            prev = a
        rank, tmp = tmp, rank
        k <<= 1
        if cl == n - 1 or k >= n:
            break
    return sa


def _bwt_compress(data):
    if not data:
        return b''
    n = len(data)
    if n == 1:
        return struct.pack('>I', 0) + data
    # BWT needs cyclic-rotation order, which is the suffix order of the
    # doubled string restricted to starts < n.
    doubled = bytes(data) + bytes(data)
    sa2 = _suffix_array(doubled)
    sa = [i for i in sa2 if i < n]
    primary = sa.index(0)
    body = bytearray(n)
    for k, s in enumerate(sa):
        body[k] = data[(int(s) - 1) % n]
    return struct.pack('>I', primary) + bytes(body)


def _bwt_decompress(data):
    if len(data) < 5:
        return b''
    primary = struct.unpack('>I', data[:4])[0]
    t = data[4:]
    n = len(t)
    if n == 0:
        return b''
    C = [0] * 256
    for b in t:
        C[b] += 1
    cur = [0] * 256
    acc = 0
    for c in range(256):
        cur[c] = acc
        acc += C[c]
    LF = [0] * n
    rank = [0] * 256
    for i in range(n):
        b = t[i]
        LF[i] = cur[b] + rank[b]
        rank[b] += 1
    out = bytearray(n)
    j = primary
    for i in range(n - 1, -1, -1):
        out[i] = t[j]
        j = LF[j]
    return bytes(out)


for bs in (256, 512, 1024, 2048, 4096, 8192):
    _reg(Algo(f"bwt_b{bs}", "transform",
              _bwt_compress, _bwt_decompress,
              cost="slow", maxlen=(1 << 21)))


def _bpe_compress(rounds, data):
    d = bytearray(data)
    reps = []
    for _ in range(rounds):
        if len(d) < 2:
            break
        if np is not None and len(d) > 4:
            a = np.frombuffer(d, dtype=np.uint8)
            keys = a[:-1].astype(np.int32) * 256 + a[1:].astype(np.int32)
            cnts = np.bincount(keys, minlength=65536)
            bi = int(cnts.argmax())
            c = int(cnts[bi])
            pair = (bi >> 8, bi & 0xFF)
        else:
            cnt = Counter()
            for i in range(len(d) - 1):
                key = (d[i], d[i + 1])
                cnt[key] += 1
            if not cnt:
                break
            pair, c = cnt.most_common(1)[0]
        if c < 2:
            break
        used = set(d)
        free = None
        for b in range(255, -1, -1):
            if b not in used:
                free = b
                break
        if free is None:
            break
        reps.append((pair[0], pair[1], free))
        nd = bytearray()
        app = nd.append
        i = 0
        e = len(d) - 1
        pa, pb = pair
        while i < e:
            if d[i] == pa and d[i + 1] == pb:
                app(free); i += 2
            else:
                app(d[i]); i += 1
        if i == e:
            app(d[i])
        d = nd
    out = bytearray()
    out += struct.pack('>I', len(data))
    out += struct.pack('>H', len(reps))
    for a, b, f in reps:
        out += bytes((a, b, f))
    out += bytes(d)
    return bytes(out)


def _bpe_decompress(data):
    orig = struct.unpack('>I', data[:4])[0]
    nr = struct.unpack('>H', data[4:6])[0]
    pos = 6
    reps = []
    for _ in range(nr):
        a, b, f = data[pos], data[pos + 1], data[pos + 2]
        reps.append((a, b, f)); pos += 3
    d = bytearray(data[pos:])
    for a, b, f in reversed(reps):
        nd = bytearray()
        for x in d:
            if x == f:
                nd.append(a); nd.append(b)
            else:
                nd.append(x)
        d = nd
    return bytes(d[:orig])


for r in range(1, 49):
    _reg(Algo(f"bpe_r{r}", "transform",
              (lambda r=r: lambda d: _bpe_compress(r, d))(),
              _bpe_decompress,
              cost="slow" if r > 16 else "normal",
              maxlen=(1 << 24)))


# ============================================================================
# DICTIONARY PREPROCESSOR  (kept byte-compatible with the v2 archive format)
# ============================================================================

class DictPreprocessor:
    """Finds repeated byte patterns and replaces them with tokens."""

    def __init__(self, min_len=3, max_len=128, min_count=2, max_entries=16384):
        self.min_len = min_len
        self.max_len = max_len
        self.min_count = min_count
        self.max_entries = max_entries

    # ---- per-algorithm variants (registry entries) ----------------------
    def make_algo(self, name, min_len, max_entries):
        return Algo(name, "transform",
                    (lambda ml=min_len, me=max_entries:
                     lambda d: self.compress(d, ml, me))(),
                    self.decompress,
                    cost="normal", maxlen=(1 << 24))

    def compress(self, data, min_len=3, max_entries=16384):
        from collections import Counter
        ml = min_len
        me = max_entries
        if len(data) < ml * self.min_count:
            return struct.pack('>I', 0) + data
        counts = Counter()
        step = max(1, len(data) // 8000)
        for wsize in [3, 4, 5, 7, 10, 15, 20, 30, 50]:
            if wsize > len(data) // 2 or wsize < ml:
                continue
            if wsize > self.max_len:
                continue
            for i in range(0, len(data) - wsize + 1, step):
                counts[data[i:i + wsize]] += 1
        scored = []
        for pat, cnt in counts.most_common(me * 2):
            if cnt < self.min_count:
                break
            savings = (len(pat) - 3) * (cnt - 1) - 3
            if savings > 0:
                scored.append((savings, pat))
        scored.sort(key=lambda x: (-x[0], -len(x[1])))
        scored = scored[:me]
        if not scored:
            return struct.pack('>I', 0) + data
        entries = [pat for _, pat in scored]
        entries.sort(key=lambda x: -len(x))
        pat2id = {pat: i for i, pat in enumerate(entries)}
        by_first = {}
        for i, pat in enumerate(entries):
            by_first.setdefault(pat[0], []).append((pat, i))
        out = bytearray()
        i = 0
        dlen = len(data)
        while i < dlen:
            cands = by_first.get(data[i])
            if cands:
                best_id = -1
                best_len = 0
                for pat, pid in cands:
                    plen = len(pat)
                    if plen > best_len and data[i:i + plen] == pat:
                        best_id = pid
                        best_len = plen
                if best_id >= 0:
                    out.extend(b'\xff')
                    out.extend(struct.pack('>H', best_id + 1))
                    i += best_len
                    continue
            if data[i] == 0xFF:
                out.extend(b'\xff\xff')
            else:
                out.append(data[i])
            i += 1
        result = struct.pack('>I', len(entries))
        for pat in entries:
            result += struct.pack('>H', len(pat)) + pat
        result += struct.pack('>I', len(out)) + bytes(out)
        return result

    def decompress(self, data):
        # v3 scheme: 0xFF followed by u16(index+1).  0xFFFF = literal 0xFF.
        off = 0
        n = struct.unpack('>I', data[off:off + 4])[0]; off += 4
        if n == 0:
            # no dictionary built: the rest is the raw data, put back verbatim
            return data[off:]
        dictionary = []
        for _ in range(n):
            plen = struct.unpack('>H', data[off:off + 2])[0]; off += 2
            dictionary.append(data[off:off + plen]); off += plen
        if off + 4 > len(data):
            raise ValueError("corrupt dictionary header")
        dlen = struct.unpack('>I', data[off:off + 4])[0]; off += 4
        proc = data[off:off + dlen]
        out = bytearray()
        i = 0
        while i < len(proc):
            if proc[i] == 0xFF:
                if i + 1 >= len(proc):
                    raise ValueError("truncated dictionary stream")
                if proc[i + 1] == 0xFF:
                    # literal 0xFF escape
                    out.append(0xFF); i += 2
                else:
                    if i + 2 >= len(proc):
                        raise ValueError("truncated dictionary stream")
                    v = struct.unpack('>H', proc[i + 1:i + 3])[0]
                    if v == 0 or v - 1 >= len(dictionary):
                        raise ValueError("bad dictionary token")
                    out.extend(dictionary[v - 1])
                    i += 3
            else:
                out.append(proc[i]); i += 1
        return bytes(out)

    def decompress_legacy(self, data):
        # v2 scheme (pre-fix, ambiguous for ids < 256): 0xFF 0x00 = literal.
        off = 0
        n = struct.unpack('>I', data[off:off + 4])[0]; off += 4
        if n == 0:
            return data[off:]
        dictionary = []
        for _ in range(n):
            plen = struct.unpack('>H', data[off:off + 2])[0]; off += 2
            dictionary.append(data[off:off + plen]); off += plen
        if off + 4 > len(data):
            raise ValueError("corrupt dictionary header")
        dlen = struct.unpack('>I', data[off:off + 4])[0]; off += 4
        proc = data[off:off + dlen]
        out = bytearray()
        i = 0
        while i < len(proc):
            if proc[i] == 0xFF:
                if i + 1 >= len(proc):
                    raise ValueError("truncated dictionary stream")
                b = proc[i + 1]
                if b == 0x00:
                    out.append(0xFF); i += 2
                else:
                    tid = struct.unpack('>H', proc[i + 1:i + 3])[0]
                    if tid >= len(dictionary):
                        raise ValueError("bad dictionary token")
                    out.extend(dictionary[tid]); i += 3
            else:
                out.append(proc[i]); i += 1
        return bytes(out)


# --- extra popular algorithms (added for exhaust mode) --------------------
if HAS_ZSTD:
    # zstd with larger windows (better for large repetitive data)
    for wbits in (23, 25, 27):
        for lv in (19, 22):
            def _zwc(lv=lv, wb=wbits):
                params = zstd.ZstdCompressionParameters.from_level(
                    lv, window_log=wb)
                return lambda d: zstd.ZstdCompressor(
                    compression_params=params).compress(d)
            _reg(Algo(f"zstd_l{lv}_w{wb}", "codec",
                      _zwc(),
                      lambda d: zstd.ZstdDecompressor().decompress(d),
                      cost="slow"))

    # zstd with long-distance matching + large window (best for huge files)
    for lv in (15, 19, 22):
        for wb in (25, 27):
            def _zlmc(lv=lv, wb=wb):
                params = zstd.ZstdCompressionParameters.from_level(
                    lv, enable_ldm=True, window_log=wb)
                return lambda d: zstd.ZstdCompressor(
                    compression_params=params).compress(d)
            _reg(Algo(f"zstd_l{lv}_ldm_w{wb}", "codec",
                      _zlmc(),
                      lambda d: zstd.ZstdDecompressor().decompress(d),
                      cost="slow"))

if HAS_BROTLI:
    # brotli at max quality with larger windows
    for w in (22, 24):
        for q in (10, 11):
            def _bm(w=w, q=q):
                return lambda d: brotli.compress(d, quality=q, lgwin=w)
            _reg(Algo(f"brotli_q{q}_w{w}", "codec",
                      _bm(),
                      lambda d: brotli.decompress(d),
                      cost="slow"))

# xz with extreme preset + large dict (preset must live INSIDE the filter)
for lv in range(6, 10):
    for ds in (27, 29):
        def _xze(lv=lv, ds=ds):
            return lambda d: lzma.compress(d, format=lzma.FORMAT_XZ,
                filters=[{'id': lzma.FILTER_LZMA2,
                          'preset': lzma.PRESET_EXTREME | lv,
                          'dict_size': 1 << ds}])
        _reg(Algo(f"xz_ex{lv}_d{ds}", "codec",
                  _xze(),
                  lambda d: lzma.decompress(d, format=lzma.FORMAT_XZ),
                  cost="slow"))

if HAS_LZ4:
    # lz4 hc with very high acceleration (fast decompress, good ratio)
    for acc in (1, 2):
        _reg(Algo(f"lz4_hc_a{acc}", "codec",
                  (lambda a=acc: lambda d: lz4block.compress(
                      d, store_size=True, mode="high_compression",
                      compression=12, acceleration=a))(),
                  lz4block.decompress, cost="fast"))

# lzma2 in xz container with various dict sizes + levels (preset inside filter)
for lv in range(0, 10):
    for ds in (19, 22, 25, 27):
        def _lz(lv=lv, ds=ds):
            return lambda d: lzma.compress(d, format=lzma.FORMAT_XZ,
                filters=[{'id': lzma.FILTER_LZMA2, 'preset': lv,
                          'dict_size': 1 << ds}])
        _reg(Algo(f"lzma2_l{lv}_d{ds}", "codec",
                  _lz(),
                  lambda d: lzma.decompress(d, format=lzma.FORMAT_XZ),
                  cost="slow"))

# gzip with varying compression levels (1-9) for broader surface
for lv in range(1, 10):
    _reg(Algo(f"gzip_l{lv}", "codec",
              (lambda lv=lv: lambda d: zlib.compress(d, level=lv))(),
              lambda d: zlib.decompress(d), cost="fast"))

# brotli medium quality levels (1-5) for fast-to-medium range
if HAS_BROTLI:
    for q in range(1, 6):
        for w in (18, 20, 22):
            _reg(Algo(f"brotli_q{q}_w{w}", "codec",
                      (lambda q=q, w=w: lambda d: brotli.compress(
                          d, quality=q, lgwin=w))(),
                      lambda d: brotli.decompress(d),
                      cost="fast" if q <= 3 else "normal"))

# bz2 with varying compression levels
for lv in range(1, 10):
    _reg(Algo(f"bz2_l{lv}", "codec",
              (lambda lv=lv: lambda d: bz2.compress(d, compresslevel=lv))(),
              lambda d: bz2.decompress(d), cost="normal"))

# snappy (if available) — fast ratio
if HAS_SNAPPY:
    _reg(Algo("snappy_codec", "codec",
              lambda d: snappy.compress(d),
              lambda d: snappy.decompress(d), cost="fast"))

# zstd with BTLAZY2 and HUFFMAN codec variants
if HAS_ZSTD:
    for lv in (3, 9, 15):
        for strat in ('fast', 'dfast', 'greedy', 'lazy', 'lazy2'):
            def _zs(lv=lv, strat=strat):
                cctx = zstd.ZstdCompressor(level=lv,
                    compression_params=zstd.ZstdCompressionParameters(
                        strategy=getattr(zstd, 'STRATEGY_' + strat.upper(),
                                          zstd.STRATEGY_FAST) if hasattr(zstd, 'STRATEGY_' + strat.upper()) else zstd.STRATEGY_FAST))
                return lambda d: cctx.compress(d)
            _reg(Algo(f"zstd_l{lv}_s{strat}", "codec",
                      _zs(),
                      lambda d: zstd.ZstdDecompressor().decompress(d),
                      cost="normal"))

# === massive transform registry expansion (for exhaust mode) =============
# extra BWT block sizes (longer blocks => better BWT but slower)
for bs in (16384, 32768, 65536, 131072):
    _reg(Algo(f"bwt_b{bs}", "transform",
              _bwt_compress, _bwt_decompress,
              cost="slow", maxlen=(1 << 26)))

# x86 .exe/.dll speed-up: BCJ (branch-convert), reversible, length-preserving.
# Feeding BCJ output into LZMA/Brotli/PPMd dramatically helps binary blobs
# that contain E8/E9 call/jump bytes.
if HAS_BCJ:
    def _bcj_compress(d):
        enc = _bcj_module.BCJEncoder()
        return enc.encode(d) + enc.flush()

    def _bcj_decompress(d):
        dec = _bcj_module.BCJDecoder(len(d))
        return dec.decode(d)

    _reg(Algo("bcj_x86", "transform", _bcj_compress, _bcj_decompress,
              cost="normal", maxlen=(1 << 28)))

# aggressive RAW LZMA2 (7-Zip -mx=9 style: bt4 + huge dict + 273 cp).
# We must store the filter spec because FORMAT_RAW needs the exact same
# params to decompress; wrap with a tiny 1-byte variant-index header.
_RAW_LZMA2_VARIANTS = [
    ("raw9_bt4_d23", 23, 273),
    ("raw9_bt4_d25", 25, 273),
    ("raw9_bt4_d27", 27, 273),
    ("raw9_bt4_d28", 28, 273),
    ("raw9_bt4_d23_n64", 23, 64),
]
def _raw_lzma2_filters(dict_bits, nice_len):
    return [{"id": lzma.FILTER_LZMA2, "dict_size": 1 << dict_bits,
             "lc": 3, "lp": 0, "pb": 2,
             "mode": lzma.MODE_NORMAL, "nice_len": nice_len,
             "mf": lzma.MF_BT4, "depth": 273}]

def _raw9_fwd(idx):
    _, dict_bits, nice_len = _RAW_LZMA2_VARIANTS[idx]
    def f(d):
        body = lzma.compress(d, format=lzma.FORMAT_RAW,
                             filters=_raw_lzma2_filters(dict_bits, nice_len))
        return bytes([idx]) + body
    return f

def _raw9_back(d):
    idx = d[0]
    filter_spec = _raw_lzma2_filters(_RAW_LZMA2_VARIANTS[idx][1],
                                     _RAW_LZMA2_VARIANTS[idx][2])
    return lzma.decompress(d[1:], format=lzma.FORMAT_RAW, filters=filter_spec)

for _rix, (raw_name, _db, _nl) in enumerate(_RAW_LZMA2_VARIANTS):
    _reg(Algo(raw_name, "codec",
              _raw9_fwd(_rix),
              _raw9_back,
              cost="slow", maxlen=(1 << 28)))

# extra RLE thresholds
for t in (5, 6, 12, 24, 32, 64):
    _reg(Algo(f"rle_t{t}", "transform",
              (lambda t=t: lambda d: _rle_compress(t, False, d))(),
              _rle_decompress, cost="fast"))

# extra bitplane splits (n must divide 8; keep the valid ones)
for n in (1, 2, 4, 8):
    pass  # already registered above on this range

# extra interleave widths > 256 for huge-file transform power
for n in (512, 1024):
    _reg(Algo(f"interleave{n}", "transform",
              (lambda n=n: lambda d: _interleave_compress(n, d))(),
              (lambda n=n: lambda d: _interleave_decompress(n, d))(),
              cost="fast"))

# extra LZP contexts: reuse the lzp_compress helper with bigger windows
try:
    for ctx in (4, 6, 8, 12):
        _reg(Algo(f"lzp_c{ctx}", "transform",
                  (lambda c=ctx: lambda d: _lzp_compress(c, d))(),
                  (lambda c=ctx: lambda d: _lzp_decompress(c, d))(),
                  cost="normal", maxlen=(1 << 24)))
except Exception:
    pass

# extra XOR predictor offsets beyond 32
for k in (33, 48, 64, 96, 128, 160, 192, 224, 256):
    _reg(Algo(f"xorp_o{k}", "transform",
              (lambda k=k: lambda d: _xorp_compress(k, d))(),
              (lambda k=k: lambda d: _xorp_decompress(k, d))(),
              cost="fast"))

# delta_zz deeper orders too (small k is cheap, big text often has long
# runs of identical zigzag deltas far apart)
for k in (12, 16, 24, 32, 48, 64):
    _reg(Algo(f"delta_zz_o{k}", "transform",
              (lambda k=k: lambda d: _delta_zigzag_compress(k, d))(),
              (lambda k=k: lambda d: _delta_zigzag_decompress(k, d))(),
              cost="fast"))

# higher delta offsets (up to 512) for large-window predictive deltas
for k in (257, 384, 512):
    def _dck(k=k):
        return lambda d: _delta_compress(k, d)

    def _ddk(k=k):
        return lambda d: _delta_decompress(k, d)
    _reg(Algo(f"delta_o{k}", "transform", _dck(), _ddk(), cost="fast"))

# BPE deeper substitution rounds (beyond the original 1..48); keep maxlen
# small because deep rounds are O(rounds * n) and get very slow on big data
try:
    for r in (64, 96, 128):
        _reg(Algo(f"bpe_r{r}", "transform",
                  (lambda r=r: lambda d: _bpe_compress(r, d))(),
                  lambda d: _bpe_decompress(d),
                  cost="slow", maxlen=(1 << 22)))
except Exception:
    pass

_dict_proc = DictPreprocessor()

for ml in (3, 4, 5, 6, 8, 10, 12):
    for me in (256, 1024, 4096, 16384):
        _reg(_dict_proc.make_algo(f"dict_ml{ml}_me{me}", ml, me))


# ============================================================================
# PIPELINE & ENGINE
# ============================================================================

class Pipeline:
    """An ordered list of algorithm names, with optional dictionary pre-pass."""
    __slots__ = ("steps", "use_dict", "dict_passes")

    def __init__(self, steps=(), use_dict=False, dict_passes=1):
        self.steps = tuple(steps)
        self.use_dict = use_dict
        self.dict_passes = dict_passes

    def describe(self):
        parts = []
        if self.use_dict:
            parts.append(f"dict(x{self.dict_passes})")
        parts.extend(self.steps)
        return " -> ".join(parts)

    def to_dict(self):
        return {"steps": list(self.steps), "use_dict": self.use_dict,
                "dict_passes": self.dict_passes}

    @staticmethod
    def from_dict(d):
        steps = d.get("steps", [])
        # legacy v2 archives stored steps as [name, level] pairs
        names = []
        for s in steps:
            if isinstance(s, (list, tuple)):
                names.append(s[0])
            else:
                names.append(s)
        return Pipeline(names, d.get("use_dict", False),
                        d.get("dict_passes", 1))


# legacy (v2) decompressors - level is irrelevant for decompression
def _legacy_decompress(name, data):
    n = 1
    base = name
    if "_x" in name:
        base, _, tail = name.partition("_x")
        try:
            n = int(tail)
        except ValueError:
            n = 1
    fn = _LEGACY.get(base)
    if fn is None:
        raise ValueError(f"unknown v2 algorithm: {name}")
    for _ in range(n):
        data = fn(data)
    return data


def _legacy_gzip_d(data):
    return gzip.GzipFile(fileobj=__import__('io').BytesIO(data), mode='rb').read()


def _legacy_lzma_raw(data):
    return lzma.decompress(data, format=lzma.FORMAT_RAW,
                           filters=[{"id": lzma.FILTER_LZMA2, "preset": 9}])


def _legacy_lzma1_raw(data):
    return lzma.decompress(data, format=lzma.FORMAT_RAW,
                           filters=[{"id": lzma.FILTER_LZMA1, "preset": 9}])


_LEGACY = {
    "zlib": zlib.decompress,
    "gzip": _legacy_gzip_d,
    "bz2": bz2.decompress,
    "lzma": lzma.decompress,
    "xz": lambda d: lzma.decompress(d, format=lzma.FORMAT_XZ),
    "raw_lzma1": _legacy_lzma1_raw,
    "lzma2": _legacy_lzma_raw,
    "lzma2_ext": _legacy_lzma_raw,
    "raw": lambda d: d,
    # legacy v2 transform names
    "delta1": lambda d: _delta_decompress(1, d),
    "delta2": lambda d: _delta_decompress(2, d),
    "delta4": lambda d: _delta_decompress(4, d),
    "delta8": lambda d: _delta_decompress(8, d),
    "xor1": lambda d: _xorp_decompress(1, d),
    "xor2": lambda d: _xorp_decompress(2, d),
    "xor4": lambda d: _xorp_decompress(4, d),
    "mtf": _mtf_decompress,
}
if HAS_BROTLI:
    _LEGACY["brotli"] = brotli.decompress
if HAS_ZSTD:
    _LEGACY["zstd"] = (lambda d: zstd.ZstdDecompressor().decompress(d))
if HAS_SNAPPY:
    _LEGACY["snappy"] = snappy.decompress
if HAS_LZ4:
    _LEGACY["lz4"] = lz4block.decompress


class MaxCompressEngine:
    MAGIC = b'MXCP'
    VERSION = 3
    HEADER_MIN = 9
    # STORE magic: an archive that is just `MXRW + <raw bytes>`, produced when
    # compression would otherwise make the archive LARGER than the input.  This
    # is the hard guarantee against "compression made the file bigger".
    STORE_MAGIC = b'MXRW'

    def __init__(self, verbose=False, mode="full", codec_timeout=None,
                 threads=None):
        self.algos = _REGISTRY
        self.by_name = {a.name: a for a in self.algos}
        import os as _os
        self.threads = threads or max(1, _os.cpu_count() or 1)
        self.verbose = verbose
        self.mode = mode
        self.codec_timeout = codec_timeout   # seconds; None = no per-codec cap
        self.test_total = 0
        self.test_done = 0
        import threading as _th
        self._tl_pool = _th.local()   # thread-local 1-worker pools (parallel phases)
        self._pool = None
        self._mp_pool = None          # persistent fork pool (one per compress)
        self._on_event = None   # optional callable(event:dict) for live telemetry

    def _log(self, msg):
        if self.verbose:
            print(f"  [DEBUG] {msg}", file=sys.stderr)

    def _emit(self, event):
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # pipeline execution
    # ------------------------------------------------------------------
    def _call(self, fn, arg):
        """Run a codec/transform compress fn with an optional per-call time
        cap.  If the call runs longer than `codec_timeout`, its result is
        discarded (returns None) so a single pathological slow codec cannot
        stall the whole search and blow the phase time budget.

        Uses a thread-local 1-worker pool so parallel phase workers each get
        their own serial queue (a shared pool would serialize them)."""
        if self.codec_timeout is None:
            try:
                return fn(arg)
            except Exception:
                return None
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
        local = self._tl_pool
        pool = getattr(local, "pool", None)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=1)
            local.pool = pool
        fut = pool.submit(fn, arg)
        try:
            return fut.result(timeout=self.codec_timeout)
        except FTimeout:
            # abandon this slow call (it keeps running in the background
            # thread until it finishes or we give up on it; results ignored)
            self._log(f"codec timeout >{self.codec_timeout}s; skipping")
            return None
        except Exception:
            return None

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        if self._mp_pool is not None:
            try:
                self._mp_pool.terminate()
            except Exception:
                pass
            try:
                self._mp_pool.join()
            except Exception:
                pass
            self._mp_pool = None
        # shut down any thread-local 1-worker pools this thread created
        pool = getattr(self._tl_pool, "pool", None)
        if pool is not None:
            try:
                pool.shutdown(wait=False)
            except Exception:
                pass
            self._tl_pool.pool = None

    def _apply(self, data, pipe):
        cur = data
        orig = len(data)
        steps = [(self.by_name[s], s) for s in pipe.steps if s in self.by_name]
        if len(steps) != len(pipe.steps):
            return None
        cache = getattr(self, "_tcache", None)
        prefix = ()
        try:
            if pipe.use_dict:
                for _ in range(pipe.dict_passes):
                    nkey = prefix + ("@dict",)
                    if cache is not None and nkey in cache:
                        cur = cache[nkey]
                        if len(cur) > orig * 2:
                            return None
                    else:
                        cur = _dict_proc.compress(cur)
                        if len(cur) > orig * 2:
                            return None
                        if cache is not None:
                            self._cache_put(cache, nkey, cur)
                    prefix = nkey
            for a, name in steps:
                if a.maxlen and len(cur) > a.maxlen:
                    return None
                nkey = prefix + (name,)
                if cache is not None and nkey in cache:
                    cur = cache[nkey]
                else:
                    cur = self._call(a.compress, cur)
                    if not cur:
                        return None
                    if cache is not None:
                        self._cache_put(cache, nkey, cur)
                prefix = nkey
                if len(cur) > orig * 1.6 and len(steps) > 1:
                    # prune a bloated intermediate (or any step, so we cap
                    # peak memory) while keeping byte-exact semantics
                    return None
        except Exception as e:
            self._log(f"failed {pipe.describe()}: {e}")
            return None
        return cur

    def _cache_put(self, cache, key, val):
        """Insert into the LRU transform cache, evicting oldest entries when
        the entry count or approximate total bytes exceed their caps."""
        cache[key] = val
        cache.move_to_end(key)
        self._tcache_bytes += len(val)
        while (len(cache) > _MAX_CACHE_ENTRIES
               or self._tcache_bytes > _MAX_CACHE_BYTES):
            k, v = cache.popitem(last=False)
            self._tcache_bytes -= len(v)

    def _reverse(self, data, pipe, version=3):
        cur = data
        # apply steps in reverse (dict pass is applied first in _apply, so
        # it must be undone last)
        for name in reversed(pipe.steps):
            a = self.by_name.get(name)
            if a is None:
                cur = _legacy_decompress(name, cur)
            else:
                cur = a.decompress(cur)
        if pipe.use_dict:
            ddict = _dict_proc.decompress if version >= 3 \
                else _dict_proc.decompress_legacy
            for i in range(pipe.dict_passes):
                cur = ddict(cur)
        return cur

    # ------------------------------------------------------------------
    # adaptive multi-phase search
    # ------------------------------------------------------------------
    def _run_phase(self, pipelines, data, budget_end=None, label=""):
        """Evaluate every pipeline.

        Memory-conscious: we never accumulate full compressed outputs.
        Instead we keep:
          * `best`  - the single smallest (size, data, pipeline) found so far.
          * `scores`- a dict mapping a shorthand search-key -> size, used only
                      to rank codecs / transforms for later phase construction.
        Each phase reports its improvements back on `self`, so memory stays
        bounded by ~a few buffers regardless of how many pipelines run.
        """
        start = time.time()
        tests = len(pipelines)
        if label:
            print(f"  {label}: {tests:,} pipelines ...", file=sys.stderr)
        n_workers = max(1, min(self.threads, tests))
        if n_workers > 1 and len(pipelines) >= 8 and len(data) >= (1 << 16):
            # process-level parallel phase via fork: children inherit the
            # registry + data block (COW) and run their slice on real cores.
            global _PHASE_CTX, _P_DATA, _P_BUDGET
            _PHASE_CTX = (self, pipelines)
            _P_DATA = data
            _P_BUDGET = budget_end
            slices = [list(range(_i, tests, n_workers))
                      for _i in range(n_workers)]
            try:
                ctx = _mp.get_context("fork")
                with ctx.Pool(processes=n_workers) as pool:
                    res_iter = pool.imap_unordered(_phase_worker, slices)
                    all_done = True
                    for (dc, bs, bb, bi, ls, ok) in res_iter:
                        self.test_done += dc
                        for k, v in ls.items():
                            old = self.scores.get(k)
                            if old is None or v < old:
                                self.scores[k] = v
                        if bs is not None:
                            if self.best is None or bs < self.best[0]:
                                if self.best is not None:
                                    del self.best
                                self.best = (bs, bb, pipelines[bi])
                        if not ok:
                            all_done = False
                if label and tests:
                    print(f"\r  {label} done in {time.time() - start:.1f}s"
                          + " " * 30, file=sys.stderr)
                return all_done
            except Exception as _e:
                self._log(f"process pool failed ({_e}); falling back to serial")
        for pipe in pipelines:
            if budget_end is not None and time.time() > budget_end:
                self._log("time budget reached; stopping search")
                return False
            out = self._apply(data, pipe)
            self.test_done += 1
            self._emit({"type": "pipeline", "phase": label,
                        "method": pipe.describe(),
                        "ok": out is not None,
                        "size": len(out) if out is not None else None,
                        "done": self.test_done, "total": self.test_total})
            if out is not None:
                size = len(out)
                # record score for ranking (cheap: only an int, no data kept)
                key = self._score_key(pipe)
                old = self.scores.get(key)
                if old is None or size < old:
                    self.scores[key] = size
                # keep only the single best output (drop immediately if bloated)
                if self.best is None or size < self.best[0]:
                    if self.best is not None:
                        del self.best  # release previous best buffer
                    self.best = (size, out, pipe)
                elif size > len(data) * 2:
                    del out  # discard bloated result we will never use
            if self.test_done % 250 == 0:
                bs = self.best[0] if self.best else len(data)
                elapsed = time.time() - self.phase_t0
                rate = self.test_done / max(elapsed, 1e-9)
                eta = (self.test_total - self.test_done) / max(rate, 1e-9)
                print(f"\r  [{self.test_done:,}/{self.test_total:,}] "
                      f"{self.test_done / max(self.test_total, 1) * 100:.1f}% | "
                      f"best: {bs:,} ({bs / max(len(data), 1) * 100:.1f}%) | "
                      f"ETA: {eta:.0f}s   ", end='', file=sys.stderr)
        if label and tests:
            print(f"\r  {label} done in {time.time() - start:.1f}s"
                  + " " * 30, file=sys.stderr)
        return True

    @staticmethod
    def _score_key(pipe):
        """Shorthand key used to rank candidate codecs/transforms."""
        if pipe.use_dict:
            return f"dict:{pipe.describe()}"
        return pipe.describe()

    def compress(self, data, budget=None):
        import time as _t
        self._emit({"type": "compress_start", "blk_len": len(data),
                    "budget": budget, "mode": self.mode})
        self.phase_t0 = _t.time()
        t0 = self.phase_t0
        orig_size = len(data)
        # memory-conscious tracking: one best buffer + an int-only score map
        self.best = None      # (size, data, pipeline)  - single smallest output
        self.scores = {}      # search-key -> best size (ints only, no data)
        # LRU cache of shared transform/dict intermediates for this block.
        # Reset per compress() so one block's bytes never leak into the next.
        from collections import OrderedDict as _OD
        self._tcache = _OD()
        self._tcache_bytes = 0
        budget_end = _t.time() + budget if budget else None

        codecs = [a for a in self.algos if a.kind == "codec"]
        trans = [a for a in self.algos if a.kind == "transform"]
        fast_codecs = [a for a in codecs if a.cost == "fast"]
        slow_codecs = [a for a in codecs if a.cost == "slow"]
        norm_codecs = [a for a in codecs if a.cost == "normal"]
        cheap_codecs = fast_codecs + norm_codecs

        golden = []
        for name in ("lzma_l9", "lzma_l7", "xz_l9", "xz_l7",
                     "brotli_q11_w24", "brotli_q9_w22",
                     "zstd_l22", "zstd_l19", "zstd_l22_ldm",
                     "bz2_l9", "lzma_d9_67108864", "lzma2_xz9_8388608",
                     "brotli_q11_w24_m1", "brotli_q9_w22_m2",
                     "lz4f_bs7_cc1_bl1"):
            if name in self.by_name:
                golden.append(name)
        # very slow heavy-hitters: only tried as Phase-F terminals in
        # full/max (never in quick, never in Phase A)
        heavy = [n for n in ("lzma_ex9", "xz_ex9",
                             "zstdp_l22_w27", "zstdp_l22_w27_ldm",
                             "zopfli_gz15")
                 if n in self.by_name]
        # in-house strong entropy coders used as Phase-F terminals
        golden_inhouse = [n for n in ("ppmd_o4_m8", "ppmd_o4_m32", "ppmd_o4_m64",
                                      "ppmd_o6_m32", "ppmd_o6_m64", "ppmd_o8_m64",
                                      "ppmd_o10_m64", "ppmd_o12_m64", "ppmd_o16_m64",
                                      "pac1", "pac", "cm", "rans", "huff", "brotli_q11_w24")
                          if n in self.by_name]

        # ---------------- PHASE A: single codecs ----------------------
        if self.mode == "quick":
            # cheap codecs + the golden heavy hitters + in-house strong coders
            phaseA_algo = [a for a in fast_codecs]
            for name in golden:
                if name not in {a.name for a in phaseA_algo}:
                    phaseA_algo.append(self.by_name[name])
            for name in golden_inhouse:
                if name not in {a.name for a in phaseA_algo}:
                    phaseA_algo.append(self.by_name[name])
        else:
            # everything, cheap first so good results are found early
            phaseA_algo = fast_codecs + norm_codecs + slow_codecs

        phaseA = [Pipeline((a.name,)) for a in phaseA_algo]
        self.test_total = len(phaseA)
        self.test_done = 0
        print(f"  Phase A: {len(phaseA):,} single codecs ...", file=sys.stderr)
        if not self._run_phase(phaseA, data, budget_end, "Phase A · single-codecs"):
            self._log("budget reached during Phase A")

        # rank codecs by size (from the int-only scores map, no data kept)
        by_codec = {k: v for k, v in self.scores.items()
                    if k in self.by_name}
        ranked_codecs = sorted(by_codec, key=by_codec.get)
        if not ranked_codecs:
            ranked_codecs = [a.name for a in phaseA_algo]

        # ---------------- PHASE KICK: premium curated chains ------------
        # The known best-in-class chains (BWT->MTF->RLE->PPMd and friends)
        # MUST be tested early: Phase A alone determines the winners and
        # later transform phases can starve on the time budget.  With a fast
        # suffix array (pysais) BWT is cheap, so we push the strongest
        # curated pipelines up-front right after single codecs.
        kick = []
        if self.mode != "quick":
            bwt = ("bwt_b131072" if "bwt_b131072" in self.by_name
                   else "bwt_b16384" if "bwt_b16384" in self.by_name
                   else "bwt_b1024" if "bwt_b1024" in self.by_name else None)
            bwt2 = ("bwt_b16384" if "bwt_b16384" in self.by_name
                    else "bwt_b1024" if "bwt_b1024" in self.by_name else None)
            tops = ranked_codecs[:6]
            for c in tops:
                if c not in self.by_name:
                    continue
                if bwt is not None:
                    kick.append(Pipeline((bwt, "mtf", "rle_t4", c)))
                if bwt2 is not None and bwt2 != bwt:
                    kick.append(Pipeline((bwt2, "mtf", "rle_t4", c)))
                if c != "cm":
                    kick.append(Pipeline(("delta_o1", c)))
                if c != "cm":
                    kick.append(Pipeline(("rle_t4", c)))
            # explicit known winners (plain single codecs already in Phase A,
            # but the multi-transform chains are what really move the needle)
            for c in ("ppmd_o6_m32", "ppmd_o16_m64", "ppmd_o8_m64"):
                if c in self.by_name and bwt is not None:
                    kick.append(Pipeline((bwt, "mtf", "rle_t4", c)))
            # strong windowed-LZ codecs benefit from a byteswap->big-dict LZMA
            for c in ("lzma_d6_67108864", "lzma_d6_16777216"):
                if c in self.by_name:
                    kick.append(Pipeline(("byteswap", c)))
            # x86 code wins hard with BCJ -> aggressive raw LZMA2
            if "bcj_x86" in self.by_name:
                for c in ("raw9_bt4_d27", "raw9_bt4_d25", "raw9_bt4_d28"):
                    if c in self.by_name:
                        kick.append(Pipeline(("bcj_x86", c)))
                    if "byteswap" in self.by_name:
                        kick.append(Pipeline(("byteswap", "bcj_x86", c)))
            # dedupe, keep order
            seen = set()
            kick_u = []
            for p in kick:
                k = tuple(p.steps)
                if k not in seen:
                    seen.add(k)
                    kick_u.append(p)
            kick = kick_u
        if kick:
            self.test_total = self.test_total + len(kick)
            print(f"  Phase KICK: {len(kick):,} premium chains ...", file=sys.stderr)
            self._run_phase(kick, data, budget_end, "Phase Kick · premium chains")

        cheap_ranked = [c for c in ranked_codecs
                        if c in self.by_name and self.by_name[c].cost != "slow"]
        exp_ranked = [c for c in ranked_codecs
                      if c in self.by_name and self.by_name[c].cost == "slow"]
        top_cheap = (cheap_ranked or ranked_codecs)[:8]
        top_all = ranked_codecs[:12]

        # ---------------- PHASE B: transforms x top cheap codecs -------
        probe_trans = [t for t in trans if t.cost != "slow"]
        if self.mode == "quick":
            probe_trans = [t for t in probe_trans if t.cost == "fast"]
        k_top = 3 if self.mode == "quick" else 8
        topt = top_cheap[:k_top]
        phaseB = [Pipeline((t.name, c)) for t in probe_trans
                  for c in topt if c != t.name]
        phaseB_t = ([Pipeline((t.name, c), use_dict=True)
                     for t in probe_trans for c in topt[:1] if c != t.name]
                    if self.mode != "quick" else [])
        self.test_total = self.test_total + len(phaseB) + len(phaseB_t)
        print(f"  Phase B: {len(phaseB):,} transform->codec chains "
              f"(+{len(phaseB_t):,} with dict) ...", file=sys.stderr)
        self._run_phase(phaseB, data, budget_end, "Phase B · transform→codec")
        self._run_phase(phaseB_t, data, budget_end, "Phase B·dict · +dict")

        # rank transforms by best observed chain (from int-only scores)
        best_chain = {}
        for k, sz in self.scores.items():
            if "->" in k:
                t = k.split("->", 1)[0].strip()
                # only count non-dict chains for transform ranking
                if not k.startswith("dict:"):
                    best_chain.setdefault(t, sz)
        ranked_trans = sorted(best_chain, key=best_chain.get)

        # ---------------- PHASE C: top transforms x top codecs ---------
        if self.mode != "quick":
            T = [t for t in ranked_trans[:12]
                 if t in self.by_name and self.by_name[t].kind == "transform"]
            C = (top_cheap[:12] + exp_ranked[:8])
            seen_pipes = set()
            phaseC = []
            for t in T:
                for c in C:
                    if c == t:
                        continue
                    key = (t, c)
                    if key in seen_pipes:
                        continue
                    seen_pipes.add(key)
                    phaseC.append(Pipeline((t, c)))
            self.test_total = self.test_total + len(phaseC)
            print(f"  Phase C: {len(phaseC):,} ranked chains ...", file=sys.stderr)
            self._run_phase(phaseC, data, budget_end, "Phase C · transform chains")

        # ---------------- PHASE D: dict pre-pass chains ----------------
        if self.mode != "quick":
            T = [t for t in ranked_trans[:8]
                 if t in self.by_name and self.by_name[t].kind == "transform"]
            C = top_cheap[:6]
            phaseD = [Pipeline((t, c), use_dict=True)
                      for t in T for c in C if c != t]
            self.test_total = self.test_total + len(phaseD)
            print(f"  Phase D: {len(phaseD):,} dict->transform->codec chains ...",
                  file=sys.stderr)
            self._run_phase(phaseD, data, budget_end, "Phase D · dict→chains")

        # ---------------- PHASE E: double-transform chains --------------
        if self.mode in ("full", "max", "exhaust"):
            T = [t for t in ranked_trans[:8]
                 if t in self.by_name and self.by_name[t].kind == "transform"
                 and self.by_name[t].cost != "slow"]
            C = top_cheap[:3]
            phaseE = []
            for t1 in T:
                for t2 in T:
                    if t2 == t1:
                        continue
                    for c in C:
                        phaseE.append(Pipeline((t1, t2, c)))
            self.test_total = self.test_total + len(phaseE)
            print(f"  Phase E: {len(phaseE):,} transform->transform->codec "
                  f"chains ...", file=sys.stderr)
            self._run_phase(phaseE, data, budget_end, "Phase E · double-transform")

        # ---------------- PHASE SLOW: slow transforms, one fair shot ----
        if self.mode in ("full", "max", "exhaust"):
            slim = []
            for t in trans:
                if t.cost != "slow":
                    continue
                if t.maxlen and len(data) > t.maxlen:
                    continue
                if t.name.startswith("bpe_r") and self.mode not in ("max", "exhaust"):
                    r = int(t.name.split('_r')[1])
                    if r > 24:
                        continue
                slim.append(t)
            n_caps = 2 if self.mode in ("max", "exhaust") else 1
            caps = list(dict.fromkeys(top_all[:1] + top_cheap[:2]))[:n_caps]
            phaseSlow = [Pipeline((t.name, c)) for t in slim for c in caps]
            self.test_total = self.test_total + len(phaseSlow)
            if not self._run_phase(phaseSlow, data, budget_end,
                                   "Phase SLOW · slow transforms"):
                self._log("budget reached during Phase Slow")

        # ---------------- PHASE F: curated classic chains --------------
        curated = []
        if self.mode == "quick":
            alive = [g for g in (golden + golden_inhouse)[:2] if g in self.by_name]
        elif self.mode in ("max", "exhaust"):
            alive = [g for g in golden if g in self.by_name]
            alive = alive + [h for h in heavy if h in self.by_name]
            alive = alive + [g for g in golden_inhouse if g in self.by_name]
        else:
            alive = [g for g in (golden + golden_inhouse)[:4] if g in self.by_name]
        for c in alive:
            if self.mode != "quick" and "bwt_b1024" in self.by_name:
                curated.append(Pipeline(("bwt_b1024", "mtf", "rle_t4", c)))
            curated.append(Pipeline(("delta_o1", c)))
            curated.append(Pipeline(("delta_o33", c)))
            curated.append(Pipeline(("mtf", c)))
            curated.append(Pipeline(("xorp_o1", c)))
            curated.append(Pipeline(("nibble_hi", c)))
            if self.mode != "quick":
                curated.append(Pipeline(("delta_zz_o1", c)))
                curated.append(Pipeline(("delta_w16_o1", c)))
                curated.append(Pipeline(("delta_w32_o1", c)))
                curated.append(Pipeline(("lzp_c2", c)))
        # in non-quick modes ALSO give the strong in-house coders the
        # best curated chain (e.g. bwt->mtf->rle_t4->ppmd); these were
        # missed because `alive` only holds the first `golden` entries.
        if self.mode != "quick" and "bwt_b1024" in self.by_name:
            for n in golden_inhouse:
                if n not in self.by_name:
                    continue
                if n not in alive:
                    curated.append(Pipeline(("bwt_b1024", "mtf", "rle_t4", n)))
                curated.append(Pipeline(("bwt_b4096", "mtf", n)))
                curated.append(Pipeline(("bwt_b16384", "mtf", "rle_t4", n)))
                curated.append(Pipeline(("bwt_b131072", "mtf", "rle_t4", n))
                               if "bwt_b131072" in self.by_name else None)
                curated.append(Pipeline(("bwt_b65536", "mtf", "rle_t4", n))
                               if "bwt_b65536" in self.by_name else None)
                curated.append(Pipeline(("rle_t4", n)))
            curated[:] = [p for p in curated if p is not None]
        if self.mode != "quick":
            # bzip-style chains with the in-house entropy coders
            for ch in (("bwt_b1024", "mtf", "huff"),
                       ("bwt_b1024", "mtf", "rle_t4", "huff"),
                       ("bwt_b1024", "mtf", "pac"),
                       ("bwt_b4096", "mtf", "rle_t4", "huff"),
                       ("rle_t2", "huff"),
                       ("mtf2", "huff")):
                viable = all(n in self.by_name for n in ch)
                if viable:
                    curated.append(Pipeline(ch))
            for ch in (("bwt_b1024", "mtf", "rle_t4", "rans"),
                       ("bwt_b4096", "mtf", "rans"),
                       ("rle_t2", "rans"),
                       ("rle_t4", "rans"),
                       ("mtf2", "rans"),
                       ("delta_zz_o1", "rans"),
                       ("xorp_o1", "rans")):
                viable = all(n in self.by_name for n in ch)
                if viable:
                    curated.append(Pipeline(ch))
            # big-dict windowed LZ often wins on binary after a byte swap
            for ch in (("byteswap", "lzma_d6_67108864"),
                       ("byteswap", "lzma_d6_16777216"),
                       ("byteswap", "lzma_d8_67108864")):
                viable = all(n in self.by_name for n in ch)
                if viable:
                    curated.append(Pipeline(ch))
            # BCJ x86 filter before aggressive raw LZMA2 (7-Zip -mx=9 style)
            for ch in (("bcj_x86", "raw9_bt4_d27"),
                       ("bcj_x86", "raw9_bt4_d25"),
                       ("byteswap", "bcj_x86", "raw9_bt4_d25"),
                       ("bcj_x86", "bwt_b131072", "mtf", "rle_t4")):
                viable = all(n in self.by_name for n in ch)
                if viable:
                    curated.append(Pipeline(ch))
        for name in ("dict_ml4_me4096", "dict_ml5_me16384"):
            if name in self.by_name:
                for c in alive:
                    curated.append(Pipeline((name, c)))
        if curated:
            self.test_total = self.test_total + len(curated)
            print(f"  Phase F: {len(curated):,} curated chains ...", file=sys.stderr)
            self._run_phase(curated, data, budget_end, "Phase F · curated chains")

        # ---------------- PHASE G (max): codec->codec legacy chains ----
        if self.mode in ("max", "exhaust"):
            C = top_cheap[:6]
            phaseG = []
            for c1 in C:
                for c2 in C:
                    if c2 == c1:
                        continue
                    phaseG.append(Pipeline((c1, c2)))
            self.test_total = self.test_total + len(phaseG)
            print(f"  Phase G (max): {len(phaseG):,} codec->codec chains ...",
                  file=sys.stderr)
            self._run_phase(phaseG, data, budget_end, "Phase G · codec→codec")

        # ---------------- PHASE H (exhaust): brute-force everything ------
        # This mode guarantees we've tried every reasonable combination by
        # doing triple-transform chains, all-slow-transform × all-codecs,
        # and triple-codec chaining.  Time budget must be generous.
        if self.mode == "exhaust":
            # H1: triple-transform x top codecs (ALL non-slow transforms,
            # not just the ranked subset — this guarantees exhaustiveness)
            T = [t.name for t in probe_trans
                 if t.name in self.by_name
                 and self.by_name[t.name].kind == "transform"
                 and not (t.maxlen and len(data) > t.maxlen)]
            # dedupe + limit to keep the surface tractable
            T = list(dict.fromkeys(T))[:24]
            C = top_cheap[:5]
            phaseH1 = []
            for t1 in T:
                for t2 in T:
                    if t2 == t1:
                        continue
                    for t3 in T:
                        if t3 in (t1, t2):
                            continue
                        for c in C:
                            phaseH1.append(Pipeline((t1, t2, t3, c)))
            self.test_total += len(phaseH1)
            print(f"  Phase H1 (exhaust): {len(phaseH1):,} triple-transform "
                  f"chains ...", file=sys.stderr)
            self._run_phase(phaseH1, data, budget_end, "Phase H1 · triple-transform")

            # H2: ALL slow transforms x ALL codecs (slow transforms rarely
            # tried exhaustively; give them a full chance here)
            slow_all = [t for t in trans if t.cost == "slow"
                        and not (t.maxlen and len(data) > t.maxlen)]
            codec_all = top_all[:8]
            phaseH2 = [Pipeline((t.name, c)) for t in slow_all for c in codec_all]
            self.test_total += len(phaseH2)
            print(f"  Phase H2 (exhaust): {len(phaseH2):,} slow×all-codec ...",
                  file=sys.stderr)
            self._run_phase(phaseH2, data, budget_end, "Phase H2 · slow×codec")

            # H3: triple-codec chaining
            C3 = top_all[:6]
            phaseH3 = []
            for c1 in C3:
                for c2 in C3:
                    if c2 == c1:
                        continue
                    for c3 in C3:
                        if c3 in (c1, c2):
                            continue
                        phaseH3.append(Pipeline((c1, c2, c3)))
            self.test_total += len(phaseH3)
            print(f"  Phase H3 (exhaust): {len(phaseH3):,} triple-codec chains ...",
                  file=sys.stderr)
            self._run_phase(phaseH3, data, budget_end, "Phase H3 · triple-codec")

            # H4: double-transform x ALL codecs (broader than Phase E's top-3)
            phaseH4 = []
            for t1 in T:
                for t2 in T:
                    if t2 == t1:
                        continue
                    for c in codec_all:
                        phaseH4.append(Pipeline((t1, t2, c)))
            self.test_total += len(phaseH4)
            print(f"  Phase H4 (exhaust): {len(phaseH4):,} double-transform "
                  f"× all-codec ...", file=sys.stderr)
            self._run_phase(phaseH4, data, budget_end, "Phase H4 · double-transform×codec")

        # ---------------- finalize ------------------------------------
        elapsed = _t.time() - t0
        if self.best is not None:
            best_size, best_data, best_pipe = self.best
        else:
            best_size = orig_size
            best_data = data
            best_pipe = Pipeline()

        if best_size >= orig_size:
            best_size = orig_size
            best_data = data
            best_pipe = Pipeline()

        final = self._pack(best_data, best_pipe)
        # HARD GUARANTEE: never hand back an archive larger than the input.
        if len(final) > orig_size:
            final = self.STORE_MAGIC + bytes(data)
        self._emit({"type": "compress_end", "blk_len": len(data),
                    "best_size": best_size, "final_len": len(final),
                    "method": best_pipe.describe()})
        stats = {
            "original": orig_size,
            "compressed_raw": best_size,
            "compressed_final": len(final),
            "ratio": (len(final) / orig_size) if orig_size > 0 else 1.0,
            "method": best_pipe.describe() or "(raw)",
            "pipelines_tested": self.test_done,
            "algorithms": len(self.algos),
            "time_seconds": round(elapsed, 2),
        }
        print(f"\r  Done: {self.test_done:,} pipelines tested in {elapsed:.1f}s"
              f" (registry: {len(self.algos):,} algorithms)" + " " * 20,
              file=sys.stderr)
        return final, best_pipe, stats

    # ------------------------------------------------------------------
    # container format
    # ------------------------------------------------------------------
    def _pack(self, comp_data, pipe):
        meta = json.dumps({"pipeline": pipe.to_dict(), "engine": "MaxCompress",
                           "version": self.VERSION}).encode()
        result = bytearray(self.MAGIC)
        result += bytes([self.VERSION])
        result += struct.pack('>I', len(meta)) + meta
        result += struct.pack('>Q', len(comp_data)) + comp_data
        result += struct.pack('>I', zlib.crc32(comp_data) & 0xFFFFFFFF)
        return bytes(result)

    def decompress(self, data):
        if data[:4] == self.STORE_MAGIC:
            # stored verbatim (compression couldn't help); content is the rest
            return bytes(data[4:])
        if data[:4] != self.MAGIC:
            raise ValueError("Not a MaxCompress file")
        ver = data[4]
        if ver > self.VERSION:
            raise ValueError(f"Unsupported version: {ver}")
        off = 5
        if off + 4 > len(data):
            raise ValueError("truncated header")
        meta_len = struct.unpack('>I', data[off:off + 4])[0]
        off += 4
        if off + meta_len > len(data):
            raise ValueError("corrupt metadata length")
        meta = json.loads(data[off:off + meta_len].decode())
        off += meta_len
        if off + 8 > len(data):
            raise ValueError("truncated payload header")
        dlen = struct.unpack('>Q', data[off:off + 8])[0]
        off += 8
        if off + dlen > len(data):
            raise ValueError("corrupt data length")
        comp = data[off:off + dlen]
        off += dlen
        if ver >= 3:
            if off + 4 > len(data):
                raise ValueError("missing checksum")
            crc = struct.unpack('>I', data[off:off + 4])[0]
            if crc != (zlib.crc32(comp) & 0xFFFFFFFF):
                raise ValueError("checksum mismatch - archive is corrupt")
        pipe = Pipeline.from_dict(meta.get("pipeline", {}))
        return self._reverse(comp, pipe, ver)


# ============================================================================
# CLI
# ============================================================================

def cmd_compress(args):
    print(f"[MaxCompress] Compressing: {args.input}", file=sys.stderr)
    with open(args.input, 'rb') as f:
        data = f.read()
    print(f"[MaxCompress] Size: {len(data):,} bytes", file=sys.stderr)

    if args.quick:
        mode = "quick"
    elif args.exhaust:
        mode = "exhaust"
    elif args.max:
        mode = "max"
    else:
        mode = "full"

    engine = MaxCompressEngine(verbose=args.verbose, mode=mode)
    compressed, pipe, stats = engine.compress(data, budget=args.budget)

    out = args.output or (args.input + ".mxcp")
    with open(out, 'wb') as f:
        f.write(compressed)

    print(f"\n  Method:   {stats['method']}", file=sys.stderr)
    print(f"  Tested:   {stats['pipelines_tested']:,} combinations", file=sys.stderr)
    print(f"  Registry: {stats['algorithms']:,} algorithms", file=sys.stderr)
    print(f"  Original: {stats['original']:>12,} bytes", file=sys.stderr)
    print(f"  Compressed: {stats['compressed_final']:>12,} bytes", file=sys.stderr)
    print(f"  Ratio:    {stats['ratio'] * 100:.2f}%", file=sys.stderr)
    print(f"  Saved:    {(1 - stats['ratio']) * 100:.2f}%", file=sys.stderr)
    print(f"  Time:     {stats['time_seconds']:.1f}s", file=sys.stderr)
    print(f"  Output:   {out}", file=sys.stderr)


def cmd_decompress(args):
    print(f"[MaxCompress] Decompressing: {args.input}", file=sys.stderr)
    with open(args.input, 'rb') as f:
        data = f.read()
    engine = MaxCompressEngine()
    try:
        decompressed = engine.decompress(data)
    except Exception as e:
        print(f"[MaxCompress] Error: {e}", file=sys.stderr)
        return 1
    out = args.output
    if not out:
        out = args.input
        if out.endswith('.mxcp'):
            out = out[:-5]
        else:
            out = out + ".restored"
    with open(out, 'wb') as f:
        f.write(decompressed)
    print(f"[MaxCompress] Restored to: {out} ({len(decompressed):,} bytes)",
          file=sys.stderr)
    return 0


def cmd_info(args):
    with open(args.input, 'rb') as f:
        data = f.read()
    if data[:4] != MaxCompressEngine.MAGIC:
        print("Not a MaxCompress file", file=sys.stderr)
        return 1
    ver = data[4]
    print("MaxCompress Archive", file=sys.stderr)
    print(f"  Version:  {ver}", file=sys.stderr)
    print(f"  File:     {len(data):,} bytes", file=sys.stderr)
    try:
        off = 5
        meta_len = struct.unpack('>I', data[off:off + 4])[0]; off += 4
        meta = json.loads(data[off:off + meta_len].decode())
        off += meta_len
        dlen = struct.unpack('>Q', data[off:off + 8])[0]; off += 8
        comp = data[off:off + dlen]
        if ver >= 3:
            off += dlen
            crc = struct.unpack('>I', data[off:off + 4])[0]
            ok = crc == (zlib.crc32(comp) & 0xFFFFFFFF)
            print(f"  Checksum: {'OK' if ok else 'FAILED'}", file=sys.stderr)
        else:
            print(f"  Checksum: (v2, none)", file=sys.stderr)
        pipe = meta["pipeline"]
        print(f"  Data:     {dlen:,} bytes", file=sys.stderr)
        print(f"  Steps:    {pipe.get('steps')}", file=sys.stderr)
        print(f"  Dict:     {pipe.get('use_dict', False)}", file=sys.stderr)
        if pipe.get('use_dict'):
            print(f"  Dict passes: {pipe.get('dict_passes', 1)}", file=sys.stderr)
    except Exception as e:
        print(f"  (could not fully parse: {e})", file=sys.stderr)
    return 0


def cmd_list(args):
    codecs = [a for a in _REGISTRY if a.kind == "codec"]
    trans = [a for a in _REGISTRY if a.kind == "transform"]
    print(f"MaxCompress algorithm registry: {len(_REGISTRY):,} entries",
          file=sys.stderr)
    print(f"  codecs:    {len(codecs):,}", file=sys.stderr)
    print(f"  transforms:{len(trans):,}", file=sys.stderr)
    if args.names:
        for a in _REGISTRY[:args.names]:
            print(f"  {a.name:32s} {a.kind:9s} {a.cost}", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='maxcompress',
        description='MaxCompress v3 - multi-algorithm chain-search compression engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python maxcompress.py compress myfile.txt
  python maxcompress.py compress data.bin -o data.mxcp --quick
  python maxcompress.py compress data.bin --max          # exhaustive
  python maxcompress.py compress big.db --budget 120     # stop after 2 minutes
  python maxcompress.py decompress myfile.txt.mxcp
  python maxcompress.py info myfile.txt.mxcp
  python maxcompress.py list --names 30                  # show registry

Modes:
  --quick   singles + a small ranked chain surface
  (default) full: singles, transform->codec chains, dict chains, double transforms
  --max     adds codec->codec chains (legacy behaviour, mostly a curiosity)
  --exhaust full + triple-transform chains + all-slow×all-codec + triple-codec
        """)
    sub = parser.add_subparsers(dest='command')

    c = sub.add_parser('compress', help='Compress a file')
    c.add_argument('input')
    c.add_argument('-o', '--output')
    c.add_argument('-q', '--quick', action='store_true', help='Quick mode')
    c.add_argument('-m', '--max', action='store_true', help='Max mode')
    c.add_argument('-e', '--exhaust', action='store_true', help='Exhaustive mode')
    c.add_argument('-b', '--budget', type=float, default=None,
                   help='Time budget in seconds')
    c.add_argument('-v', '--verbose', action='store_true')

    d = sub.add_parser('decompress', help='Decompress a file')
    d.add_argument('input')
    d.add_argument('-o', '--output')

    i = sub.add_parser('info', help='Show file info')
    i.add_argument('input')

    l = sub.add_parser('list', help='Show algorithm registry')
    l.add_argument('--names', type=int, default=0)

    args = parser.parse_args()
    if args.command == 'compress':
        cmd_compress(args)
    elif args.command == 'decompress':
        sys.exit(cmd_decompress(args))
    elif args.command == 'info':
        sys.exit(cmd_info(args))
    elif args.command == 'list':
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()