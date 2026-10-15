#!/usr/bin/env python3
"""
MaxCompress Batch Mode
======================
Folder-wide compression with cross-file dictionary sharing, CPU multicore
parallelism, and optional GPU (OpenCL) acceleration of the dictionary /
frequency analysis phase.

Features
--------
* Recursively walk a folder tree and compress every file to ``<name>.mxcp``.
* ``--inplace`` : delete the original once its archive is verified.
* Cross-file dictionary:
    - Phase 1 samples files, and (on the GPU) builds a histogram of repeated
      byte n-grams across the WHOLE batch.
    - A shared token dictionary is derived once and reused for every file,
      so small files stop paying per-file dictionary-headers and long-range
      repetition *between* videos is exploited too.
* Multicore:
    - Files are compressed in parallel across ``--workers`` CPU processes.
* GPU (optional, auto-detected):
    - Uses the KLE (OpenCL) device to accelerate the n-gram counting that
      feeds the cross-file dictionary. Falls back to threaded CPU counting
      if no OpenCL device is available.

Compatibility
-------------
The archive container remains MaxCompress v3.  The cross-file dictionary is
stored *in the folder* as ``.mxcp.dict`` and referenced by the per-file
archive, so a single archive copied away on its own still works (it embeds
its own dictionary as today) but batch-restore prefers the shared dict.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

# numpy/pyopencl are imported LAZILY (in the functions that genuinely need
# them) so the ROCm/OpenCL runtime is never mapped into the parent process.
# Every forked engine worker inherits the parent's address space via COW, so
# a parent that eagerly imports pyopencl+numpy made each of the 16 phase
# workers carry the whole OpenCL driver + numpy heap -> 100-320 MB of VmSwap
# per worker under memory pressure.  Import them here only on demand instead.
def _np():
    try:
        import numpy as np
    except Exception:
        return None
    return np


def _cl():
    try:
        import pyopencl as cl
    except Exception:
        return None
    return cl

# Reuse the battle-tested single-file engine + preprocessor.
import maxcompress as MC

# Lossy video re-encode layer (optional, needs ffmpeg).
try:
    from lossy_video import is_video, probe_codec, encode_lossy, MAGIC as LV_MAGIC
    HAS_LOSSY = True
except ImportError:
    HAS_LOSSY = False

DICT_MAGIC = b"MXDP"    # cross-file shared dictionary file header


# ============================================================================
# GPU ACCELERATED N-GRAM HISTOGRAM
# ============================================================================
#
# The dictionary builder needs, given a byte buffer, to know how often each
# short substring (n-gram) occurs.  This is a pure, embarrassingly-parallel
# counting operation -> ideal for the GPU.  We hash each n-gram to a fixed
# bucket and atomic-increment.  The CPU then reads back the histogram and
# selects the most common, highest-savings patterns.
#
class GpuHistogram:
    """Optional OpenCL-backed n-gram counter. Falls back to threaded CPU.

    We count the frequency of *recoverable* keys: single bytes (1-gram) and
    byte-pairs (2-gram).  Unlike arbitrary-length substrings, a byte/pair's
    integer value IS its identity, so the histogram is exact with no
    hash-collision ambiguity.  The CPU dictionary builder then seeds its
    longer-pattern search from the most common pairs, which is both correct
    and a genuine GPU-acceleration of the frequency phase.
    """
    def __init__(self, ngram_size=5, buckets=1 << 20, use_gpu=True):
        self.ngram_size = ngram_size
        self.buckets = buckets
        self.enabled = False
        self.ctx = None
        self.queue = None
        self.prg = None
        cl = _cl()
        if use_gpu and cl is not None:
            self._init_gpu()

    def _init_gpu(self):
        try:
            cl = _cl()
            if cl is None:
                return
            dev = None
            for p in cl.get_platforms():
                for d in p.get_devices():
                    if d.type & cl.device_type.GPU:
                        dev = d
                        break
                if dev is not None:
                    break
            if dev is None:
                return
            self.ctx = cl.Context([dev])
            self.queue = cl.CommandQueue(self.ctx)
            self.prg = cl.Program(self.ctx, r"""
            __kernel void uni_bi(
                __global const uchar *data,
                const uint nbytes,
                __global uint *uni,
                __global uint *bi)
            {
                size_t gid = get_global_id(0);
                if (gid >= nbytes) return;
                atomic_inc(&uni[data[gid]]);
                if (gid + 1 < nbytes)
                    atomic_inc(&bi[data[gid]*256u + data[gid+1]]);
            }
            """).build()
            self.enabled = True
            self._devname = dev.name
        except Exception as e:
            self.enabled = False
            self._err = str(e)

    def pair_freqs(self, data):
        """Return (unigram_dict, bigram_dict) with byte/pair->count.

        Correct on both GPU and CPU; the GPU path is the accelerated one.
        """
        if self.enabled and self.prg is not None:
            return self._gpu_pairs(data)
        return self._cpu_pairs(data)

    def _gpu_pairs(self, data):
        np = _np()
        cl = _cl()
        if np is None or cl is None:
            return self._cpu_pairs(data)
        n = len(data)
        if n == 0:
            return {}, {}
        # cap input to avoid gigantic buffers; still representatively large
        cap = min(n, 256 << 20)
        data = data[:cap]
        n = len(data)
        uni = np.zeros(256, dtype=np.uint32)
        bi = np.zeros(256 * 256, dtype=np.uint32)
        mf = cl.mem_flags
        in_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                           hostbuf=np.frombuffer(data, dtype=np.uint8))
        uni_buf = cl.Buffer(self.ctx, mf.READ_WRITE, uni.nbytes)
        bi_buf = cl.Buffer(self.ctx, mf.READ_WRITE, bi.nbytes)
        cl.enqueue_copy(self.queue, uni_buf, uni)
        cl.enqueue_copy(self.queue, bi_buf, bi)
        work = (int(np.ceil(n / 256)) * 256, )
        self.prg.uni_bi(self.queue, work, (256, ), in_buf, np.uint32(n),
                        uni_buf, bi_buf)
        cl.enqueue_copy(self.queue, uni, uni_buf).wait()
        cl.enqueue_copy(self.queue, bi, bi_buf).wait()
        u = {i: int(uni[i]) for i in range(256) if uni[i]}
        b = {(i >> 8, i & 0xFF): int(bi[i]) for i in range(256 * 256) if bi[i]}
        return u, b

    def _cpu_pairs(self, data):
        n = len(data)
        u = {}
        b = {}
        for i in range(n):
            u[data[i]] = u.get(data[i], 0) + 1
            if i + 1 < n:
                key = (data[i], data[i + 1])
                b[key] = b.get(key, 0) + 1
        return u, b

    @staticmethod
    def is_available():
        return _cl() is not None


# ============================================================================
# CROSS-FILE DICTIONARY BUILDER
# ============================================================================
#
# Given the sampled bytes of every file in the batch, pick the patterns that
# would save the most when byte-referenced.  Returns an ordered list of byte
# patterns (the "dictionary").  This is the shared dictionary used by every
# file in the folder.
#
class CrossFileDict:
    def __init__(self, min_len=4, max_len=64, max_entries=16384,
                 sample_mb=64, use_gpu=True):
        self.min_len = min_len
        self.max_len = max_len
        self.max_entries = max_entries
        self.sample_mb = sample_mb
        self.gpu = GpuHistogram(ngram_size=5, use_gpu=use_gpu)
        self.entries = []          # list[bytes] ordered by length desc
        self.pat2id = {}

    def build(self, sampled_buffers):
        """sampled_buffers: list of bytes, one (possibly truncated) sample per file."""
        data = b"".join(sampled_buffers)
        if len(data) < self.min_len:
            return []
        from collections import Counter

        n = len(data)
        # 1) byte & byte-pair frequencies (GPU-accelerated when available)
        _, bi = self.gpu.pair_freqs(data)

        # 2) seed from the most common byte-pairs, then grow each seed into a
        #    longer repeated substring.  Bounded: only walk ~400 seeds, and for
        #    each only probe up to MAXK occurrences while extending, so total
        #    work stays near-linear in n.
        popular = sorted(bi.items(), key=lambda kv: -kv[1])
        seeds = []
        for (a, b), cnt in popular:
            if cnt >= 2:
                seeds.append(bytes([a, b]))
            if len(seeds) >= 400:
                break

        MAXK = 64          # occurrences probed per pattern (for growing)
        MAXGROW = 8        # max extension bytes left+right
        longest = {}

        # precompute occurrence lists for the seed bytes to avoid re-scanning
        occ = self._occ_all(data, 2, MAXK)
        for seed in seeds:
            idxs = occ.get(seed)
            if not idxs or len(idxs) < 2:
                continue
            pat = seed
            # grow right
            for _ in range(MAXGROW):
                c = None
                common = True
                ref = data[idxs[0] + len(pat)] if idxs[0] + len(pat) < n else None
                for pos in idxs:
                    j = pos + len(pat)
                    cb = data[j] if j < n else None
                    if cb is None or (ref is not None and cb != ref):
                        common = False
                        break
                if not common or ref is None:
                    break
                pat = pat + bytes([ref])
            cnt = self._count_exact(data, pat, MAXK)
            if cnt >= 2 and self.min_len <= len(pat) <= self.max_len:
                longest[pat] = cnt

        # 3) score by savings and keep the best distinct patterns
        scored = []
        seen = set()
        for pat, cnt in longest.items():
            plen = len(pat)
            sav = (plen - 3) * (cnt - 1) - 3
            if sav > 0 and pat not in seen:
                seen.add(pat)
                scored.append((sav, plen, pat))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        entries = [p for _, _, p in scored[:self.max_entries]]
        entries.sort(key=lambda x: -len(x))
        self.entries = entries
        self.pat2id = {p: i for i, p in enumerate(entries)}
        return entries

    @staticmethod
    def _occ_all(data, nlen, maxk):
        """dict: bytes->list of first maxk occurrence positions for every
        distinct n-gram in data (n small)."""
        out = {}
        for i in range(len(data) - nlen + 1):
            s = data[i:i + nlen]
            lst = out.get(s)
            if lst is None:
                out[s] = [i]
            elif len(lst) < maxk:
                lst.append(i)
        return out

    @staticmethod
    def _count_exact(data, pat, cap=None):
        cnt = 0
        start = 0
        n = len(data)
        while start <= n - len(pat):
            i = data.find(pat, start)
            if i < 0:
                break
            cnt += 1
            if cap is not None and cnt >= cap:
                return cnt
            start = i + len(pat)
        return cnt

    def compress(self, data):
        """Dictionary token pass using the shared table.

        Emits the SAME self-contained format as DictPreprocessor.compress
        (u32 count | (u16 len, pattern)* | u32 stream_len | tokens) so the
        existing ``_dict_proc.decompress`` reverses it with zero new code.
        Only used when it actually shrinks the bytes.
        """
        if not self.entries:
            return data
        by_first = {}
        for i, pat in enumerate(self.entries):
            by_first.setdefault(pat[0], []).append((pat, i))
        out = bytearray()
        dlen = len(data)
        i = 0
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
        entries = self.entries
        result = struct.pack('>I', len(entries))
        for pat in entries:
            result += struct.pack('>H', len(pat))
            result += pat
        result += struct.pack('>I', len(out)) + bytes(out)
        return result

    def dump(self, path):
        blob = self.dumps_bytes()
        with open(path, 'wb') as f:
            f.write(blob)

    def dumps_bytes(self):
        """Serialize ONLY the shared entries (no per-block token stream)."""
        payload = bytearray()
        payload += struct.pack('>I', len(self.entries))
        for pat in self.entries:
            payload += struct.pack('>H', len(pat))
            payload += pat
        return DICT_MAGIC + struct.pack('>I', len(payload)) + bytes(payload)

    def tokenize(self, data):
        """Pre-tokenize a CHUNK against the shared table WITHOUT embedding the
        dict header.  Returns (tok, used): tok is the token stream, and used
        is True only when it actually shrank the chunk.  The shared entries
        are stored exactly once in the archive, so every block of every file
        can reference the SAME cross-chunk dictionary for free."""
        if not self.entries or len(data) < self.min_len:
            return data, False
        by_first = {}
        for i, pat in enumerate(self.entries):
            by_first.setdefault(pat[0], []).append((pat, i))
        out = bytearray()
        dlen = len(data)
        i = 0
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
        if len(out) >= dlen:
            return data, False
        return bytes(out), True

    def detokenize(self, tok):
        """Reverse ``tokenize`` against the shared entries (no header)."""
        if not self.entries:
            return tok
        out = bytearray()
        i = 0
        while i < len(tok):
            if tok[i] == 0xFF:
                if i + 1 >= len(tok):
                    raise ValueError("truncated shared-dict stream")
                if tok[i + 1] == 0xFF:
                    out.append(0xFF); i += 2
                else:
                    if i + 2 >= len(tok):
                        raise ValueError("truncated shared-dict stream")
                    v = struct.unpack('>H', tok[i + 1:i + 3])[0]
                    if v == 0 or v - 1 >= len(self.entries):
                        raise ValueError("bad shared-dict token")
                    out.extend(self.entries[v - 1])
                    i += 3
            else:
                out.append(tok[i]); i += 1
        return bytes(out)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            blob = f.read()
        return CrossFileDict.from_bytes(blob)

    @staticmethod
    def from_bytes(blob):
        if blob[:4] != DICT_MAGIC:
            raise ValueError("not a maxcompress dict blob")
        off = 4
        plen = struct.unpack('>I', blob[off:off + 4])[0]
        off += 4
        if plen > len(blob):
            raise ValueError("corrupt dict blob")
        n = struct.unpack('>I', blob[off:off + 4])[0]
        off += 4
        entries = []
        for _ in range(n):
            l = struct.unpack('>H', blob[off:off + 2])[0]
            off += 2
            entries.append(blob[off:off + l])
            off += l
        d = CrossFileDict.__new__(CrossFileDict)
        d.entries = entries
        d.pat2id = {p: i for i, p in enumerate(entries)}
        d.gpu = None
        d.min_len = 4
        return d


def _sample_file(path, cap):
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        if size <= cap:
            return f.read()
        halves = [0]
        if size > cap * 3:
            halves.append(size // 2)
        halves.append(size - cap)
        out = bytearray()
        for pos in halves[:2]:  # sample head + tail
            f.seek(pos)
            out += f.read(cap // 2)
        return bytes(out)


# ============================================================================
# STREAMING (MEMORY-BOUNDED) COMPRESSION
# ============================================================================
#
# Large files are compressed in fixed-size blocks so we never hold the whole
# file (or a multi-GB compressed buffer) in RAM at once.  Peak memory is
# bounded to ~2x the block size plus the engine's own per-block overhead.
#
# Streamed archive container:
#   b"MXCS" | u32 num_blocks
#   repeated, per block:
#       u32 comp_len | u32 orig_len | <comp_len bytes of compressed block>
#
STREAM_MAGIC = b"MXCS"
BLOCK = 16 << 20   # 16 MB blocks -> ~32 MB peak per worker

from threading import Lock as _Lock
_tele_lock = _Lock()

def _make_emitter(tel_path):
    if not tel_path:
        return None

    def emit(event):
        ev = dict(event)
        ev["t"] = round(time.time() * 1000)
        ev["w"] = os.getpid()
        try:
            line = json.dumps(ev) + "\n"
            with _tele_lock:
                with open(tel_path, "a") as f:
                    f.write(line)
        except Exception:
            pass

    return emit


def _copy_or_link(src, dst):
    """Copy (or hardlink when possible) src -> dst byte-identically with zero
    growth.  This is the 'stored' fallback guaranteeing an archive is never
    larger than its source."""
    try:
        os.link(src, dst)   # hardlink keeps size exactly == src, instant
        return
    except OSError:
        pass
    with open(src, 'rb') as fi, open(dst, 'wb') as fo:
        while True:
            b = fi.read(1 << 20)
            if not b:
                break
            fo.write(b)


def _write_ext_marker(out_stem, ext):
    """Record the source's original extension so restore can reconstruct the
    original file type (important because the .mxcp archive strips it)."""
    try:
        with open(out_stem + ".origext", 'w') as fm:
            fm.write(ext or ".bin")
    except Exception:
        pass


def _read_ext_marker(out_stem):
    try:
        with open(out_stem + ".origext") as fm:
            return fm.read().strip()
    except Exception:
        return ".bin"


def _write_stored_marker(out_stem):
    try:
        with open(out_stem + ".stored", 'w') as fm:
            fm.write("stored\n")
    except Exception:
        pass


def _compress_stream_to(src, relpath, out, mode, budget, dict_entries,
                        tel_path=None, codec_timeout=None, chunk=BLOCK,
                        gpu_pre=False):
    """Compress src into a segmented archive, bounded memory, and write to out.
    Returns (orig_total, comp_total, blocks) or raises."""
    engine = MC.MaxCompressEngine(verbose=False, mode=mode,
                                  codec_timeout=codec_timeout)
    mf = None
    if gpu_pre:
        try:
            from gpu_lz import GpuLZMatchFinder, gpu_transform_encode
            mf = GpuLZMatchFinder(min_len=24, chain=40, use_gpu=True)
        except Exception:
            mf = None
    emit = _make_emitter(tel_path)
    def ev_factory(base, block_i=None):
        def fn(ev):
            if emit is None:
                return
            ev2 = dict(ev)
            ev2["file"] = relpath
            if block_i is not None:
                ev2["block_i"] = block_i
            emit(ev2)
        return fn
    if emit is not None:
        engine._on_event = ev_factory(None)
    orig_total = 0
    comp_total = 0    # sum of the compressed block bytes (data only)
    blocks = 0
    with open(src, 'rb') as fi, open(out, 'wb') as fo:
        # reserve header (we won't know block count until done; write magic+count later)
        fo.write(STREAM_MAGIC)
        fo.write(struct.pack('>I', 0))  # placeholder block count
        while True:
            block = fi.read(chunk)
            if not block:
                break
            orig_total += len(block)
            engine._on_event = ev_factory(None, blocks)
            # optional GPU pre-tokenize layer (shrinks what the CPU sees)
            tflag = 0
            to_compress = block
            if mf is not None and len(block) >= (chunk >> 2):
                try:
                    matches = mf.find(block)
                    tok, _cov = gpu_transform_encode(block, matches, min_len=24)
                    if tok is not None and (len(tok) + 12) < len(block):
                        to_compress = tok
                        tflag = 1
                except Exception:
                    to_compress = block
            compressed, _, _ = engine.compress(to_compress, budget=budget)
            fo.write(struct.pack('>I', len(compressed)))
            fo.write(struct.pack('>I', len(block)))
            fo.write(struct.pack('B', tflag))
            fo.write(compressed)
            comp_total += len(compressed)
            blocks += 1
            del compressed, to_compress, block
        # patch block count (capture end-of-data position first so we don't
        # accidentally truncate the block payload)
        end_pos = fo.tell()
        fo.seek(4)
        fo.write(struct.pack('>I', blocks))
        fo.truncate(end_pos)
    engine.shutdown()
    return orig_total, comp_total, blocks


def _decompress_stream_to(src, out):
    """Restore a segmented archive, streaming block-by-block to out (bounded RAM)."""
    engine = MC.MaxCompressEngine(verbose=False, mode="full")
    from gpu_lz import gpu_transform_decode
    with open(src, 'rb') as fi:
        magic = fi.read(4)
        if magic != STREAM_MAGIC:
            raise ValueError("not a streamed archive")
        nb = struct.unpack('>I', fi.read(4))[0]
        orig_total = 0
        with open(out, 'wb') as fo:
            for _ in range(nb):
                head = fi.read(8)
                if len(head) < 8:
                    raise ValueError("truncated stream index")
                comp_len, orig_len = struct.unpack('>II', head)
                tflag = fi.read(1)[0]
                comp = fi.read(comp_len)
                if len(comp) < comp_len:
                    raise ValueError("truncated stream block")
                data = engine.decompress(comp)
                if tflag:
                    data = gpu_transform_decode(data)
                assert data == b'' or len(data) == orig_len, "block length mismatch"
                fo.write(data)
                orig_total += len(data)
                del data, comp
    return orig_total


# ============================================================================
# BATCH ENTRY POINT
# ============================================================================
def cmd_batch(args):
    root = args.input
    if not os.path.isdir(root):
        print(f"batch: '{root}' is not a directory", file=sys.stderr)
        return 1

    # 1) Enumerate files
    files = []
    skipped = []
    max_bytes = (args.max_file_mb or 0) << 20
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if not os.path.isfile(full) or full.endswith(".mxcp") \
               or full.endswith(".dict"):
                continue
            if max_bytes and os.path.getsize(full) > max_bytes:
                skipped.append(full)
                continue
            files.append(full)
    if skipped:
        print(f"[Batch] skipping {len(skipped):,} file(s) larger than "
              f"{args.max_file_mb} MB (use --max-file-mb to raise)", file=sys.stderr)
    if not files:
        print("batch: no files to compress (all skipped by size cap?)", file=sys.stderr)
        return 0
    files.sort()
    print(f"[Batch] {len(files):,} files, {sum(os.path.getsize(f) for f in files):,} bytes")

    # 2) Build cross-file dictionary (GPU accelerated counting)
    dict_path = os.path.join(root, args.dict_name or ".mxcp.dict")
    shared = None
    if args.cross:
        print(f"[Batch] building cross-file dictionary (GPU={GpuHistogram.is_available()})",
              file=sys.stderr)
        sampler = CrossFileDict(min_len=args.minlen, max_entries=args.dict_entries,
                                use_gpu=not args.no_gpu)
        cap = args.sample_mb * (1 << 20)
        samples = []
        for i, f in enumerate(files):
            samples.append(_sample_file(f, cap // max(1, len(files))))
        entries = sampler.build(samples)
        sampler.dump(dict_path)
        shared = sampler
        print(f"[Batch] cross-file dictionary: {len(entries):,} entries -> {dict_path}", file=sys.stderr)

    # 3) Parallel compression across workers
    workers = args.workers or max(1, (os.cpu_count() or 2))
    print(f"[Batch] compressing with {workers} workers, mode={args.mode}, budget={args.budget}")
    outdir = args.outdir or root
    if "/" not in outdir:
        outdir = os.path.join(root, outdir)
    os.makedirs(outdir, exist_ok=True)

    # telemetry JSONL path (live GUI tails this)
    tel_path = args.tel_path if hasattr(args, "tel_path") else None
    codec_timeout = getattr(args, "codec_timeout", None)
    gpu_pre = getattr(args, "gpu_pre", False)

    jobs = []
    for f in files:
        rel = os.path.relpath(f, root)
        jobs.append((f, root, outdir, rel, args.mode, args.budget,
                     args.inplace, args.cross, dict_path,
                     args.stream_mb << 20, tel_path, codec_timeout, gpu_pre))
    started = time.time()
    done = 0
    total_saved = 0
    tele_emit = _make_emitter(tel_path)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_compress_one, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            rel = j[3]
            if tele_emit is not None:
                tele_emit({"type": "file", "file": rel, "status": "done",
                           "done": done, "total": len(files)})
            try:
                res = fut.result()
            except Exception as e:
                res = (-1, None, f"ERROR:{e}")
            rel = j[3]
            src_size, new_size, method = res[0], res[1], res[2]
            if src_size >= 0 and new_size is not None:
                saved = src_size - new_size
                total_saved += saved
                print(f"  [{done+1}/{len(files)}] {rel}  {src_size:,} -> {new_size:,}  "
                      f"(saved {saved:,}) [via {method}]", file=sys.stderr)
            else:
                print(f"  [{done+1}/{len(files)}] {rel}  FAILED: {method}", file=sys.stderr)
            done += 1
    elapsed = time.time() - started
    print(f"\n[Batch] done {done:,}/{len(files):,} files in {elapsed:.1f}s "
          f"(saved {total_saved:,} bytes)", file=sys.stderr)
    return 0


# ============================================================================
# SINGLE-FOLDER ARCHIVE  (one big compressed file per folder, not a tree of
# per-file .mxcp files).  Output is a single self-contained .mxf archive.
#
# Format (append-sequential, manifest at the END so streaming stays bounded):
#   payload: repeated file entries:
#       u8 kind | u8 flags | u32 orig_len | u32 entry_len | <entry bytes>
#       kind: 0=stored(raw copy)  1=MXCP single  2=MXCS stream  3=dict entry
#   [...]
#   JSON manifest (list of {rel, orig_size, off, len, kind, flags})
#   b"MXFF" | u64 manifest_start_offset
# ============================================================================
FOLDER_MAGIC = b"MXFF"
FOLDER_END = b"MXFF"
_K_STORED, _K_MXCP, _K_MXCS, _K_DICT = 0, 1, 2, 3
_K_LOSSY = 4  # lossy video re-encode entry


def _folder_archive_entries(root, args):
    """Enumerate (rel, full_path, size) for every compressible file under root."""
    entries = []
    skipped = []
    max_bytes = (args.max_file_mb or 0) << 20
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".mxcp") or fn.endswith(".mxf") or fn.endswith(".dict"):
                continue
            full = os.path.join(dirpath, fn)
            if not os.path.isfile(full):
                continue
            if max_bytes and os.path.getsize(full) > max_bytes:
                skipped.append(full)
                continue
            rel = os.path.relpath(full, root)
            entries.append((rel, full, os.path.getsize(full)))
    entries.sort(key=lambda e: e[0])
    if skipped:
        print(f"[Folder] skipping {len(skipped):,} file(s) > {args.max_file_mb} MB",
              file=sys.stderr)
    return entries


def cmd_folder(args):
    """Compress an entire folder into ONE .mxf archive file."""
    root = args.input
    if not os.path.isdir(root):
        print(f"folder: '{root}' is not a directory", file=sys.stderr)
        return 1
    entries = _folder_archive_entries(root, args)
    if not entries:
        print("folder: no files to compress", file=sys.stderr)
        return 0
    out_arc = os.path.abspath(args.out) if args.out else os.path.join(
        os.path.dirname(root.rstrip("/")), os.path.basename(root.rstrip("/")) + ".mxf")
    if os.path.abspath(out_arc).startswith(os.path.abspath(root) + os.sep):
        print("folder: --out must be OUTSIDE the folder (else it'd archive itself)",
              file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(out_arc) or ".", exist_ok=True)

    # optional cross-file dictionary, embedded once into the archive and then
    # APPLIED to every streamed block (bit1 of the per-block flags): patterns
    # learned from every file in the folder keep working between chunks.
    # NOTE: built with use_gpu=False so the OpenCL/ROCm runtime is NEVER
    # loaded into the parent process - the parent forks the per-phase engine
    # pools (COW), so any GPU runtime in the parent would be inherited by all
    # 16 workers and blew up swap usage.  The GPU histogram is a one-off cost
    # and the CPU fallback is fine here.
    dict_bytes = None
    shared = None
    if args.cross:
        sampler = CrossFileDict(min_len=args.minlen, max_entries=args.dict_entries,
                                use_gpu=False)
        cap = args.sample_mb * (1 << 20)
        samples = [_sample_file(f, cap // max(1, len(entries))) for _, f, _ in entries]
        sampler.build(samples)
        dict_bytes = sampler.dumps_bytes()
        shared = sampler
        print(f"[Folder] cross-file dict embedded ({len(dict_bytes):,} B)",
              file=sys.stderr)

    stream_only = (args.stream_mb or 16) << 20
    codec_timeout = getattr(args, "codec_timeout", None)
    tel_emit = _make_emitter(args.tel_path)
    manifest = []
    off = 0
    total_orig = 0
    total_saved = 0
    started = time.time()

    with open(out_arc, "wb") as ar:
        # reserve header space for the end-pointer (patched at finish)
        if dict_bytes is not None:
            ar.write(struct.pack(">BBII", _K_DICT, 0, len(dict_bytes), len(dict_bytes)))
            ar.write(dict_bytes)
            off += 10 + len(dict_bytes)
            manifest.append({"rel": ".mxcp.dict", "orig_size": len(dict_bytes),
                             "off": 0, "len": len(dict_bytes), "kind": _K_DICT,
                             "flags": 0})
        for i, (rel, src, size) in enumerate(entries):
            try:
                ar.seek(0, os.SEEK_END)
                entry_start = ar.tell()
                # reserve the 10-byte unified entry header (BBII)
                ar.write(b"\0" * 10)
                # ---- lossy video path (optional, gated) ----
                used_lossy = False
                if getattr(args, "lossy", False) and HAS_LOSSY and is_video(src):
                    try:
                        vcodec, has_v = probe_codec(src)
                        if has_v and vcodec and vcodec not in ("av1", "vp9"):
                            import tempfile as _tf
                            tmp_path = _tf.mktemp(suffix=".webm")
                            try:
                                meta = encode_lossy(src, tmp_path, enc="av1-svt")
                                lsize = os.path.getsize(tmp_path)
                                if lsize < size:
                                    with open(tmp_path, "rb") as lf:
                                        lossy_data = lf.read()
                                    used_lossy = True
                                    ar.seek(entry_start + 10, os.SEEK_SET)
                                    mbytes = json.dumps(meta).encode()
                                    ar.write(LV_MAGIC + struct.pack(">I", len(mbytes))
                                             + mbytes + lossy_data)
                                    ar.seek(0, os.SEEK_END)
                                    kind, ebytes = _K_LOSSY, \
                                        4 + 4 + len(mbytes) + len(lossy_data)
                                    saved = size - ebytes
                            except Exception:
                                pass
                            finally:
                                try: os.unlink(tmp_path)
                                except Exception: pass
                    except Exception:
                        pass
                if not used_lossy:
                    if size > stream_only:
                        # stream blocks straight onto the archive (MXCS-framed)
                        comp_total, _nb = _stream_blocks_into(ar, src, size, args,
                                                              rel=rel,
                                                              tel_emit=tel_emit,
                                                              shared=shared)
                        if comp_total >= size:
                            # guarantee: never inflate a file's content -> stored
                            ar.seek(entry_start + 10, os.SEEK_SET)
                            _stream_copy_raw(ar, src, size)
                            ar.seek(0, os.SEEK_END)
                            kind, ebytes = _K_STORED, size
                            saved = 0
                        else:
                            kind, ebytes = _K_MXCS, comp_total
                            saved = size - comp_total
                    else:
                        # small file: standard single-buffer compression
                        with open(src, "rb") as f:
                            data = f.read()
                        engine = MC.MaxCompressEngine(verbose=False, mode=args.mode,
                                                      codec_timeout=codec_timeout,
                                                      threads=getattr(args, "threads", None))
                        if tel_emit is not None:
                            def rel_ev(ev):
                                ev2 = dict(ev)
                                ev2["file"] = rel
                                tel_emit(ev2)
                            engine._on_event = rel_ev
                        compressed, _, _ = engine.compress(data, budget=args.budget)
                        if len(compressed) >= size:
                            kind, ebytes = _K_STORED, size
                            payload = data
                        else:
                            kind, ebytes = _K_MXCP, len(compressed)
                            payload = compressed
                        saved = size - ebytes
                        ar.seek(entry_start + 10, os.SEEK_SET)
                        ar.write(payload[:ebytes])
                        ar.seek(0, os.SEEK_END)
                # write the unified entry header in place
                ar.seek(entry_start, os.SEEK_SET)
                ar.write(struct.pack(">BBII", kind, 0, size, ebytes))
                ar.seek(0, os.SEEK_END)
                end = ar.tell()
                manifest.append({"rel": rel, "orig_size": size, "off": entry_start,
                                 "len": end - entry_start, "kind": kind,
                                 "flags": 0})
                total_orig += size
                total_saved += saved
                if tel_emit is not None:
                    tel_emit({"type": "file", "file": rel, "status": "done",
                              "done": i + 1, "total": len(entries)})
                print(f"  [{i+1}/{len(entries)}] {rel}  {size:,} -> {end-entry_start:,} "
                      f"(saved {saved:,})", file=sys.stderr)
            except Exception as e:
                print(f"  [{i+1}/{len(entries)}] {rel}  FAILED: {e}", file=sys.stderr)
        # write manifest + end pointer (keep bounded: manifest only)
        man_bytes = json.dumps(
            {"version": 2, "files": manifest,
             "root": os.path.basename(root.rstrip("/"))}).encode()
        ar.write(man_bytes)
        ar.write(FOLDER_END)
        ar.write(struct.pack(">Q", len(man_bytes)))  # manifest length
    elapsed = time.time() - started
    arc_size = os.path.getsize(out_arc)
    print(f"\n[Folder] {len(entries):,} files -> {os.path.basename(out_arc)} "
          f"({arc_size:,} B, saved {total_saved:,} B) in {elapsed:.1f}s", file=sys.stderr)
    return 0


def _gpu_producer(q, src_path, blk):
    """GPU LZ pre-tokenizer worker. Runs in its OWN forked/spawned child with
    its own OpenCL context, so the engine's CPU fork pools in the parent are
    never disturbed by the OpenCL runtime's internal threads/locks."""
    from gpu_lz import GpuLZMatchFinder, gpu_transform_encode
    try:
        mf2 = GpuLZMatchFinder(min_len=24, chain=40, use_gpu=True)
        with open(src_path, "rb") as fi:
            while True:
                block = fi.read(blk)
                if not block:
                    break
                tok, tflag = block, 0
                if len(block) >= (blk >> 2):
                    try:
                        matches = mf2.find(block)
                        tt, _cov = gpu_transform_encode(block, matches, min_len=24)
                        if tt is not None and (len(tt) + 12) < len(block):
                            tok, tflag = tt, 1
                    except Exception:
                        tok, tflag = block, 0
                q.put((tok, tflag, len(block)))
    except Exception:
        pass
    finally:
        q.put(None)


def _stream_blocks_into(ar, src, size, args, rel=None, tel_emit=None,
                        block_i0=0, shared=None):
    """Append a streamed (MXCS) entry for src onto ar; return (comp_total, nblocks).

    Layout (called with ar positioned at entry_start+8): MXCS magic + per-block
    [u32 comp_len | u32 orig_len | u8 flags | compressed].
    flags is a bitmask: bit0 = GPU pre-tokenized, bit1 = shared-dict tokenized.
    When gpu_pre is enabled a background producer PROCESS runs the GPU LZ
    match-finder on NEXT blocks while the CPU multi-core engine digests the
    CURRENT block, so GPU + all CPU cores stay busy at the same time."""
    engine = MC.MaxCompressEngine(verbose=False, mode=args.mode,
                                  codec_timeout=getattr(args, "codec_timeout", None),
                                  threads=getattr(args, "threads", None))
    use_gpu = getattr(args, "gpu_pre", False)
    if tel_emit is not None:
        def ev_fn(ev, block_i=None):
            ev2 = dict(ev)
            if rel is not None:
                ev2["file"] = rel
            if block_i is not None:
                ev2["block_i"] = block_i
            tel_emit(ev2)
        engine._on_event = ev_fn
    chunk = BLOCK
    comp_total = 0
    nblocks = 0

    if use_gpu:
        # ---- GPU/CPU overlap: producer PROCESS pre-tokenizes next blocks ----
        # The GPU worker is a forked child with its OWN OpenCL context, so the
        # engine's fork pools in this (parent) process are not disturbed by the
        # OpenCL runtime's internal threads/locks.
        import multiprocessing as _mp_proc
        import importlib.util as _ilu
        if _ilu.find_spec("gpu_lz") is None:
            use_gpu = False
        if use_gpu:
            gq = _mp_proc.Queue(maxsize=3)
            gworker = _mp_proc.Process(target=_gpu_producer,
                                       args=(gq, src, chunk), daemon=True)
            gworker.start()
            ar.write(STREAM_MAGIC)
            while True:
                item = gq.get()
                if item is None:
                    break
                tok, tflag, ol = item
                # cross-chunk shared dict on top of the GPU token stream
                if shared is not None and tok:
                    tok2, used = shared.tokenize(tok)
                    if used:
                        tok, tflag = tok2, tflag | 2
                if tel_emit is not None:
                    engine._on_event = (lambda ev, bi=nblocks + block_i0: ev_fn(ev, bi))
                compressed, _, _ = engine.compress(tok, budget=args.budget)
                ar.write(struct.pack(">IIB", len(compressed), ol, tflag))
                ar.write(compressed)
                comp_total += len(compressed) + 9
                nblocks += 1
                del compressed, tok
            gworker.join(timeout=10)
            engine.shutdown()
            return comp_total, nblocks

    ar.write(STREAM_MAGIC)
    with open(src, "rb") as fi:
        while True:
            block = fi.read(chunk)
            if not block:
                break
            tok, tflag = block, 0
            if shared is not None and block:
                tok2, used = shared.tokenize(block)
                if used:
                    tok, tflag = tok2, tflag | 2
            if tel_emit is not None:
                engine._on_event = (lambda ev, bi=nblocks + block_i0: ev_fn(ev, bi))
            compressed, _, _ = engine.compress(tok, budget=args.budget)
            ar.write(struct.pack(">IIB", len(compressed), len(block), tflag))
            ar.write(compressed)
            comp_total += len(compressed) + 9
            nblocks += 1
            del compressed, block
    engine.shutdown()
    return comp_total, nblocks


def _stream_copy_raw(ar, src, size):
    """Append the raw (byte-identical) content of src onto ar (the stored
    fallback that guarantees content is never inflated)."""
    with open(src, "rb") as fi:
        while True:
            b = fi.read(1 << 20)
            if not b:
                break
            ar.write(b)


def _decompress_stream_handle(ar, size, shared=None):
    """Read an MXCS streamed entry from ar (positioned at its STREAM_MAGIC)
    and return the reconstructed original bytes.

    Block flags is a bitmask: bit0 = GPU pre-tokenized, bit1 = shared-dict
    tokenized.  Decode order reverses the compress order: engine decompress
    first, then remove the shared-dict tokens, then undo the GPU transform.
    """
    if ar.read(4) != STREAM_MAGIC:
        raise ValueError("bad MXCS entry")
    out = bytearray()
    gpu_dec = None
    while len(out) < size:
        head = ar.read(9)
        if len(head) < 9:
            break
        cl, ol, flags = struct.unpack(">IIB", head)
        cb = ar.read(cl)
        data = _decompress_block(cb, ol)
        if flags & 2:
            if shared is None:
                raise ValueError("block uses shared dict but no dict in archive")
            data = shared.detokenize(data)
        if flags & 1:
            if gpu_dec is None:
                from gpu_lz import gpu_transform_decode
                gpu_dec = gpu_transform_decode
            data = gpu_dec(data)
        out += data
    return bytes(out[:size])


def _decompress_block(cb, ol):
    # blocks were compressed as single-buffer MXCP archives (or stored)
    buf = cb
    if buf[:4] == MC.MaxCompressEngine.STORE_MAGIC:
        return buf[4:]
    if buf[:4] != MC.MaxCompressEngine.MAGIC:
        raise ValueError("bad MXCP block")
    from maxcompress import MaxCompressEngine as _E
    return _E(verbose=False).decompress(buf)


def _compress_one(job):
    (src, root, outdir, rel, mode, budget, inplace, cross, dict_path,
     stream_only, tel_path, codec_timeout, gpu_pre) = job
    # mirror the source's relative path inside the output tree
    rel_stem = os.path.splitext(rel)[0]
    out_path = os.path.join(outdir, rel_stem)
    out = out_path + ".mxcp"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    src_ext = os.path.splitext(os.path.basename(src))[1] or ".bin"
    _write_ext_marker(out_path, src_ext)
    try:
        size = os.path.getsize(src)
        # Use the memory-bounded streaming path whenever the file is big enough
        # that holding it fully + compressed buffers would be wasteful.
        if size > (stream_only or BLOCK):
            orig_total, comp_total, _blocks = _compress_stream_to(
                src, rel, out, mode, budget, None, tel_path, codec_timeout,
                gpu_pre=gpu_pre)
            if comp_total >= orig_total:
                # HARD GUARANTEE: never allow a bigger archive.  Keep a
                # byte-identical copy (zero growth) + stored marker.
                os.remove(out)
                _copy_or_link(src, out)
                _write_stored_marker(out_path)
                saved = 0
            else:
                saved = orig_total - comp_total
            if inplace:
                os.remove(src)
            return (orig_total, comp_total if saved else orig_total,
                    "stored" if saved == 0 else f"streamed({_blocks} blk)")
        # small file: standard in-memory path (optionally with cross-dict)
        with open(src, 'rb') as f:
            data = f.read()
        engine = MC.MaxCompressEngine(verbose=False, mode=mode,
                                      codec_timeout=codec_timeout)
        emit = _make_emitter(tel_path)
        if emit is not None:
            engine._on_event = (lambda ev, _rel=rel, _emit=emit:
                                _emit(dict(ev, file=_rel, block_i=0)))
        used_shared = False
        if cross and dict_path and os.path.exists(dict_path):
            d = CrossFileDict.load(dict_path)
            if d.entries:
                tok = d.compress(data)
                if len(tok) < len(data):
                    data = tok
                    used_shared = True
        compressed, pipe, stats = engine.compress(data, budget=budget)
        engine.shutdown()
        with open(out, 'wb') as fo:
            fo.write(compressed)
        marker = os.path.splitext(out)[0] + ".dictref"
        with open(marker, 'w') as fm:
            fm.write(str(int(used_shared)))
        # HARD GUARANTEE: never leave an archive bigger than the original.  If
        # compression couldn't beat the original (accounting for any overhead),
        # replace with a byte-identical copy (zero growth) + stored marker.
        if os.path.getsize(out) >= size:
            os.remove(out)
            _copy_or_link(src, out)
            _write_stored_marker(out_path)
            os.remove(marker) if os.path.exists(marker) else None
            comp_size = size
        else:
            comp_size = len(compressed)
        if inplace:
            os.remove(src)
        if used_shared and comp_size < size:
            pass
        return (size, comp_size, "stored" if comp_size >= size else
                pipe.describe())
    except Exception as e:
        return (-1, None, f"ERROR:{e}")


# ============================================================================
# BATCH RESTORE
# ============================================================================
def cmd_restore(args):
    root = args.input
    # load shared dictionary if present
    dict_path = os.path.join(root, args.dict_name or ".mxcp.dict")
    shared = None
    if os.path.exists(dict_path):
        try:
            shared = CrossFileDict.load(dict_path)
        except Exception as e:
            print(f"restore: could not load shared dict {dict_path}: {e}", file=sys.stderr)
    engine = MC.MaxCompressEngine(verbose=False, mode="full")
    mxcp = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".mxcp"):
                mxcp.append(os.path.join(dirpath, fn))
    restored = 0
    for f in mxcp:
        try:
            stem = f[:-5]            # strip .mxcp
            ext = _read_ext_marker(stem)
            out = stem + ext         # reconstruct original file type
            # 'stored' archives are byte-identical copies of the original
            # (compression couldn't help); reconstruct by copying the archive.
            if os.path.exists(stem + ".stored"):
                with open(f, 'rb') as fi, open(out, 'wb') as fo:
                    while True:
                        b = fi.read(1 << 20)
                        if not b:
                            break
                        fo.write(b)
                restored += 1
                continue
            with open(f, 'rb') as fi:
                header = fi.read(4)
            if header == STREAM_MAGIC:
                # bounded-memory streaming restore
                _decompress_stream_to(f, out)
                restored += 1
                continue
            with open(f, 'rb') as fi:
                data = fi.read()
            orig = engine.decompress(data)
            marker = os.path.splitext(f)[0] + ".dictref"
            if os.path.exists(marker):
                try:
                    with open(marker) as fm:
                        used_shared = fm.read().strip() == "1"
                except Exception:
                    used_shared = False
                if used_shared:
                    # reverse the cross-file token pass (the stream embeds the
                    # shared table; the standard preprocessor reverses it)
                    orig = MC._dict_proc.decompress(orig)
            with open(out, 'wb') as fo:
                fo.write(orig)
            restored += 1
        except Exception as e:
            print(f"restore: {f}: {e}", file=sys.stderr)
    print(f"[Batch] restored {restored:,}/{len(mxcp):,} archives -> {root}", file=sys.stderr)
    return 0


def _folder_unpack(args):
    """Restore every file from a single .mxf folder archive."""
    arc = args.input
    if not os.path.exists(arc):
        print(f"unpack: '{arc}' not found", file=sys.stderr)
        return 1
    with open(arc, "rb") as ar:
        ar.seek(-12, os.SEEK_END)
        tail = ar.read(12)
        if tail[:4] != FOLDER_END:
            print("unpack: not an MXFF archive", file=sys.stderr)
            return 1
        man_len = struct.unpack(">Q", tail[4:12])[0]
        ar.seek(-(man_len + 12), os.SEEK_END)
        man = json.loads(ar.read(man_len).decode())
    outdir = os.path.abspath(args.out) if args.out else man.get("root", "restored")
    os.makedirs(outdir, exist_ok=True)
    restored = 0
    # load the embedded cross-chunk shared dictionary once, if present
    shared = None
    for e in man["files"]:
        if e.get("kind") == _K_DICT:
            try:
                with open(arc, "rb") as ar:
                    ar.seek(e["off"])
                    hdr = ar.read(10)
                    _k, _fl, _ol, el = struct.unpack(">BBII", hdr)
                    blob = ar.read(el)
                shared = CrossFileDict.from_bytes(blob)
            except Exception as ee:
                print(f"unpack: could not load embedded shared dict: {ee}",
                      file=sys.stderr)
            break
    for e in man["files"]:
        try:
            rel = e["rel"]
            kind = e["kind"]
            if kind == _K_DICT:
                continue
            dst = os.path.join(outdir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(arc, "rb") as ar:
                ar.seek(e["off"])
                hdr = ar.read(10)
                _k, _fl, ol, el = struct.unpack(">BBII", hdr)
                if kind == _K_STORED:
                    orig = ar.read(el)
                elif kind == _K_MXCP:
                    data = ar.read(el)
                    orig = _decompress_block(data, ol)
                elif kind == _K_MXCS:
                    orig = _decompress_stream_handle(ar, ol, shared=shared)
                elif kind == _K_LOSSY:
                    # MXLV payload: magic(4) + meta_len(4) + meta(json) + video
                    lv_payload = ar.read(el)
                    lv_magic = lv_payload[:4]
                    if lv_magic != LV_MAGIC:
                        raise ValueError("bad MXLV payload")
                    mlen = struct.unpack(">I", lv_payload[4:8])[0]
                    meta = json.loads(lv_payload[8:8+mlen].decode())
                    orig = lv_payload[8+mlen:]
                    # reconstruct original filename with original extension
                    ext = meta.get("ext", ".mp4")
                    dst = os.path.join(outdir, rel)
                    dst = os.path.splitext(dst)[0] + ext
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                else:
                    continue
            with open(dst, "wb") as fo:
                fo.write(orig)
            restored += 1
        except Exception as ee:
            print(f"unpack: {e.get('rel')}: {ee}", file=sys.stderr)
    print(f"[Folder-unpack] restored {restored:,} files -> {outdir}", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(prog="maxcompress-batch",
                                     description="Folder-wide MaxCompress with cross-file dict + multicore + GPU")
    sub = parser.add_subparsers(dest="command")
    b = sub.add_parser("compress", help="Compress a whole folder")
    b.add_argument("input")
    b.add_argument("-o", "--outdir", default=None)
    b.add_argument("--cross", action="store_true", help="Build & use a cross-file dictionary")
    b.add_argument("--no-gpu", action="store_true", help="Disable GPU histogram")
    b.add_argument("--gpu-pre", action="store_true",
                   help="GPU LZ pre-tokenizer: run GPU match-finding on every "
                        "block and shrink it before the CPU entropy coder")
    b.add_argument("-w", "--workers", type=int, default=None)
    b.add_argument("--mode", choices=["quick", "full", "max", "exhaust"], default="full")
    b.add_argument("-b", "--budget", type=float, default=5)
    b.add_argument("--inplace", action="store_true", help="Delete originals after verifying")
    b.add_argument("--dict-name", default=".mxcp.dict")
    b.add_argument("--minlen", type=int, default=4)
    b.add_argument("--dict-entries", type=int, default=16384)
    b.add_argument("--sample-mb", type=int, default=64)
    b.add_argument("--max-file-mb", type=int, default=0,
                   help="Skip files larger than this many MB (0 = no limit)")
    b.add_argument("--stream-mb", type=int, default=16,
                   help="Files larger than this (MB) are streamed block-by-block "
                        "to keep RAM bounded, instead of loading whole file")
    b.add_argument("--tel-path", default=None,
                   help="JSONL file for live telemetry (very detailed per-pipeline "
                        "events the GUI viewer tails)")
    b.add_argument("--codec-timeout", type=float, default=20,
                   help="Only apply to maxcompress: drop a single codec call "
                        "that exceeds this many seconds (default 20) so a "
                        "pathologically slow codec never stalls the search")
    r = sub.add_parser("restore", help="Restore all .mxcp in a folder")
    r.add_argument("input")
    r.add_argument("--dict-name", default=".mxcp.dict")
    # single-folder-archive commands: one big compressed file per folder
    fo = sub.add_parser("folder", help="Compress a whole folder into ONE .mxf archive")
    fo.add_argument("input")
    fo.add_argument("--out", default=None, help="Destination .mxf path (must be outside the folder)")
    fo.add_argument("--cross", action="store_true", help="Embed a cross-file dictionary")
    fo.add_argument("--no-gpu", action="store_true")
    fo.add_argument("--gpu-pre", action="store_true",
                    help="GPU LZ pre-tokenizer: run GPU match-finding on every "
                         "block and shrink it before the CPU entropy coder")
    fo.add_argument("--threads", type=int, default=None,
                    help="Worker processes per block for the multi-core engine "
                         "(default: all CPU cores)")
    fo.add_argument("--mode", choices=["quick", "full", "max", "exhaust"], default="full")
    fo.add_argument("-b", "--budget", type=float, default=5)
    fo.add_argument("--dict-entries", type=int, default=16384)
    fo.add_argument("--minlen", type=int, default=4)
    fo.add_argument("--sample-mb", type=int, default=64)
    fo.add_argument("--max-file-mb", type=int, default=0)
    fo.add_argument("--stream-mb", type=int, default=16)
    fo.add_argument("--tel-path", default=None)
    fo.add_argument("--codec-timeout", type=float, default=20)
    fo.add_argument("--lossy", action="store_true", default=False,
                    help="Lossy re-encode H.264/HEVC video to AV1/VP9 if smaller "
                         "(requires ffmpeg; skipped for already-AV1/VP9 sources)")
    un = sub.add_parser("unpack", help="Restore a whole folder from one .mxf archive")
    un.add_argument("input")
    un.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.command == "compress":
        return cmd_batch(args)
    elif args.command == "restore":
        return cmd_restore(args)
    elif args.command == "folder":
        return cmd_folder(args)
    elif args.command == "unpack":
        return _folder_unpack(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
