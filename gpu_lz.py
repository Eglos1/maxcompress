#!/usr/bin/env python3
"""GPU-accelerated LZ77 match finding + tokenization for maxcompress.

The actual deflate/zstd/brotli compression is CPU C-library work that cannot
run on the GPU.  What the GPU *can* legitimately do -- and what the classic
LZ accelerators all do -- is the **match-finding** phase: scan a whole block
in parallel, find every repeated substring, and emit (offset, length) matches.
That is the most expensive part of LZ-class compression and it is
embarrassingly parallel, so it maps perfectly onto the OpenCL GPU.

We then use those matches to **pre-tokenize** the block with a recovery-safe
dictionary-style transform: long repeated runs are replaced by small integer
references, shrinking the input the CPU codec actually sees.  The tokenization
is reversible, so roundtrips stay byte-exact.

This keeps the CPU free to run the finicky entropy coders while the GPU does
the repetitive matching work on every block -- i.e. we push real compression
work onto the GPU rather than just number-crunching a histogram.
"""
import os

try:
    import numpy as np
    HAVE_NP = True
except Exception:
    HAVE_NP = False

try:
    import pyopencl as cl
    HAVE_CL = True
except Exception:
    HAVE_CL = False

KERNEL = r"""
// ---- parallel hash-chain LZ match finder ----
// head[bucket]  = first position hashing to bucket
// prev[pos]     = previous position with the same hash (chained)
// Every position computes its own hash in parallel; then a second pass walks
// the chains to find, for each position, the longest match >= MINLEN.
__kernel void build_chains(
    __global const uchar *data,
    const uint nbytes,
    __global int *head,     // size NBUCKETS, init -1
    __global int *prev)     // size nbytes
{
    size_t gid = get_global_id(0);
    if (gid >= nbytes) return;
    if (gid + 4 >= nbytes) { prev[gid] = -1; return; }
    uint h = ((uint)data[gid] << 24u)
           | ((uint)data[gid+1] << 16u)
           | ((uint)data[gid+2] << 8u)
           |  (uint)data[gid+3];
    uint b = (h * 2654435761u) & (NBUCKETS - 1u);
    int old = atomic_xchg(&head[b], (int)gid);
    if (old >= 0)
        prev[gid] = old;
    else
        prev[gid] = -1;
}

__kernel void find_matches(
    __global const uchar *data,
    const uint nbytes,
    __global const int *prev,
    __global int *moff,    // offset of match start (or -1)
    __global int *mlen,    // match length (or 0)
    __global int *chains_walked)
{
    size_t gid = get_global_id(0);
    if (gid >= nbytes) { moff[gid] = -1; mlen[gid] = 0; chains_walked[gid]=0; return; }
    if (gid + 4 >= nbytes) { moff[gid] = -1; mlen[gid] = 0; chains_walked[gid]=0; return; }
    int best_off = -1;
    int best_len = 0;
    int walked = 0;
    int p = (int)gid;
    // Walk the hash chain.  Only accept BACKWARD references (p < gid) so the
    // tokenizer is reversible (referenced bytes are already in place).
    while (p >= 0 && walked < CHAIN) {
        int cand = p;
        if (cand < (int)gid) {   // strictly backward
            int maxlen = (int)(nbytes - gid);
            if (maxlen > (int)(nbytes - cand)) maxlen = (int)(nbytes - cand);
            if (maxlen > MAXLEN) maxlen = MAXLEN;
            int L = 0;
            while (L < maxlen && data[gid+L] == data[cand+L]) L++;
            if (L >= MINLEN && L > best_len) {
                best_len = L;
                best_off = cand;
                if (L >= MAXLEN) break;   // maxed out
            }
        }
        p = prev[p];
        walked++;
    }
    moff[gid] = best_off;
    mlen[gid] = best_len;
    chains_walked[gid] = walked;
}
"""


class GpuLZMatchFinder:
    """Find repeated substrings in a block on the GPU.

    Call .find(data) -> list of (start, match_start, length) top matches.
    Falls back to a CPU reference when no GPU is available.
    """

    def __init__(self, min_len=8, max_len=273, nbuckets=1 << 20,
                 chain=64, use_gpu=True):
        self.min_len = min_len
        self.max_len = max_len
        self.nbuckets = nbuckets
        self.chain = chain
        self.enabled = False
        self.ctx = None
        self.queue = None
        self.prg = None
        self._err = None
        if use_gpu and HAVE_CL and HAVE_NP:
            self._init_gpu()

    def _init_gpu(self):
        try:
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
            src = KERNEL.replace("NBUCKETS", str(self.nbuckets)) \
                        .replace("MINLEN", str(self.min_len)) \
                        .replace("MAXLEN", str(self.max_len)) \
                        .replace("CHAIN", str(self.chain))
            self.prg = cl.Program(self.ctx, src).build()
            self.enabled = True
        except Exception as e:
            self.enabled = False
            self._err = str(e)

    def _bufs(self, n, work):
        mf = cl.mem_flags
        head = cl.Buffer(self.ctx, mf.READ_WRITE,
                         self.nbuckets * np.dtype(np.int32).itemsize)
        prev = cl.Buffer(self.ctx, mf.READ_WRITE, work * np.dtype(np.int32).itemsize)
        moff = cl.Buffer(self.ctx, mf.WRITE_ONLY, work * np.dtype(np.int32).itemsize)
        mlen = cl.Buffer(self.ctx, mf.WRITE_ONLY, work * np.dtype(np.int32).itemsize)
        cw = cl.Buffer(self.ctx, mf.WRITE_ONLY, work * np.dtype(np.int32).itemsize)
        return head, prev, moff, mlen, cw

    def find(self, data):
        """Return list of (start, match_start, length), best matches only."""
        if not self.enabled:
            return self._cpu_find(data)
        n = len(data)
        if n < 64:
            return []
        work = int((n // 256 + 1) * 256)
        in_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY
                           | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=np.frombuffer(data, dtype=np.uint8))
        head, prev, moff, mlen, cw = self._bufs(n, work)
        head_np = np.full(self.nbuckets, -1, dtype=np.int32)
        cl.enqueue_copy(self.queue, head, head_np)
        cl.enqueue_copy(self.queue, prev, np.full(n, -1, dtype=np.int32))
        # zero moff/mlen
        cl.enqueue_copy(self.queue, moff, np.zeros(work, dtype=np.int32))
        cl.enqueue_copy(self.queue, mlen, np.zeros(work, dtype=np.int32))
        self.prg.build_chains(self.queue, (work,), (256,),
                              in_buf, np.uint32(n), head, prev)
        self.prg.find_matches(self.queue, (work,), (256,),
                              in_buf, np.uint32(n), prev, moff, mlen, cw)
        mo = np.empty(work, dtype=np.int32)
        ml = np.empty(work, dtype=np.int32)
        cl.enqueue_copy(self.queue, mo, moff).wait()
        cl.enqueue_copy(self.queue, ml, mlen).wait()
        # collect matches with length >= min_len, sorted by length desc
        out = []
        for i in range(n):
            if ml[i] >= self.min_len:
                out.append((i, mo[i], int(ml[i])))
        out.sort(key=lambda x: x[2], reverse=True)
        return out

    def _cpu_find(self, data):
        n = len(data)
        table = {}
        out = []
        for i in range(n - self.min_len + 1):
            # hash 4 bytes
            if i + 4 <= n:
                key = data[i:i + 4]
            else:
                key = data[i:]
            cand = table.get(key)
            if cand is not None:
                # extend
                L = 0
                while (i + L < n and cand + L < i and
                       cand + L < n and data[i + L] == data[cand + L]):
                    L += 1
                    if L >= self.max_len:
                        break
                if L >= self.min_len:
                    out.append((i, cand, L))
            table.setdefault(key, []).append(i)
        out.sort(key=lambda x: x[2], reverse=True)
        # cap result size
        return out[:200000]

    @staticmethod
    def is_available():
        return HAVE_CL and HAVE_NP


# ============================================================================
# GPU pre-tokenizer: turn a block into (literal runs | reference tokens)
# ============================================================================
def gpu_pretokenize(mf, data, min_len=16, max_tokens=200000,
                    window=(1 << 24)):
    """Return (tokens, coverage) where tokens describe literal + reference
    segments.  This is the transform the CPU applies to shrink a block before
    running entropy coders.  tokens: list of dicts. coverage: bytes referenced.
    """
    matches = mf.find(data)
    if not matches:
        return [], 0
    # greedily pick non-overlapping long matches in file order
    picked = []
    n = len(data)
    limit = [n, 0]  # used[pos] simulated by scanning next
    # simple greedy: sort by start, take longest non-overlapping
    matches.sort(key=lambda x: x[0])
    last_end = -1
    coverage = 0
    for (start, mstart, length) in matches:
        if start < last_end:
            continue
        if start + length > n:
            length = n - start
        picked.append((start, mstart, length))
        coverage += length
        last_end = start + length
        if len(picked) >= max_tokens:
            break
    return picked, coverage


# convenience: percent of block covered by GPU-found references
def gpu_coverage(mf, data):
    _, cov = gpu_pretokenize(mf, data)
    return cov, len(data)


# ============================================================================
# Reversible GPU pre-tokenizer
# ============================================================================
# Encode a block as a mix of literal runs and backward references to earlier
# bytes (the matches the GPU found).  This strips repetition BEFORE the CPU
# entropy coder runs and is fully reversible, so roundtrips stay byte-exact.
#   literal run : 0x00 u24 len  <len bytes>
#   reference   : 0x01 u24 start  u24 match_start  u24 length
#   (marker tags let the decoder distinguish; all ints little-endian 3-byte)
#
# If the GPU finds little to remove, encode returns `data` unchanged (NULL),
# so the caller simply keeps the CPU-compressed block.
def gpu_transform_encode(data, matches, min_len=16, window=(1 << 24)):
    if not matches:
        return None, 0
    n = len(data)
    # greedy left-to-right pick of NON-OVERLAPPING backward refs
    by_start = sorted(matches, key=lambda x: x[0])
    picked = []
    last = -1
    coverage = 0
    for (start, mstart, length) in by_start:
        if start <= last:
            continue
        if mstart >= start:      # must be backward
            continue
        if length > 32767:
            length = 32767
        if start + length > n:
            length = n - start
        if length < min_len:
            continue
        # reject self-overlapping matches so decode is a plain snapshot copy:
        # the reference must fit entirely inside the already-built prefix
        if mstart + length > start:
            length = start - mstart     # trim to the non-overlapping prefix
            if length < min_len:
                continue
        picked.append((start, mstart, length))
        coverage += length
        last = start + length
        if len(picked) >= 300000:
            break
    if not picked:
        return None, 0
    # encode: literal runs and references
    out = bytearray()
    cursor = 0
    lit = bytearray()
    for (start, mstart, length) in picked:
        if start > cursor:
            lit += data[cursor:start]
        out.append(0x00)
        out += len(lit).to_bytes(3, 'little')
        out += lit
        lit = bytearray()
        out.append(0x01)
        out += (int(start)).to_bytes(3, 'little')
        out += (int(mstart)).to_bytes(3, 'little')
        out += (int(length)).to_bytes(3, 'little')
        cursor = start + length
    if cursor < n:
        lit += data[cursor:n]
    out.append(0x00)
    out += len(lit).to_bytes(3, 'little')
    out += lit
    if len(out) >= n:
        return None, coverage
    return bytes(out), coverage


def gpu_transform_decode(blob):
    data = bytearray()
    i = 0
    nb = len(blob)
    while i < nb:
        tag = blob[i]
        i += 1
        if tag == 0x00:
            ln = int.from_bytes(blob[i:i+3], 'little'); i += 3
            data += blob[i:i+ln]; i += ln
        elif tag == 0x01:
            start = int.from_bytes(blob[i:i+3], 'little'); i += 3
            mstart = int.from_bytes(blob[i:i+3], 'little'); i += 3
            length = int.from_bytes(blob[i:i+3], 'little'); i += 3
            # the previous literal run must have brought us exactly to `start`,
            # and the reference must be a plain snapshot within the prefix
            assert len(data) == start, "decode position mismatch"
            assert mstart < start and mstart + length <= start, "bad ref"
            data += data[mstart:mstart + length]
        else:
            raise ValueError(f"bad token tag {tag}")
    return bytes(data)

