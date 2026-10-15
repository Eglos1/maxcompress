#!/usr/bin/env python3
"""Lossy video re-encoding layer for maxcompress.

For video that is already entropy-compressed (H.264/HEVC/etc.), lossless
recompression of the *bytes* gains almost nothing -- the codec has already
squeezed out the entropy.  The only way to make a finished video much smaller
is to *re-encode* it with a more efficient codec (VP9/AV1).  That is what this
layer does, and it is deliberately LOSSY: decompression gives you back the same
container/type (the original extension is preserved) but the *pixels* are
re-encoded, not bit-restored.

This must therefore be flagged clearly and treated as a distinct archive kind
from the byte-exact lossless path.

Container format (single file):
  b"MXLV"  | u32 meta_len BE | meta:json | <re-encoded video bytes>
  meta = {"ext": <original extension e.g. ".mp4">, "src": <original codec>,
          "enc": <encoder used>, "q": <quality factor>}

On restore we write the re-encoded bytes to "<stem><ext>" so the extracted
file keeps its original file type, per design.
"""
import json
import os
import struct
import subprocess
import tempfile

MAGIC = b"MXLV"
ERR = {}

try:
    import zstandard as _zstd
    HAVE_ZSTD = True
except Exception:
    HAVE_ZSTD = False

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv",
             ".ts", ".m2ts", ".3gp", ".wmv", ".ogv", ".mpg", ".mpeg"}

# encoder presets (name -> ffmpeg args).  AV1 via SVT or libaom is the target.
_ENC_BY_NAME = {
    # SVT-AV1: fast + good. crf 30 is a strong visual/size balance.
    "av1-svt": ["-c:v", "libsvtav1", "-preset", "5", "-crf", "30"],
    # libaom AV1: slower, best compression-to-quality at this size scale
    "av1-aom": ["-c:v", "libaom-av1", "-cpu-used", "4", "-crf", "30",
                "-b:v", "0"],
    # VP9 (WebM family) via libvpx; the classic "webm" path
    "vp9": ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-row-mt", "1"],
    # H.264 same-codec re-encode (helps when source is bloated h264)
    "h264": ["-c:v", "libx264", "-preset", "slow", "-crf", "28"],
}


def _meta_pack(meta):
    b = json.dumps(meta).encode("utf-8")
    return struct.pack(">I", len(b)) + b


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXT


def probe_codec(path):
    """Return (video_codec, has_video) via ffprobe, or (None, False)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=120)
        codec = r.stdout.strip()
        return (codec if codec else None, bool(codec))
    except Exception:
        return None, False


def encode_lossy(src, out_path, enc="av1-svt", q=None, audio=True):
    """Re-encode src video to out_path (the re-encoded container bytes).
    Returns meta dict or raises.  `out_path` receives the new codec stream.
    Audio is kept (just remuxed) so the result remains a real watchable file.
    """
    vcodec, has_video = probe_codec(src)
    if not has_video:
        raise ValueError("no video stream")
    enc_args = list(_ENC_BY_NAME.get(enc, _ENC_BY_NAME["av1-svt"]))
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src,
           "-map", "0:v:0"] + enc_args
    if not audio:
        cmd.append("-an")
    else:
        cmd += ["-map", "0:a?", "-c:a", "copy"]
    # container: .webm holds vp9/av1+opus well; but to honor "original type"
    # we write the re-encoded bytes to the caller's out (any extension).
    cmd.append(out_path)
    subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                   check=True)
    meta = {"ext": os.path.splitext(src)[1], "src": vcodec, "enc": enc,
            "q": q}
    return meta


def restore_lossy(archive_path, out_dir, stem):
    """Parse an MXLV archive and write the re-encoded video to
    out_dir/<stem><original ext>.  Returns (out_path, meta)."""
    with open(archive_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("not an MXLV archive")
        mlen = struct.unpack(">I", f.read(4))[0]
        meta = json.loads(f.read(mlen).decode("utf-8"))
        payload = f.read()
    ext = meta.get("ext", ".mp4")
    out_path = os.path.join(out_dir, stem + ext)
    with open(out_path, "wb") as f:
        f.write(payload)
    return out_path, meta
