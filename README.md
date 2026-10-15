# MaxCompress

The "find the absolute smallest representation" compressor.

A registry of 1000+ algorithms in two classes — entropy **codecs**
(zlib/gzip/bz2/lzma/xz/brotli/zstd/lz4) and reversible byte **transforms**
(delta, XOR-prev, RLE, MTF, BWT, BPE, bit-planes, dictionary substitution…).

The engine never chains codec-after-codec (that re-compresses incompressible
data and never helps). What works is chaining **transforms** to expose
structure, then **one** strong entropy coder:

```
delta -> lzma              time series / sorted data
BWT -> MTF -> RLE -> lzma  bzip2-style, great on text
dictionary -> zstd         JSON / XML / logs
bitplane -> brotli         low-bit-depth imagery
```

The search is multi-phase and adaptive: cheap single codecs first, then chains
built from the top-ranked ingredients only.

## v3 container
`'MXCP' | ver:1 | meta_len:u32BE | meta:json | data_len:u64BE | data | crc32:u32BE`
(v2 files without a checksum are still readable.)

## Files
| file | purpose |
|------|---------|
| `maxcompress.py` | the engine + CLI (single-file, dependency-free) |
| `maxcompress_gui.py` | Tkinter GUI |
| `maxcompress_batch.py` | batch / directory driver |
| `gpu_lz.py`, `live_viewer.py`, `lossy_video.py` | experiments |

## Run
```sh
python3 maxcompress.py --help
python3 maxcompress.py compress   bigfile.bin out.mxcp
python3 maxcompress.py decompress out.mxcp restored.bin
```

## License
MIT
