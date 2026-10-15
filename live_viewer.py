#!/usr/bin/env python3
"""Live viewer for the maxcompress batch run.

Tails the telemetry JSONL file produced by `maxcompress_batch.py compress
--tel-path ...` and renders a real-time, animated "flying through" console:
every pipeline tested, per-worker activity, phase, best-so-far size, savings,
throughput, elapsed and ETA.

Usage:  python3 live_viewer.py /path/to/telemetry.jsonl
"""
import json
import os
import sys
import time

try:
    import tkinter as tk
    from tkinter import scrolledtext
except Exception as e:  # pragma: no cover
    print("tkinter unavailable:", e, file=sys.stderr)
    sys.exit(1)


DARK_BG = "#0b0f1a"
DARK_BG2 = "#111827"
PANEL = "#131a2b"
ACCENT = "#22d3ee"
ACCENT2 = "#a78bfa"
GOOD = "#4ade80"
GOODDIM = "#14532d"
BAD = "#f87171"
WARN = "#fbbf24"
TEXT = "#dbeafe"
DIM = "#64748b"

FONT_MONO = ("DejaVu Sans Mono", 10)
FONT_MONO_SM = ("DejaVu Sans Mono", 9)
FONT_BIG = ("DejaVu Sans Mono", 26, "bold")
FONT_TITLE = ("DejaVu Sans", 15, "bold")


class LiveViewer:
    def __init__(self, root, path):
        self.tk = root
        root.title("maxcompress — live")
        root.configure(bg=DARK_BG)
        self.path = path
        self.offset = 0
        self.inode = None
        self.max_lines = 12000

        # rolling state
        self.methods = 0
        self.total_tests = 0
        self.files_done = 0
        self.files_total = 0
        self.active = {}          # worker pid -> dict
        self.last_activity = time.time()
        self.best_size = None
        self.orig_size = None
        self.file = "(waiting)"
        self.block = 0
        self.phase = ""
        self.workers_active = set()
        self.started = time.time()
        self.rate = 0.0
        self.saved = 0

        self._build()
        self.after_id = root.after(120, self._poll)

    # ------------------------------------------------------------------ UI
    def _build(self):
        self.tk.geometry("1500x900")
        self.tk.minsize(960, 600)

        top = tk.Frame(self.tk, bg=DARK_BG)
        top.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(top, text="⚡ MAXCOMPRESS", fg=ACCENT, bg=DARK_BG,
                 font=FONT_TITLE).pack(side="left")
        self.lbl_status = tk.Label(top, text="starting…", fg=WARN, bg=DARK_BG,
                                   font=FONT_MONO)
        self.lbl_status.pack(side="right")

        # big metrics strip
        metrics = tk.Frame(self.tk, bg=DARK_BG)
        metrics.pack(fill="x", padx=16)
        self.big_saved = self._metric(metrics, "TOTAL SAVED", "0 B", GOOD)
        self.big_rate = self._metric(metrics, "THROUGHPUT", "0/s", ACCENT)
        self.big_best = self._metric(metrics, "BEST", "—", WARN)
        self.big_elapsed = self._metric(metrics, "ELAPSED", "0s", TEXT)
        self.big_eta = self._metric(metrics, "ETA", "—", ACCENT2)

        # worker activity row
        self.lbl_workers = tk.Label(self.tk,
                                    text="workers: [WAIT]", fg=DIM, bg=DARK_BG,
                                    font=FONT_MONO_SM, anchor="w")
        self.lbl_workers.pack(fill="x", padx=16, pady=(6, 2))

        # current-file line
        self.lbl_cur = tk.Label(self.tk, text="", fg=ACCENT2, bg=DARK_BG,
                                font=FONT_MONO, anchor="w", wraplength=1460,
                                justify="left")
        self.lbl_cur.pack(fill="x", padx=16, pady=(2, 2))

        # phase / block / best line
        self.lbl_phase = tk.Label(self.tk, text="", fg=DIM, bg=DARK_BG,
                                  font=FONT_MONO_SM, anchor="w")
        self.lbl_phase.pack(fill="x", padx=16, pady=(0, 4))

        # files progress bar
        self.prog = tk.Canvas(self.tk, height=16, bg=GOODDIM,
                              highlightthickness=0)
        self.prog.pack(fill="x", padx=16, pady=(0, 6))

        mid = tk.Frame(self.tk, bg=DARK_BG)
        mid.pack(fill="both", expand=True, padx=16, pady=(0, 10))

        # left: detailed scrolling log ("flying through")
        logbox = tk.LabelFrame(mid, text="  flying through  ", fg=ACCENT,
                               bg=PANEL, font=FONT_MONO_SM, bd=0,
                               highlightbackground=ACCENT, highlightthickness=1)
        logbox.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.log = scrolledtext.ScrolledText(
            logbox, bg=PANEL, fg=TEXT, font=FONT_MONO_SM, wrap="none",
            insertbackground=ACCENT, relief="flat", state="disabled",
            highlightthickness=0)
        self.log.pack(fill="both", expand=True, padx=2, pady=2)
        self.log.tag_configure("good", foreground=GOOD)
        self.log.tag_configure("acc", foreground=ACCENT)
        self.log.tag_configure("ph", foreground=ACCENT2)
        self.log.tag_configure("bad", foreground=BAD)
        self.log.tag_configure("dim", foreground=DIM)
        self.auto = tk.BooleanVar(value=True)
        tk.Checkbutton(logbox, text="auto-scroll", variable=self.auto,
                       bg=PANEL, fg=DIM, selectcolor=PANEL,
                       activebackground=PANEL, activeforeground=TEXT,
                       font=FONT_MONO_SM,
                       highlightthickness=0).pack(anchor="w")

        # right: worker detail panel
        rp = tk.Frame(mid, bg=DARK_BG)
        rp.pack(side="right", fill="y", padx=(0, 0))
        tk.Label(rp, text="workers", fg=DIM, bg=DARK_BG,
                 font=FONT_MONO_SM).pack(anchor="w")
        self.worker_cards = {}
        rp_c = tk.Frame(rp, bg=DARK_BG)
        rp_c.pack(fill="y", anchor="n")
        self.worker_cards_container = rp_c

    def _metric(self, parent, label, value, color):
        box = tk.Frame(parent, bg=PANEL, padx=14, pady=8)
        box.pack(side="left", expand=True, fill="both", padx=4)
        tk.Label(box, text=label, fg=DIM, bg=PANEL, font=FONT_MONO_SM
                 ).pack(anchor="w")
        lbl = tk.Label(box, text=value, fg=color, bg=PANEL, font=FONT_BIG,
                       anchor="w")
        lbl.pack(anchor="w")
        return (label, lbl)

    def _ensure_worker_cards(self, n):
        while len(self.worker_cards) < n:
            i = len(self.worker_cards) + 1
            card = tk.Frame(self.worker_cards_container, bg=PANEL, padx=8,
                            pady=4, highlightbackground=DIM,
                            highlightthickness=1)
            card.pack(fill="x", pady=2)
            tl = tk.Label(card, text=f"W{i}  idle", fg=DIM, bg=PANEL,
                          font=FONT_MONO_SM, anchor="w", width=26)
            tl.pack(fill="x")
            bl = tk.Label(card, text="", fg=DIM, bg=PANEL, font=FONT_MONO_SM,
                          anchor="w", width=34, wraplength=300, justify="left")
            bl.pack(fill="x")
            self.worker_cards[i] = (tl, bl)

    # ------------------------------------------------------------- helpers
    def _red(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag or "dim")
        # trim oldest lines
        linecount = int(self.log.index("end-1c").split(".")[0])
        if linecount > self.max_lines:
            self.log.delete("1.0", f"{linecount - self.max_lines}.0")
        if self.auto.get():
            self.log.see("end")
        self.log.configure(state="disabled")

    def _fmt_saved(self, b):
        sign = "" if b >= 0 else "-"
        b = abs(b)
        for u in ("B", "KB", "MB", "GB", "TB"):
            if b < 1024 or u == "TB":
                return f"{sign}{b:,.1f} {u}" if u != "B" else f"{sign}{b:,} {u}"
            b /= 1024
        return f"{sign}{b:,.1f} TB"

    def _fmt_eta(self, secs):
        if secs is None or secs < 0 or secs > 1e7:
            return "—"
        s = int(secs)
        d = s // 86400
        s %= 86400
        h = s // 3600
        s %= 3600
        m = s // 60
        s %= 60
        if d:
            return f"{d}d{h}h"
        if h:
            return f"{h}h{m}m"
        return f"{m}m{s}s"

    def _research(self, key, min_w=0):
        # paint metrics via stored lbl references
        pass

    # --------------------------------------------------------------- poll
    def _poll(self):
        self._read_tail()
        self._render_metrics()
        self.after_id = self.tk.after(150, self._poll)

    def _read_tail(self):
        try:
            st = os.stat(self.path)
            if self.inode is None:
                self.inode = st.st_ino
            elif st.st_ino != self.inode:
                # file replaced (recreated); restart from 0
                self.inode = st.st_ino
                self.offset = 0
            if st.st_size < self.offset:
                self.offset = 0  # truncated
        except OSError:
            return
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
                self.offset = f.tell()
        except OSError:
            return
        for raw in data.split(b"\n"):
            if not raw.strip():
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            self._handle(ev)

    def _handle(self, ev):
        self.last_activity = time.time()
        typ = ev.get("type")
        w = ev.get("w")
        if typ == "pipeline":
            self.total_tests += 1
            self.methods += 1
            m = ev.get("method", "")
            ok = ev.get("ok")
            size = ev.get("size")
            tag = "dim"
            if ev.get("total") == ev.get("done"):
                tag = "ph"
            self._red(f"[{m}] ok={ok} size={size}"
                      + ("  ◄ BEST" if ok and size is not None and
                         self.best_size is not None and size <= self.best_size
                         else ""), tag)
            # active worker
            if w is not None and w in self.active:
                self.active[w]["method"] = m
        elif typ == "phase" or typ == "compress_start":
            self.phase = ev.get("phase") or ev.get("mode") or ""
            if typ == "compress_start":
                self.orig_size = ev.get("blk_len")
                self.best_size = ev.get("blk_len")
                self.block = ev.get("block_i")
                f = ev.get("file", "")
                self.file = f
                self._red(f"── block {self.block} · {self.orig_size} B · "
                          f"{self.file}", "acc")
                self.lbl_cur.configure(
                    text=f"  {self.file}  ·  block {self.block} "
                         f"· {self._fmt_saved(self.orig_size or 0)}")
        elif typ == "compress_end":
            if self.best_size is not None and ev.get("best_size") is not None:
                if ev.get("best_size") < self.best_size:
                    self.saved += (self.best_size - ev["best_size"])
                    self.best_size = ev["best_size"]
                self._red(f"  ✔ block best {self._fmt_saved(ev['best_size'])} "
                          f"via {ev.get('method','?')} · "
                          f"saved {self._fmt_saved(self.best_size -
                                                   ev['best_size'])}", "good")
        elif typ == "file":
            self.files_done = ev.get("done", self.files_done)
            self.files_total = max(self.files_total, ev.get("total",
                                                            self.files_total))
            if ev.get("status") == "done":
                pass
                # self._red(f"done file {ev.get('file','')}", "good")
        if "file" in ev:
            self.file = ev.get("file", self.file)

    def _render_metrics(self):
        now = time.time()
        elapsed = now - self.started
        rate = self.total_tests / max(elapsed, 1e-9)
        self.big_rate[1].configure(text=f"{rate:,.0f}/s")
        self.big_elapsed[1].configure(text=self._fmt_eta(elapsed))
        self.big_saved[1].configure(text=self._fmt_saved(self.saved))
        if self.best_size is not None:
            self.big_best[1].configure(
                text=self._fmt_saved(self.best_size))
        # pipeline throughput ETA
        if self.files_total:
            frac = self.files_done / self.files_total
            if frac > 0:
                self.big_eta[1].configure(
                    text=self._fmt_eta(elapsed / frac - elapsed))
            self.prog.delete("all")
            self.prog.create_rectangle(
                0, 0, self.prog.winfo_width() * frac, 16,
                fill=ACCENT, outline="")
        self.lbl_status.configure(
            text=f"files {self.files_done}/{self.files_total} · "
                 f"tests {self.total_tests:,}")
        # activity timestamps
        idle = now - self.last_activity > 8
        self.lbl_status.configure(
            fg=WARN if idle else GOOD,
            text=(f"idle — no events ({self._fmt_eta(now - self.last_activity)})"
                  if idle else
                  f"files {self.files_done}/{self.files_total} · "
                  f"tests {self.total_tests:,} · {self.phase}"))


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        # default to a conventional telemetry path if it exists
        cands = ["/tmp/opencode/maxcompress.telemetry.jsonl",
                 "/home/user/maxcompress/.telemetry.jsonl"]
        for c in cands:
            if os.path.exists(c):
                sys.argv.append(c)
                break
        else:
            sys.exit(1)
    path = sys.argv[1]
    root = tk.Tk()
    app = LiveViewer(root, path)
    root.mainloop()


if __name__ == "__main__":
    main()
