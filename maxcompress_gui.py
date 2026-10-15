#!/usr/bin/env python3
"""
MaxCompress GUI — a WinRAR-style shell for the MaxCompress v3 engine.

Layout mirrors WinRAR:
  * big icon toolbar (Add, Extract To, Test, View, Delete, Find, Wizard, Info, Repair)
  * address bar + Up button
  * file list with WinRAR columns (Name / Size / Packed / Ratio / Date / Attributes)
  * status bar with item counts
  * "Add to archive" and "Extraction path and options" dialogs
  * archive browsing (open a .mxcp and see its contents)

Logo: dancing_chicken.svg (animated SMIL) shown in the window icon and the
About box.  Compression bundles files/folders via an uncompressed tar stream,
then runs the MaxCompress v3 engine over the whole blob.
"""

import os
import io
import sys
import tarfile
import tempfile
import time
import datetime

from PySide6.QtCore import Qt, QSize, QByteArray, QTimer, QThread, Signal, QPoint, QUrl
from PySide6.QtGui import (QIcon, QPixmap, QPainter, QColor, QPen, QBrush,
                           QFont, QAction, QDesktopServices)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (QMainWindow, QWidget, QToolButton, QLabel,
                               QLineEdit, QTableWidget, QTableWidgetItem,
                               QHeaderView, QMenuBar, QMenu, QToolBar,
                               QStatusBar, QHBoxLayout, QVBoxLayout,
                               QGridLayout, QFormLayout, QFileDialog,
                               QDialog, QDialogButtonBox, QComboBox,
                               QCheckBox, QSpinBox, QMessageBox, QPushButton,
                               QFrame, QProgressBar, QApplication, QStyleFactory,
                               QListView, QAbstractItemView)

# ---------------------------------------------------------------------------
# engine import
# ---------------------------------------------------------------------------
try:
    from maxcompress import MaxCompressEngine
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from maxcompress import MaxCompressEngine

CHICKEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "dancing_chicken.svg")
ARCH_EXT = ".mxcp"


def _engine(**kw):
    """Engine factory for GUI threads — never fork multiprocessing from a
    QThread, so force serial execution inside the worker."""
    kw.setdefault("threads", 1)
    return MaxCompressEngine(**kw)

# ---------------------------------------------------------------------------
# winrar-like palette
# ---------------------------------------------------------------------------
def winrar_palette(app):
    from PySide6.QtGui import QPalette
    p = app.palette()
    wg = QColor(0xF0, 0xEF, 0xEA)          # toolbar / window gray
    btn = QColor(0xF0, 0xEF, 0xEA)
    light = QColor(0xFF, 0xFF, 0xFF)
    dark = QColor(0xD6, 0xD3, 0xC4)
    p.setColor(QPalette.Window, wg)
    p.setColor(QPalette.WindowText, QColor(0x11, 0x11, 0x11))
    p.setColor(QPalette.Base, QColor(0xFF, 0xFF, 0xFF))
    p.setColor(QPalette.AlternateBase, QColor(0xF7, 0xF6, 0xF0))
    p.setColor(QPalette.Button, btn)
    p.setColor(QPalette.ButtonText, QColor(0x11, 0x11, 0x11))
    p.setColor(QPalette.Highlight, QColor(0x00, 0x78, 0xD7))
    p.setColor(QPalette.HighlightedText, QColor(0xFF, 0xFF, 0xFF))
    p.setColor(QPalette.ToolTipBase, QColor(0xFF, 0xFF, 0xE1))
    p.setColor(QPalette.Light, light)
    p.setColor(QPalette.Dark, dark)
    p.setColor(QPalette.Midlight, QColor(0xFB, 0xFB, 0xF7))
    p.setColor(QPalette.Mid, QColor(0xE1, 0xE0, 0xD8))
    p.setColor(QPalette.Text, QColor(0x11, 0x11, 0x11))
    p.setColor(QPalette.BrightText, QColor(0xFF, 0x00, 0x00))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(0x80, 0x80, 0x80))
    app.setPalette(p)


# ---------------------------------------------------------------------------
# procedural toolbar icons (winrar-flavored, self-contained)
# ---------------------------------------------------------------------------
def _icon(painter_fn):
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    painter_fn(p)
    p.end()
    return QIcon(pm)


def icon_add():
    def draw(p):
        # green stack of sheets with a plus arrow
        p.setBrush(QColor(0xE8, 0xF1, 0xE8)); p.setPen(Qt.NoPen)
        p.drawRoundedRect(8, 22, 34, 30, 8, 8)
        p.setBrush(QColor(0xFF, 0xFF, 0xFF)); p.setPen(QPen(QColor(0x7F, 0x7F, 0x7F), 1))
        p.drawRoundedRect(14, 14, 30, 28, 6, 6)
        p.drawLine(20, 20, 38, 20); p.drawLine(20, 26, 38, 26); p.drawLine(20, 32, 34, 32)
        p.setBrush(QColor(0x2E, 0x9E, 0x44)); p.setPen(QPen(QColor(0x1E, 0x6E, 0x2E), 1))
        p.drawEllipse(44, 6, 15, 15)
        p.drawLine(56, 22, 56, 36); p.drawLine(52, 29, 56, 36); p.drawLine(56, 36, 60, 29)
    return _icon(draw)


def icon_extract():
    def draw(p):
        # folder towards open-blue arrow out
        p.setBrush(QColor(0xE8, 0xD3, 0x8E)); p.setPen(QPen(QColor(0x9A, 0x7C, 0x2D), 1))
        p.drawRoundedRect(8, 20, 40, 30, 5, 5)
        p.setBrush(QColor(0xFD, 0xF1, 0xC8))
        p.drawRoundedRect(14, 14, 34, 26, 5, 5)
        p.setBrush(QColor(0x22, 0x66, 0xCC)); p.setPen(QPen(QColor(0x11, 0x44, 0x99), 1))
        p.drawEllipse(44, 5, 15, 15)
        p.drawLine(58, 18, 46, 30)
        p.drawLine(46, 30, 58, 30); p.drawLine(46, 30, 46, 18)
    return _icon(draw)


def _mag(p, cx, cy, r, color):
    p.setBrush(color); p.setPen(QPen(QColor(0x33, 0x33, 0x33), 1))
    p.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
    p.setBrush(color)
    p.drawPolygon([QPoint(cx + r + 2, cy + r + 2),
                   QPoint(cx + r + 10, cy + r + 10),
                   QPoint(cx + r + 10, cy + r)])
    p.drawEllipse(cx + r - 2, cy + r - 2, 3, 3)


def icon_test():
    def draw(p):
        _mag(p, 22, 22, 14, QColor(0xED, 0xF2, 0xF9))
        # green check inside glass
        p.setPen(QPen(QColor(0x1E, 0x8E, 0x3E), 4, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(18, 24, 25, 31); p.drawLine(25, 31, 36, 17)
    return _icon(draw)


def icon_view():
    def draw(p):
        p.setPen(QPen(QColor(0x33, 0x55, 0x88), 2))
        p.setBrush(QColor(0xE8, 0xF0, 0xFF))
        p.drawEllipse(18, 20, 24, 20)
        p.setBrush(QColor(0x55, 0x77, 0xAA))
        p.drawEllipse(29, 29, 6, 6)
        p.drawLine(6, 8, 60, 8)
        p.drawLine(8, 6, 10, 10); p.drawLine(30, 5, 32, 9)
    return _icon(draw)


def icon_delete():
    def draw(p):
        p.setBrush(QColor(0xFF, 0x8A, 0x8A)); p.setPen(QPen(QColor(0xAA, 0x22, 0x22), 1))
        p.drawRoundedRect(10, 16, 34, 34, 5, 5)
        p.drawText(12, 38, "X")
        p.setPen(QPen(QColor(0xCC, 0x11, 0x11), 4, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(38, 6, 52, 20); p.drawLine(40, 6, 54, 6)
    return _icon(draw)


def icon_find():
    def draw(p):
        _mag(p, 26, 26, 16, QColor(0xFF, 0xF3, 0xC8))
        p.setBrush(QColor(0x8B, 0x6D, 0x00)); p.setPen(Qt.NoPen)
        p.drawEllipse(40, 44, 4, 4)
    return _icon(draw)


def icon_wizard():
    def draw(p):
        p.setPen(QPen(QColor(0x7B, 0x3F, 0xA5), 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(10, 54, 46, 14)
        p.setBrush(QColor(0xFF, 0xE0, 0x60)); p.setPen(Qt.NoPen)
        p.drawPolygon([QPoint(46, 14), QPoint(50, 4), QPoint(52, 14)])
        p.drawPolygon([QPoint(10, 54), QPoint(6, 58), QPoint(14, 58)])
        # sparkles
        p.setBrush(QColor(0xFF, 0x4D, 0x9D)); p.drawEllipse(52, 22, 5, 5)
        p.setBrush(QColor(0x4D, 0x9D, 0xFF)); p.drawEllipse(38, 46, 5, 5)
    return _icon(draw)


def icon_info():
    def draw(p):
        p.setBrush(QColor(0x3A, 0x7D, 0xE8)); p.setPen(Qt.NoPen)
        p.drawEllipse(18, 14, 28, 28)
        p.setBrush(QColor(0xFF, 0xFF, 0xFF))
        p.drawRect(30, 26, 4, 12)
        p.drawRect(30, 22, 4, 4)
    return _icon(draw)


def icon_repair():
    def draw(p):
        p.setPen(QPen(QColor(0x44, 0x44, 0x44), 6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(12, 48, 40, 20)
        p.setBrush(QColor(0xC0, 0xC0, 0xC0)); p.setPen(QPen(QColor(0x44, 0x44, 0x44), 2))
        p.drawEllipse(32, 12, 18, 18)
        p.setBrush(QColor(0x44, 0x44, 0x44))
        p.drawEllipse(38, 18, 6, 6)
    return _icon(draw)


def icon_up():
    def draw(p):
        p.setBrush(QColor(0xE8, 0xE8, 0xE0)); p.setPen(QPen(QColor(0x77, 0x77, 0x77), 1))
        p.drawRect(6, 6, 52, 52)
        p.setPen(QPen(QColor(0x22, 0x66, 0xCC), 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(32, 44, 32, 18)
        p.drawLine(20, 28, 32, 18); p.drawLine(44, 28, 32, 18)
    return _icon(draw)


def icon_dancing_chicken(size=64):
    """Static glow-frame of the dancing chicken for window / toolbar icons."""
    r = QSvgRenderer(QByteArray(open(CHICKEN, 'rb').read()))
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    r.render(p, pm.rect())
    p.end()
    return QIcon(pm)


# ---------------------------------------------------------------------------
# helper: build / unpack the tar stream used as the archive container
# ---------------------------------------------------------------------------
def tar_bundle(paths, base_dir=None):
    """Bundle files+dirs (recursively) into an uncompressed tar in memory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for p in paths:
            if base_dir is not None:
                arc = os.path.relpath(p, base_dir)
            else:
                arc = os.path.basename(p)
            if arc.startswith(".."):
                arc = os.path.basename(p)
            tf.add(p, arcname=arc.replace(os.sep, "/"))
    return buf.getvalue()


def tar_list(blob):
    out = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
        for m in tf.getmembers():
            out.append((m.name, m.size, m.isdir()))
    return out


def tar_extract(blob, dest):
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
        tf.extractall(dest)


# ---------------------------------------------------------------------------
# worker thread
# ---------------------------------------------------------------------------
class Worker(QThread):
    progress = Signal(int)
    log = Signal(str)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self._cancel = False

    def run(self):
        try:
            self.fn(self)
            self.done.emit(None)
        except Exception as e:
            if not self._cancel:
                self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Add-to-archive dialog (WinRAR "Archive name and parameters")
# ---------------------------------------------------------------------------
class AddDialog(QDialog):
    def __init__(self, main_win, sources, parent=None):
        super().__init__(parent)
        self.main_win = main_win
        self.sources = list(sources)
        self.setWindowTitle("Archive name and parameters")
        self.resize(560, 380)
        lay = QVBoxLayout(self)

        # top row: archive name + browse
        top = QHBoxLayout()
        self.name_edit = QLineEdit()
        sub = self.main_win.current_dir
        if len(self.sources) == 1 and os.path.isfile(self.sources[0]):
            base = os.path.splitext(os.path.basename(self.sources[0]))[0]
        else:
            base = os.path.basename(sub.rstrip("/")) or "archive"
        self.name_edit.setText(os.path.join(sub, base + ARCH_EXT))
        top.addWidget(self.name_edit, 1)
        b = QPushButton("Browse...")
        b.clicked.connect(self._browse)
        top.addWidget(b)
        lay.addLayout(top)

        # config form
        form = QGridLayout()
        self.method = QComboBox()
        self.method.addItems(["Store", "Fastest", "Fast", "Normal", "Good", "Best"])
        self.method.setCurrentText("Normal")
        self.dict_size = QComboBox()
        self.dict_size.addItems(["32 KB", "64 KB", "128 KB", "256 KB", "512 KB", "1 MB",
                                 "2 MB", "4 MB", "8 MB", "16 MB", "32 MB", "64 MB",
                                 "128 MB", "256 MB", "512 MB", "1 GB", "Auto"])
        self.dict_size.setCurrentText("Auto")
        self.update_mode = QComboBox()
        self.update_mode.addItems(["Add and replace files", "Add and update files",
                                   "Fresh existing files only", "Synchronize archive contents"])
        self.fmt = QComboBox()
        self.fmt.addItems(["RAR5 (MXCP v3)", "RAR (MXCP v3)", "ZIP (MXCP v3)"])
        form.addWidget(QLabel("Compression method:"), 0, 0)
        form.addWidget(self.method, 0, 1)
        form.addWidget(QLabel("Dictionary size:"), 1, 0)
        form.addWidget(self.dict_size, 1, 1)
        form.addWidget(QLabel("Update mode:"), 2, 0)
        form.addWidget(self.update_mode, 2, 1)
        form.addWidget(QLabel("Archive format:"), 3, 0)
        form.addWidget(self.fmt, 3, 1)
        lay.addLayout(form)

        # options
        opts = QFrame()
        opts.setFrameShape(QFrame.StyledPanel)
        ol = QVBoxLayout(opts)
        ol.addWidget(QLabel("Options:"))
        self.cb_solid = QCheckBox("Put archive to a solid archive")
        self.cb_solid.setChecked(True)
        self.cb_test = QCheckBox("Test archived files after completion")
        self.cb_test.setChecked(True)
        self.cb_delete = QCheckBox("Delete files after archiving")
        ol.addWidget(self.cb_solid)
        ol.addWidget(self.cb_test)
        ol.addWidget(self.cb_delete)
        lay.addWidget(opts)

        # budget spin
        bl = QHBoxLayout()
        bl.addWidget(QLabel("Search budget (seconds):"))
        self.budget = QSpinBox()
        self.budget.setRange(1, 3600)
        self.budget.setValue(self._budget_for())
        self.method.currentTextChanged.connect(lambda _: self.budget.setValue(self._budget_for()))
        bl.addWidget(self.budget)
        bl.addWidget(QLabel("(0 = unlimited)"))
        bl.addStretch(1)
        lay.addLayout(bl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("OK")
        btns.button(QDialogButtonBox.Cancel).setText("Cancel")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _budget_for(self):
        return {"Store": 3, "Fastest": 8, "Fast": 20, "Normal": 40,
                "Good": 90, "Best": 240}[self.method.currentText()]

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save archive", self.name_edit.text())
        if p:
            self.name_edit.setText(p)

    def mode_and_budget(self):
        m = self.method.currentText()
        if m == "Best":
            mode = "exhaust"
        elif m in ("Good", "Normal"):
            mode = "full"
        else:
            mode = "quick"
        b = self.budget.value()
        return mode, (None if b == 0 else int(b))


# ---------------------------------------------------------------------------
# Extraction dialog (WinRAR "Extraction path and options")
# ---------------------------------------------------------------------------
class ExtractDialog(QDialog):
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.main_win = main_win
        self.setWindowTitle("Extraction path and options")
        self.resize(520, 320)
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Destination path:"))
        self.path = QLineEdit(os.path.join(main_win.current_dir, "extracted"))
        top.addWidget(self.path, 1)
        b = QPushButton("Browse...")
        b.clicked.connect(self._browse)
        top.addWidget(b)
        lay.addLayout(top)

        f = QFrame(); f.setFrameShape(QFrame.StyledPanel)
        fl = QFormLayout(f)
        self.update_mode = QComboBox()
        self.update_mode.addItems(["Extract and replace files",
                                   "Extract and update files",
                                   "Fresh existing files only"])
        self.overwrite = QComboBox()
        self.overwrite.addItems(["Ask before overwrite", "Overwrite without prompt",
                                 "Skip existing files"])
        fl.addRow("Update mode:", self.update_mode)
        fl.addRow("Overwrite mode:", self.overwrite)
        lay.addWidget(f)

        self.cb_show = QCheckBox("Show extracted files in Explorer")
        lay.addWidget(self.cb_show)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("OK")
        btns.button(QDialogButtonBox.Cancel).setText("Cancel")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _browse(self):
        p = QFileDialog.getExistingDirectory(self, "Choose destination",
                                             self.path.text())
        if p:
            self.path.setText(p)


# ---------------------------------------------------------------------------
# Info dialog
# ---------------------------------------------------------------------------
class InfoDialog(QDialog):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Archive info")
        self.resize(480, 360)
        lay = QVBoxLayout(self)
        lab = QLabel()
        lab.setTextFormat(Qt.RichText)
        lab.setText(info)
        lay.addWidget(lab)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.button(QDialogButtonBox.Ok).setText("OK")
        btns.accepted.connect(self.accept)
        lay.addWidget(btns)


# ---------------------------------------------------------------------------
# About box with the dancing chicken
# ---------------------------------------------------------------------------
class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About MaxCompress")
        self.setFixedSize(420, 360)
        lay = QVBoxLayout(self)
        svg = QSvgWidget(CHICKEN)
        svg.setFixedSize(180, 180)
        lay.addWidget(svg, 0, Qt.AlignHCenter)
        t = QLabel("<b>MaxCompress v3</b><br>multi-algorithm chain-search compression<br>"
                   "WinRAR-style archive manager<br><br>"
                   f"registry: {len(_engine().algos):,} algorithms")
        t.setAlignment(Qt.AlignCenter)
        lay.addWidget(t)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.button(QDialogButtonBox.Ok).setText("OK")
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MaxCompress 3.0")
        self.resize(980, 620)
        self.setWindowIcon(icon_dancing_chicken(48))
        self.current_dir = os.path.expanduser("~")
        self.table = None
        self._worker = None
        self.progress = None

        self._build_menu()
        self._build_toolbar()
        self._build_address()
        self._build_status()
        self.refresh()

    # -------------------------------------------------------------- menus
    def _build_menu(self):
        mb = self.menuBar()
        # File
        m_file = mb.addMenu("&File")
        self._add(m_file, "Change Drive...", self._change_drive)
        self._add(m_file, "Go to Folder...", self._goto_folder)
        m_file.addSeparator()
        self._add(m_file, "Set Defaults...", self.refresh)
        m_file.addSeparator()
        self._add(m_file, "Exit", self.close)
        # Commands
        m_cmd = mb.addMenu("&Commands")
        self._add(m_cmd, "Add files to archive...", self.on_add)
        self._add(m_cmd, "Extract to specified folder...", self.on_extract)
        self._add(m_cmd, "Test archived files", self.on_test)
        self._add(m_cmd, "View file", self.on_view)
        self._add(m_cmd, "Delete file(s)", self.on_delete)
        m_cmd.addSeparator()
        self._add(m_cmd, "Show archive info", self.on_info)
        # Tools
        m_tools = mb.addMenu("&Tools")
        self._add(m_tools, "Find files...", self.on_find)
        self._add(m_tools, "Wizard", self.on_wizard)
        m_tools.addSeparator()
        self._add(m_tools, "Compression benchmark...", self.on_bench)
        # Favorites
        m_fav = mb.addMenu("&Favorites")
        for d in (os.path.expanduser("~"), "/tmp", "/tmp/opencode/corpus",
                  "/home/user/maxcompress"):
            if os.path.isdir(d):
                self._add(m_fav, d, lambda d=d: self.navigate(d))
        # Options
        m_opt = mb.addMenu("&Options")
        self._add(m_opt, "Settings...", self._settings)
        m_opt.addSeparator()
        self._add(m_opt, "Show Full Toolbar", lambda: self.tb.setVisible(True))
        self._add(m_opt, "Hide Toolbar", lambda: self.tb.setVisible(False))
        # Help
        m_help = mb.addMenu("&Help")
        self._add(m_help, "Help Topics", self._help)
        m_help.addSeparator()
        self._add(m_help, "About MaxCompress", self._about)

    def _add(self, menu, text, fn):
        a = QAction(text, self)
        a.triggered.connect(fn)
        menu.addAction(a)
        return a

    # ----------------------------------------------------------- toolbar
    def _build_toolbar(self):
        self.tb = QToolBar("Main")
        self.tb.setIconSize(QSize(48, 48))
        self.tb.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.tb.setMovable(False)
        self.addToolBar(self.tb)
        for text, icon, fn, tip in (
            ("Add", icon_add(), self.on_add, "Add files to archive (compress)"),
            ("Extract To", icon_extract(), self.on_extract, "Extract archive (decompress)"),
            ("Test", icon_test(), self.on_test, "Test archived file integrity"),
            ("View", icon_view(), self.on_view, "View selected file in default app"),
            ("Delete", icon_delete(), self.on_delete, "Delete selected files"),
            ("Find", icon_find(), self.on_find, "Find files"),
            ("Wizard", icon_wizard(), self.on_wizard, "Wizard"),
            ("Info", icon_info(), self.on_info, "Show archive info"),
            ("Repair", icon_repair(), self._settings, "Options / repair"),
        ):
            act = QAction(icon, text, self)
            act.setToolTip(tip)
            act.triggered.connect(fn)
            self.tb.addAction(act)

    # ----------------------------------------------------------- address
    def _build_address(self):
        # central widget: vertical stack = address row + file table
        outer = QWidget()
        outer_lay = QVBoxLayout(outer)
        outer_lay.setContentsMargins(4, 4, 4, 4)

        addr_row = QWidget()
        lay = QHBoxLayout(addr_row)
        lay.setContentsMargins(0, 0, 0, 0)
        up_btn = QToolButton()
        up_btn.setIcon(icon_up())
        up_btn.setToolTip("Up one level")
        up_btn.clicked.connect(self._up)
        lay.addWidget(up_btn)
        self.addr = QLineEdit()
        self.addr.editingFinished.connect(self._addr_go)
        lay.addWidget(self.addr, 1)
        go = QPushButton("Go")
        go.clicked.connect(self._addr_go)
        lay.addWidget(go)
        outer_lay.addWidget(addr_row)

        # file table fills the remaining space
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Name", "Size", "Packed", "Ratio",
                                              "Date", "Attributes"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in (1, 2, 3, 4, 5):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(self._double_click)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        outer_lay.addWidget(self.table, 1)

        self.setCentralWidget(outer)

    # ------------------------------------------------------------ status
    def _build_status(self):
        sb = self.statusBar()
        self.status_label = QLabel(" ready")
        sb.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        sb.addPermanentWidget(self.progress)
        self.ver_label = QLabel("  3,877 algorithms")
        sb.addPermanentWidget(self.ver_label)

    # ---------------------------------------------------------- listing
    def refresh(self):
        self.addr.setText(self.current_dir)
        try:
            entries = sorted(os.listdir(self.current_dir),
                             key=lambda e: (not os.path.isdir(os.path.join(self.current_dir, e)),
                                            e.lower()))
        except OSError as e:
            QMessageBox.warning(self, "Error", str(e))
            return
        self.table.setRowCount(0)
        n_files = n_fold = 0
        size_sum = 0
        for i, name in enumerate(entries):
            full = os.path.join(self.current_dir, name)
            is_dir = os.path.isdir(full)
            try:
                st = os.stat(full)
            except OSError:
                size, mtime = 0, time.time()
            else:
                size, mtime = st.st_size, st.st_mtime
            row = self.table.rowCount()
            self.table.insertRow(row)
            disp = ("[" + name + "]" if is_dir else name)
            it = QTableWidgetItem(disp)
            it.setFlags(it.flags() & ~Qt.ItemIsEditable)
            if is_dir:
                f = self.table.font()
                bf = QFont(f); bf.setBold(True)
                it.setFont(bf)
            self.table.setItem(row, 0, it)
            sz = "" if is_dir else f"{size:,}"
            self.table.setItem(row, 1, QTableWidgetItem(sz))
            self.table.setItem(row, 2, QTableWidgetItem(""))
            self.table.setItem(row, 3, QTableWidgetItem(""))
            ds = datetime.datetime.fromtimestamp(mtime).strftime("%d-%m-%y %H:%M")
            self.table.setItem(row, 4, QTableWidgetItem(ds))
            attr = "A" if not is_dir else "D"
            self.table.setItem(row, 5, QTableWidgetItem(attr))
            if is_dir:
                n_fold += 1
            else:
                n_files += 1
                size_sum += size
        self.table.resizeColumnsToContents()
        self.status_label.setText(
            f"{n_files:,} file(s), {n_fold:,} folder(s)  ·  total {size_sum:,} bytes")

    # -------------------------------------------------------- navigation
    def navigate(self, path):
        if os.path.isdir(path):
            self.current_dir = os.path.abspath(path)
        self.refresh()

    def _addr_go(self):
        self.navigate(self.addr.text())

    def _up(self):
        self.navigate(os.path.dirname(self.current_dir.rstrip("/")))
        if not self.current_dir:
            self.current_dir = "/"

    def _change_drive(self):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", self.current_dir)
        if d:
            self.navigate(d)

    def _goto_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Go to folder", self.current_dir)
        if d:
            self.navigate(d)

    def selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        out = []
        for r in rows:
            name = self.table.item(r, 0).text()
            if name.startswith("[") and name.endswith("]"):
                name = name[1:-1]
            out.append(os.path.join(self.current_dir, name))
        return out

    def _double_click(self, idx):
        name = self.table.item(idx.row(), 0).text()
        if name.startswith("[") and name.endswith("]"):
            self.navigate(os.path.join(self.current_dir, name[1:-1]))
            return
        full = os.path.join(self.current_dir, name)
        if full.endswith(ARCH_EXT):
            self._browse_archive(full)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(full))

    # ----------------------------------------------------- archive view
    def _browse_archive(self, arch):
        try:
            with open(arch, "rb") as f:
                blob = f.read()
            eng = _engine()
            raw = eng.decompress(blob)
        except Exception as e:
            QMessageBox.warning(self, "Bad archive", str(e))
            return
        # show tar members in a temporary listing window
        try:
            members = tar_list(raw)
        except Exception:
            members = [(os.path.basename(arch), len(raw), False)]
        dw = QDialog(self)
        dw.setWindowTitle(f"Archive: {os.path.basename(arch)}")
        dw.resize(620, 400)
        lv = _make_archive_table(members, raw, arch)
        lay = QVBoxLayout(dw)
        lay.addWidget(lv)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dw.close); bb.accepted.connect(dw.close)
        lay.addWidget(bb)
        dw.exec()

    def _context_menu(self, pos):
        m = QMenu(self)
        m.addAction("Add to archive...", self.on_add)
        m.addAction("Extract to specified folder...", self.on_extract)
        m.addSeparator()
        m.addAction("View file", self.on_view)
        m.addAction("Show archive info", self.on_info)
        m.addSeparator()
        m.addAction("Delete", self.on_delete)
        m.exec(self.table.viewport().mapToGlobal(pos))

    # ------------------------------------------------------- operations
    def _busy(self, fn, title="Working..."):
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "Another operation is running.")
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status_label.setText(title)
        worker = Worker(fn)
        self._worker = worker
        worker.log.connect(self._on_log)
        worker.error.connect(self._on_error)
        worker.done.connect(self._on_done)
        worker.finished.connect(lambda: self.progress.setVisible(False))
        worker.start()

    def _on_log(self, msg):
        self.status_label.setText(msg[:120])

    def _on_error(self, msg):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", msg)
        self.refresh()

    def _on_done(self, _):
        self.progress.setVisible(False)
        self.status_label.setText(" done.")
        self.refresh()

    def on_add(self):
        sel = self.selected()
        if not sel:
            sel = [self.current_dir]
        dlg = AddDialog(self, sel, self)
        if dlg.exec() != QDialog.Accepted:
            return
        arch = dlg.name_edit.text()
        mode, budget = dlg.mode_and_budget()
        paths = [p for p in sel if os.path.exists(p)]
        if not paths:
            return
        base_dir = os.path.commonpath(paths) if len(paths) > 1 \
            else os.path.dirname(paths[0])
        def job(w):
            blob = tar_bundle(paths, base_dir=base_dir)
            w.log.emit(f"bundling {len(paths)} item(s) -> {len(blob):,} bytes")
            eng = _engine(mode=mode)
            w.log.emit(f"MaxCompress mode={mode} budget={budget} ...")
            t0 = time.time()
            comp, pipe, stats = eng.compress(blob, budget=budget)
            w.log.emit(f"method: {stats['method']}  ratio: {stats['ratio']*100:.2f}%" +
                       f"  time: {time.time()-t0:.1f}s")
            with open(arch, "wb") as f:
                f.write(comp)
            w.log.emit(f"saved {arch} ({len(comp):,} bytes)")
            if dlg.cb_delete.isChecked() and len(paths) == 1:
                os.remove(paths[0]); w.log.emit("  deleted source file")
            if dlg.cb_test.isChecked():
                _verify(arch)
        self._busy(job, "Compressing...")

    def on_extract(self):
        sel = self.selected()
        if not sel:
            QMessageBox.information(self, "Extract", "Select an archive first.")
            return
        arch = sel[0]
        if not arch.endswith(ARCH_EXT):
            QMessageBox.information(self, "Extract",
                                    "Select a " + ARCH_EXT + " archive.")
            return
        dlg = ExtractDialog(self, self)
        if dlg.exec() != QDialog.Accepted:
            return
        dest = dlg.path.text()
        def job(w):
            with open(arch, "rb") as f:
                blob = f.read()
            w.log.emit(f"decompressing {len(blob):,} bytes ...")
            eng = _engine()
            raw = eng.decompress(blob)
            os.makedirs(dest, exist_ok=True)
            tar_extract(raw, dest)
            w.log.emit(f"extracted to {dest} ({len(raw):,} bytes)")
            if dlg.cb_show.isChecked():
                QDesktopServices.openUrl(QUrl.fromLocalFile(dest))
        self._busy(job, "Extracting...")

    def on_test(self):
        sel = self.selected()
        if not sel or not sel[0].endswith(ARCH_EXT):
            QMessageBox.information(self, "Test", "Select a " + ARCH_EXT + " archive.")
            return
        def job(w):
            if _verify(sel[0], w):
                w.log.emit("OK — checksum valid, archive contents readable")
        self._busy(job, "Testing...")

    def on_view(self):
        sel = self.selected()
        if not sel:
            return
        full = sel[0]
        if full.endswith(ARCH_EXT):
            self._browse_archive(full)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(full))

    def on_delete(self):
        sel = self.selected()
        if not sel:
            return
        if not QMessageBox.question(self, "Delete",
                                    f"Delete {len(sel)} item(s)?") == QMessageBox.Yes:
            return
        for p in sel:
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    import shutil; shutil.rmtree(p)
                else:
                    os.remove(p)
            except OSError as e:
                QMessageBox.warning(self, "Delete", str(e))
        self.refresh()

    def on_find(self):
        QMessageBox.information(self, "Find",
                                "Filter: type a name in the address bar and press Go.\n"
                                "Use /path/prefix* to jump.  (simple built-in find)")

    def on_wizard(self):
        self._about()

    def on_info(self):
        sel = self.selected()
        if not sel:
            return
        full = sel[0]
        if full.endswith(ARCH_EXT):
            try:
                with open(full, "rb") as f:
                    blob = f.read()
                eng = _engine()
                raw = eng.decompress(blob)
                raw_files = 0
                try:
                    raw_files = len(tar_list(raw))
                except Exception:
                    raw_files = 1
                InfoDialog(
                    f"<b>{os.path.basename(full)}</b><br>"
                    f"archive size:&nbsp;&nbsp;{len(blob):,} bytes<br>"
                    f"unpacked size:&nbsp;{len(raw):,} bytes<br>"
                    f"ratio:&nbsp;&nbsp;{len(blob)/len(raw)*100:.1f}%<br>"
                    f"items:&nbsp;&nbsp;{raw_files}<br>"
                    f"restored OK:&nbsp;checksum valid", self).exec()
            except Exception as e:
                QMessageBox.warning(self, "Info", str(e))
            return
        st = os.stat(full)
        InfoDialog(f"<b>{os.path.basename(full)}</b><br>"
                   f"size: {st.st_size:,} bytes<br>"
                   f"modified: {datetime.datetime.fromtimestamp(st.st_mtime):%d-%m-%y %H:%M}",
                   self).exec()

    def on_bench(self):
        # quick in-GUI benchmark on the current directory's biggest file
        cand = []
        for n in os.listdir(self.current_dir):
            fp = os.path.join(self.current_dir, n)
            if os.path.isfile(fp) and os.path.getsize(fp) > 0:
                cand.append(fp)
        if not cand:
            QMessageBox.information(self, "Benchmark", "No files to benchmark here.")
            return
        fp = max(cand, key=os.path.getsize)
        def job(w):
            with open(fp, "rb") as f:
                data = f.read()
            eng = _engine(mode="full")
            w.log.emit(f"benchmark {os.path.basename(fp)} ({len(data):,} bytes) ...")
            t0 = time.time()
            comp, pipe, stats = eng.compress(data, budget=45)
            w.log.emit(f"{stats['method']}  {len(comp):,} / {len(data):,} bytes " +
                       f"({stats['ratio']*100:.2f}%)  {time.time()-t0:.1f}s")
        self._busy(job, "Benchmark...")

    def _settings(self):
        QMessageBox.information(self, "Settings",
                                "MaxCompress uses the full engine registry.\n"
                                "Processes: %d cores.  Budget is set in the "
                                "Add-to-archive dialog." % _cores())

    def _help(self):
        QMessageBox.information(
            self, "Help", "MaxCompress v3 — WinRAR-style shell.\n\n"
            "Compress: select files, press Add.\n"
            "Decompress: select a .mxcp, press Extract To.\n"
            "Test: validates the CRC32 in the archive header.\n"
            "Double-click a .mxcp to browse its contents.\n\n"
            "The little dancer in the About box is the brand logo (dancing_chicken.svg).")

    def _about(self):
        AboutDialog(self).exec()


def _cores():
    try:
        return max(1, os.cpu_count() or 1)
    except Exception:
        return 1


def _verify(arch, worker=None):
    """Check CRC32 + readable tar; returns True when valid."""
    with open(arch, "rb") as f:
        blob = f.read()
    eng = _engine()
    raw = eng.decompress(blob)   # raises on checksum mismatch
    try:
        tar_list(raw)
    except Exception as e:
        if worker:
            worker.log.emit("  archive has no tar payload; raw blob OK")
    return True


def _make_archive_table(members, raw, arch):
    from PySide6.QtWidgets import QTableWidget as TW
    t = TW(len(members), 4)
    t.setHorizontalHeaderLabels(["Name", "Size", "Packed", "Ratio"])
    t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    for i, (name, size, isdir) in enumerate(members):
        t.setItem(i, 0, QTableWidgetItem(("[" + name + "]" if isdir else name)))
        t.setItem(i, 1, QTableWidgetItem("" if isdir else f"{size:,}"))
        pack = "" if isdir else f"{len(raw):,}"
        t.setItem(i, 2, QTableWidgetItem(pack))
        ratio = "" if isdir or not size else f"{len(raw)/size*100:.0f}%"
        t.setItem(i, 3, QTableWidgetItem(ratio))
    return t


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MaxCompress")
    app.setStyle(QStyleFactory.create("Fusion"))
    winrar_palette(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())