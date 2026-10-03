"""
PyQt5 front end for Alti.py: pick a folder with Altiverb IRs (Altiverb 6 IR folders and/or Altiverb 7
.irbulk files, anywhere inside it) and an output folder, and convert. The conversion itself is
Alti.py's, so the result is the same as on the command line.

    pip install PyQt5 numpy soundfile pillow
    python3 AltiGUI.py
"""

import os
import struct
import sys
import types

from PyQt5.QtCore import QSettings, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QVBoxLayout,
    QWidget,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Alti  # noqa: E402

STAT_KEYS = ("decoded", "wav read", "failed", "rate guessed", "gain missing", "pictures",
             "pictures borrowed")


class Cancelled(Exception):
    pass


class ProgressStats(dict):
    """Alti's stats dict, reporting every converted channel and stopping when cancelled."""

    def __init__(self, on_channel, cancelled):
        super().__init__({k: 0 for k in STAT_KEYS})
        self.on_channel = on_channel
        self.cancelled = cancelled

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key in ("decoded", "wav read", "failed"):
            self.on_channel()
        if self.cancelled():
            raise Cancelled()


class LogFile:
    """Alti writes its log to a file; also show each line in the window."""

    def __init__(self, fh, emit):
        self.fh = fh
        self.emit = emit

    def write(self, text):
        self.fh.write(text)
        if text.strip():
            self.emit(text.rstrip())


def find_sources(folder):
    """Altiverb 7 .irbulk files, and the folder itself if it holds Altiverb 6 channel files."""
    bulks, channels = [], 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".irbulk"):
                bulks.append(os.path.join(root, f))
            elif Alti.CHANNEL_SUFFIX.search(f) and not f.startswith("."):
                channels += 1
    for b in bulks:
        try:
            with open(b, "rb") as fh:
                head = fh.read(0x58)
                if head[:8] == b"_IRBLK3_":
                    fh.seek(struct.unpack_from("<Q", head, 0x50)[0])
                    channels += struct.unpack("<Q", fh.read(8))[0]
        except (OSError, struct.error):
            pass
    return bulks, channels


class Worker(QThread):
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    done = pyqtSignal(dict, str)

    def __init__(self, src, out, args):
        super().__init__()
        self.src, self.out, self.args = src, out, args
        self.count = 0
        self.cancel = False

    def step(self):
        self.count += 1
        self.progress.emit(self.count)

    def run(self):
        stats = ProgressStats(self.step, lambda: self.cancel)
        status = "ok"
        try:
            os.makedirs(self.out, exist_ok=True)
            bulks, _ = find_sources(self.src)
            with open(os.path.join(self.out, "conversion log.txt"), "w") as fh:
                log = LogFile(fh, self.message.emit)
                self.message.emit("Pastas Altiverb 6: " + self.src)
                Alti.convert_tree(self.src, self.out, self.args, stats, log)
                for b in bulks:
                    self.message.emit("Altiverb 7 .irbulk: " + b)
                    Alti.convert_irbulk(b, self.out, self.args, stats, log)
                fh.write("\n%s\n" % dict(stats))
        except Cancelled:
            status = "cancelled"
        except Exception as e:  # show anything unexpected instead of dying silently
            status = "Erro: %s" % e
        self.done.emit(dict(stats), status)


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Altiverb IR → WAV")
        self.settings = QSettings("Altiverb6_Wav", "AltiGUI")
        self.worker = None

        self.src = QLineEdit(self.settings.value("src", ""))
        self.out = QLineEdit(self.settings.value("out", ""))
        src_btn = QPushButton("Escolher…")
        out_btn = QPushButton("Escolher…")
        src_btn.clicked.connect(lambda: self.pick(self.src, "Pasta com IRs do Altiverb"))
        out_btn.clicked.connect(lambda: self.pick(self.out, "Pasta de saída"))

        self.float_out = QRadioButton("32-bit float")
        self.pcm24 = QRadioButton("24-bit PCM")
        self.float_out.setChecked(True)
        self.raw = QCheckBox("Não aplicar os ganhos de canal do info.iri (Altiverb 6)")
        self.peak = QDoubleSpinBox()
        self.peak.setRange(-24.0, 0.0)
        self.peak.setSingleStep(0.1)
        self.peak.setValue(-0.1)
        self.peak.setSuffix(" dBFS")

        form = QFormLayout()
        form.addRow("Pasta de origem", self.row(self.src, src_btn))
        form.addRow("Pasta de saída", self.row(self.out, out_btn))
        form.addRow("Formato", self.row(self.float_out, self.pcm24))
        form.addRow("Pico por pasta de IR", self.peak)
        form.addRow("", self.raw)

        self.bar = QProgressBar()
        self.status = QLabel("Escolha a pasta de origem e a pasta de saída.")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.start = QPushButton("Converter")
        self.start.clicked.connect(self.toggle)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.start)
        layout.addWidget(self.bar)
        layout.addWidget(self.status)
        layout.addWidget(self.log)
        self.resize(760, 520)

    @staticmethod
    def row(*widgets):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in widgets:
            h.addWidget(x)
        return w

    def pick(self, field, title):
        path = QFileDialog.getExistingDirectory(self, title, field.text() or os.path.expanduser("~"))
        if path:
            field.setText(path)

    def toggle(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel = True
            self.status.setText("Parando…")
            return
        src, out = self.src.text().strip(), self.out.text().strip()
        if not os.path.isdir(src):
            QMessageBox.warning(self, "Origem", "Escolha uma pasta de origem existente.")
            return
        if not out:
            QMessageBox.warning(self, "Saída", "Escolha uma pasta de saída.")
            return
        if os.path.abspath(out).startswith(os.path.abspath(src) + os.sep) or os.path.abspath(out) == os.path.abspath(src):
            QMessageBox.warning(self, "Saída", "A pasta de saída deve ficar fora da pasta de origem.")
            return
        self.settings.setValue("src", src)
        self.settings.setValue("out", out)

        self.status.setText("Procurando arquivos…")
        QApplication.processEvents()
        bulks, total = find_sources(src)
        if not total:
            QMessageBox.information(self, "Nada para converter",
                                    "Nenhum canal de IR do Altiverb nem arquivo .irbulk nesta pasta.")
            self.status.setText("Nada para converter.")
            return
        self.bar.setRange(0, total)
        self.bar.setValue(0)
        self.log.clear()
        self.status.setText("Convertendo %d canais (%d arquivos .irbulk)…" % (total, len(bulks)))

        args = types.SimpleNamespace(raw=self.raw.isChecked(), pcm24=self.pcm24.isChecked(),
                                     peak=self.peak.value())
        self.worker = Worker(src, out, args)
        self.worker.progress.connect(self.bar.setValue)
        self.worker.message.connect(self.log.appendPlainText)
        self.worker.done.connect(self.finished)
        self.worker.start()
        self.start.setText("Parar")

    def finished(self, stats, status):
        self.start.setText("Converter")
        summary = ("%d decodificados, %d WAVs lidos, %d falhas, %d imagens"
                   % (stats["decoded"], stats["wav read"], stats["failed"], stats["pictures"]))
        if status == "ok":
            self.bar.setValue(self.bar.maximum())
            self.status.setText("Concluído: " + summary)
        else:
            self.status.setText("%s: %s" % ({"cancelled": "Cancelado"}.get(status, status), summary))
        self.log.appendPlainText("\n" + summary + "\nLog: " + os.path.join(self.out.text(), "conversion log.txt"))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel = True
            self.worker.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
