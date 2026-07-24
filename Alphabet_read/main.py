import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QBrush, QLinearGradient

from canvas import Canvas
from img_segment import segment_chars
from preprocess import CropPreviewWindow
from selectmodel import ModelsResultPanel
from inference_snn import load_model as snn_load, predict as snn_predict
from inference_dnn import load_model as dnn_load, predict as dnn_predict
from start_page import StartPage

# path ของไฟล์ weight อิงตำแหน่งไฟล์นี้ ไม่ใช่ cwd — รันจากที่ไหนก็ได้
BASE_DIR = Path(__file__).resolve().parent
SNN_WEIGHTS = BASE_DIR / "best_emnist_snn_vgg_direct_beta0.6.pth"
DNN_WEIGHTS = BASE_DIR / "EMNIST_DNN.pth"

# แอพรันแบบ script จากโฟลเดอร์ตัวเอง → ต้องเติม project root เพื่อ import shared/
_ROOT = BASE_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.energy_client import send_energy  # noqa: E402

GAME_NAME = "alphabet"

# คลังคำสำหรับสุ่มให้ผู้เล่นเขียน — ต้องเป็นตัวพิมพ์เล็ก a-z ล้วน
# (โมเดลรู้จักแค่ 26 คลาสนี้ ไม่มีเลข/ตัวใหญ่)
WORD_POOL = [
    "cat", "dog", "done", "small", "home", "book", "tree", "fish",
    "game", "word", "play", "code", "data", "test", "blue", "gold",
    "star", "moon", "rain", "snow", "food", "milk", "cake", "lion",
    "frog", "bear", "duck", "goat", "hand", "bird", "sun", "map",
]
NUM_TARGET_WORDS = 1

# ขนาดหน้าต่างล็อคตายตัว — ต้องเท่ากับ Tictac เป๊ะ ๆ (ห้ามแก้ข้างเดียว)
WINDOW_W, WINDOW_H = 1100, 780
CANVAS_PX = 620          # ใหญ่สุดที่ยังพอดีในกรอบ 780 (เดิม 1000 → หน้าต่างทะลุกรอบ)

# พื้นหลังหน้าเล่นจริง = gradient เดียวกับ Hub (hub/gui/leaderboard.py)
HUB_BG_TOP, HUB_BG_BOTTOM = "#3A3168", "#1E1943"
STATUS_LIGHT = "#EDEAFB"   # สีตัวหนังสือสว่าง อ่านออกบนพื้นม่วง


class InferenceWorker(QThread):
    snn_done = pyqtSignal(str, float, float, float)  # word, ms, energy, avg_conf
    dnn_done = pyqtSignal(str, float, float, float)
    all_done = pyqtSignal()

    def __init__(self, snn_net, dnn_net, tensors: list, target: str):
        super().__init__()
        self._snn_net = snn_net
        self._dnn_net = dnn_net
        self._tensors = tensors
        self._target  = target   # สตริงตัวอักษรเป้าหมาย (4 คำต่อกัน) เรียงซ้าย→ขวา

    def _run_model(self, predict_fn, net):
        """ทายทีละตัว → ต่อเป็นคำ + คิด avg confidence เทียบกับ target
        ตัวที่ทายถูก (ตรง target ตำแหน่งเดียวกัน) ใช้ conf จริง,
        ตัวผิด/ขาด/เกิน คิดเป็น 0 แล้วหารด้วยจำนวนตัวอักษรเป้าหมายทั้งหมด
        """
        t0 = time.perf_counter()
        word, total_energy = "", 0.0
        matched_confs = []
        for i, t in enumerate(self._tensors):
            idx, lbl, conf, e = predict_fn(net, t)
            word         += lbl
            total_energy += e   # energy รวมทุกตัวเหมือนเดิม ไม่ขึ้นกับถูก/ผิด
            # เทียบทีละตัวกับ target ด้วย if/else
            if i < len(self._target) and lbl == self._target[i]:
                matched_confs.append(conf[idx])   # ถูก → เก็บ conf จริง
            # else: ผิด/เกิน → ไม่เก็บ = conf 0
        elapsed_ms = (time.perf_counter() - t0) * 1000
        denom      = len(self._target) if self._target else len(self._tensors)
        avg_conf   = sum(matched_confs) / denom if denom else 0.0
        return word, elapsed_ms, total_energy, avg_conf

    def run(self):
        # SNN
        word, elapsed_ms, total_energy, avg_conf = self._run_model(snn_predict, self._snn_net)
        self.snn_done.emit(word, elapsed_ms, total_energy, avg_conf)

        # DNN
        word, elapsed_ms, total_energy, avg_conf = self._run_model(dnn_predict, self._dnn_net)
        self.dnn_done.emit(word, elapsed_ms, total_energy, avg_conf)

        self.all_done.emit()


def _btn_style(bg: str, hover: str, pressed: str) -> str:
    return f"""
        QPushButton {{
            background-color: {bg};
            color: white;
            font-size: 14px;
            font-weight: bold;
            border-radius: 6px;
            padding: 0 24px;
        }}
        QPushButton:hover {{ background-color: {hover}; }}
        QPushButton:pressed {{ background-color: {pressed}; }}
        QPushButton:disabled {{ background-color: #b0b0b0; }}
    """


class GamePage(QWidget):
    """หน้าวาด + ผลโมเดล (โครงเดิม) + ปุ่มจบเกม"""

    finished = pyqtSignal()  # กดจบเกม → กลับหน้าใส่ชื่อ

    def __init__(self):
        super().__init__()

        self._snn_net      = None
        self._dnn_net      = None
        self._worker       = None
        self._crop_window  = CropPreviewWindow()
        self._player_name  = ""
        self._snn_energy_j = 0.0
        self._target_words = []   # 4 คำที่สุ่มให้เขียน
        self._target_str   = ""   # 4 คำต่อกันเป็นสตริงเดียว ใช้เทียบทีละตัว

        self.canvas = Canvas(CANVAS_PX, CANVAS_PX)
        self.canvas.changed.connect(self._on_canvas_changed)

        self.models_panel = ModelsResultPanel()

        self.player_label = QLabel("Player: -")
        self.player_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #8FB8FF; background: transparent;"
        )

        self.target_label = QLabel("Word")
        self.target_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #333; "
            "background: #eef3ff; border: 1px solid #cfdcff; "
            "border-radius: 6px; padding: 8px 12px;"
        )

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setFixedHeight(40)
        self.btn_clear.setStyleSheet(_btn_style("#e74c3c", "#BD0E00", "#c0392b"))
        self.btn_clear.clicked.connect(self._on_clear)

        self.btn_submit = QPushButton("Submit")
        self.btn_submit.setFixedHeight(40)
        self.btn_submit.setStyleSheet(_btn_style("#2768F5", "#1a52cc", "#1440a0"))
        self.btn_submit.clicked.connect(self._on_submit)

        self.btn_finish = QPushButton("Finish (next player)")
        self.btn_finish.setFixedHeight(40)
        self.btn_finish.setStyleSheet(_btn_style("#7f8c8d", "#6b7778", "#596263"))
        self.btn_finish.clicked.connect(self._on_finish)

        self.status_label = QLabel("Draw the letters")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {STATUS_LIGHT}; background: transparent;"
        )

        canvas_frame = QWidget()
        canvas_frame.setStyleSheet("background: white; border: 2px solid #ccc; border-radius: 4px;")
        QVBoxLayout(canvas_frame).addWidget(self.canvas)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(self.btn_clear)
        btn_row.addWidget(self.btn_submit)

        left = QVBoxLayout()
        left.setSpacing(12)
        left.addWidget(self.player_label)
        left.addWidget(self.target_label)
        left.addWidget(self.status_label)
        left.addWidget(canvas_frame, alignment=Qt.AlignCenter)
        left.addStretch()

        right = QVBoxLayout()
        right.setSpacing(12)
        right.addLayout(btn_row)
        right.addWidget(self.models_panel)
        right.addWidget(self.btn_finish)
        right.addStretch()

        # ครอบเนื้อหาไว้ในกล่องกว้างสูงสุด ~1160 แล้วจัดกลาง — จอใหญ่/เต็มจอ panel จะไม่ยืดโหว่
        content = QWidget()
        content.setStyleSheet("background: transparent;")   # กัน #f0f0f0 แผ่ทับ gradient
        content.setMaximumWidth(1160)
        content_lay = QHBoxLayout(content)
        content_lay.setContentsMargins(0, 0, 0, 0)
        content_lay.setSpacing(20)
        content_lay.addLayout(left, stretch=2)
        content_lay.addLayout(right, stretch=1)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(0)
        main_layout.addStretch()
        main_layout.addWidget(content)
        main_layout.addStretch()

    def start_game(self, player_name: str):
        """เริ่มเกมใหม่ของผู้เล่นคนนี้ — เคลียร์ทุกอย่างจากคนก่อน"""
        self._player_name = player_name
        self.player_label.setText(f"Player: {player_name}")
        # สุ่ม 4 คำใหม่ให้ผู้เล่นคนนี้ (Clear ไม่เปลี่ยนคำ — ได้คำใหม่ตอนเริ่มเกมคนใหม่)
        self._target_words = random.sample(WORD_POOL, NUM_TARGET_WORDS)
        self._target_str   = "".join(self._target_words)
        self.target_label.setText("Write:   " + "   ".join(self._target_words))
        self._on_clear()

    def _on_clear(self):
        self.canvas.clear()
        self.models_panel.clear()
        self._snn_energy_j = 0.0
        self._set_status("Draw the letters", STATUS_LIGHT)
        if self._crop_window.isVisible():
            self._crop_window.hide()

    def _on_finish(self):
        if self._worker and self._worker.isRunning():
            return
        self._on_clear()
        self._player_name  = ""
        self._target_words = []
        self._target_str   = ""
        self.player_label.setText("Player: -")
        self.target_label.setText("Write: ")
        self.finished.emit()

    def _on_submit(self):
        if self._worker and self._worker.isRunning():
            return

        pil     = self.canvas.to_pil()
        bgr     = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        tensors = segment_chars(bgr)

        if not tensors:
            self._set_status("No letters found", "#e74c3c")
            return

        self._crop_window.show_crops(tensors)

        if self._snn_net is None:
            self._set_status("Loading model...", STATUS_LIGHT)
            self._snn_net = snn_load(str(SNN_WEIGHTS))
        if self._dnn_net is None:
            self._dnn_net = dnn_load(str(DNN_WEIGHTS))

        self.btn_submit.setEnabled(False)
        self.btn_finish.setEnabled(False)
        self._set_status(f"Processing {len(tensors)} letters...", STATUS_LIGHT)

        self._worker = InferenceWorker(self._snn_net, self._dnn_net, tensors, self._target_str)
        self._worker.snn_done.connect(self._on_snn_done)
        self._worker.dnn_done.connect(self.models_panel.update_dnn)
        self._worker.all_done.connect(self._on_inference_done)
        self._worker.start()

    def _on_snn_done(self, word: str, elapsed_ms: float, energy_j: float, conf: float):
        self.models_panel.update_snn(word, elapsed_ms, energy_j, conf)
        self._snn_energy_j = energy_j  # เก็บไว้ส่งเข้า Hub ตอน inference จบ

    def _on_inference_done(self):
        self.btn_submit.setEnabled(True)
        self.btn_finish.setEnabled(True)

        # ส่ง energy ของ SNN เข้าตารางกลางของ Hub (append ทุกครั้งที่ submit)
        try:
            record_id = send_energy(GAME_NAME, self._player_name, self._snn_energy_j)
        except Exception as exc:
            self._set_status(f"Failed to send energy: {exc}", "#e74c3c")
            return

        energy_pj = self._snn_energy_j * 1e12
        self._set_status(
            f"✓ Sent — {self._player_name} | SNN {energy_pj:,.1f} pJ "
            f"(record #{record_id})",
            "#27ae60",
        )

    def _on_canvas_changed(self):
        tensor      = self.canvas.to_tensor()
        dark_pixels = int((tensor > 0.5).sum())
        if dark_pixels > 50:
            self._set_status(f"Drawn ({dark_pixels} pixels)", STATUS_LIGHT)

    def _set_status(self, text: str, color: str = STATUS_LIGHT):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {color}; background: transparent;"
        )

    def paintEvent(self, event):
        """พื้นหลังหน้าเล่นจริง = gradient แนวตั้งเดียวกับ Hub"""
        p = QPainter(self)
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0.0, QColor(HUB_BG_TOP))
        g.setColorAt(1.0, QColor(HUB_BG_BOTTOM))
        p.fillRect(self.rect(), QBrush(g))
        p.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SNN Drawing Board")
        self.setStyleSheet("background-color: #f0f0f0;")

        self.start_page = StartPage()
        self.game_page  = GamePage()

        self.start_page.started.connect(self._on_start_game)
        self.game_page.finished.connect(self._on_finish_game)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.start_page)  # index 0
        self.stack.addWidget(self.game_page)   # index 1

        self.setCentralWidget(self.stack)
        # ไม่ล็อคขนาดแล้ว — ขยาย/เต็มจอได้ แต่คง 1100×780 เป็นขั้นต่ำ (เท่ากับ Tictac เป๊ะ)
        self.setMinimumSize(WINDOW_W, WINDOW_H)
        self.resize(WINDOW_W, WINDOW_H)

    def _on_start_game(self, player_name: str):
        self.game_page.start_game(player_name)
        self.stack.setCurrentWidget(self.game_page)

    def _on_finish_game(self):
        self.start_page.reset()
        self.stack.setCurrentWidget(self.start_page)

    def keyPressEvent(self, event):
        """F11 สลับเต็มจอ, Esc ออกจากเต็มจอ"""
        if event.key() == Qt.Key_F11:
            self._toggle_fullscreen()
        elif event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
        else:
            super().keyPressEvent(event)

    def _toggle_fullscreen(self):
        self.showNormal() if self.isFullScreen() else self.showFullScreen()


if __name__ == "__main__":
    # ── Uniform scale-to-fit ──
    # ให้ทุกจอ/ทุกเครื่องเห็น layout สัดส่วนเดียวกับ design 1100×780 ที่ออกแบบไว้
    #   1) probe ขนาดจอด้วย QApplication ชั่วคราว
    #   2) คำนวณ scale = min(กว้าง/design, สูง/design) → พอดีจอเสมอ (จอเล็กย่อ, จอใหญ่ขยาย)
    #   3) เปิดแอพจริงด้วย QT_SCALE_FACTOR = scale (Qt สเกลทั้ง UI เท่ากันหมด: canvas/ปุ่ม/ฟอนต์)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)   # กันจอ Windows ที่ scale 125/150%
    _probe = QApplication(sys.argv)
    _geo   = _probe.primaryScreen().availableGeometry()
    _scale = min(_geo.width() / WINDOW_W, _geo.height() / WINDOW_H)
    _probe.quit()
    del _probe

    os.environ["QT_SCALE_FACTOR"] = f"{_scale:.4f}"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showFullScreen()
    sys.exit(app.exec_())
