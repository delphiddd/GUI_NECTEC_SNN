import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget, QFrame,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QPainter, QColor, QBrush, QLinearGradient,
)

from canvas import Canvas, GRID
from img_segment import segment_grid, detect_filled
from inference_dnn_detect import classify_grid
from snn_inference import ai_choose_move, set_difficulty
from dnn_inference import ai_choose_move as dnn_choose_move
from checkwin import TicTacToe, EMPTY
from energy_window import EnergyPanel
from ui_styles import (
    make_button, WINDOW_BG, STATUS_STYLE, CANVAS_FRAME_STYLE,
)
from start_page import StartPage, DEFAULT_SUBTITLE

# แอพรันแบบ script จากโฟลเดอร์ตัวเอง → ต้องเติม project root เพื่อ import shared/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.energy_client import send_energy  # noqa: E402

GAME_NAME = "tictac"

# ชื่อฝ่ายตรงข้ามที่โชว์บน status ตามโมเดลที่เลือก
AI_NAMES = {"EASY": "SNN ง่าย", "HARD": "SNN ยาก", "DNN": "DNN"}

# โมเดลที่นับพลังงานส่งเข้า Hub — DNN ไม่ส่ง (ส่งเฉพาะ SNN ตามที่ตกลงไว้)
SNN_LEVELS = ("EASY", "HARD")

# ขนาดหน้าต่างล็อคตายตัว — ต้องเท่ากับ Alphabet_read เป๊ะ ๆ (ห้ามแก้ข้างเดียว)
WINDOW_W, WINDOW_H = 1100, 780


# ---------------------------------------------------------------- SELECT SCREEN
# พื้นหลังเดียวกับ Hub (hub/gui/leaderboard.py)
HUB_BG_TOP, HUB_BG_BOTTOM = "#3A3168", "#1E1943"
CARD_PURPLE_A, CARD_PURPLE_B = "#C15BDB", "#9B37C4"   # ม่วงเรืองแสงตอน hover (โทนเรฟ)
CARD_TEXT_DARK = "#2c2246"

# ── ขีดระดับความยาก (ใช้แทน emoji ในเฟสเลือกโมเดล) ──
BAR_ON, BAR_OFF = "#6c3483", "#D9CFE8"   # แท่งติด / แท่งดับ ตอนการ์ดยังไม่ hover
BAR_DNN = "#2980b9"                      # DNN คนละตระกูลกับ SNN → ฟ้า
BAR_W, BAR_GAP = 18, 9                   # ความกว้างแท่ง / ระยะห่าง
BAR_H_MIN, BAR_H_MAX = 18, 46            # ความสูงแท่งแรก → แท่งสุดท้าย
BARS_H, BADGE_H = 46, 22                 # โซนขีด / โซน badge (จองไว้เสมอให้ทุกการ์ดขีดตรงแนวกัน)


class GradientPage(QWidget):
    """QWidget พื้นหลัง gradient แนวตั้งเดียวกับ Hub — ใช้เป็นฐานหน้าเล่นจริง"""

    def paintEvent(self, event):
        p = QPainter(self)
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0.0, QColor(HUB_BG_TOP))
        g.setColorAt(1.0, QColor(HUB_BG_BOTTOM))
        p.fillRect(self.rect(), QBrush(g))
        p.end()


class DifficultyBars(QWidget):
    """ขีดระดับความยากแบบแท่งไล่ความสูง — level แท่งแรกติด ที่เหลือจาง

    badge: ข้อความ chip เล็ก ๆ ใต้ขีด (ใช้กำกับ DNN ว่าเป็นคนละตระกูล) — None = ไม่วาด
    """

    def __init__(self, level: int, total: int = 3, accent: str = BAR_ON,
                 badge: str = None, parent=None):
        super().__init__(parent)
        self._level = level
        self._total = total
        self._accent = accent
        self._badge = badge
        self._hover = False
        self.setFixedSize(total * BAR_W + (total - 1) * BAR_GAP, BARS_H + BADGE_H)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)   # ให้คลิกทะลุไปที่การ์ด

    def set_hover(self, hover: bool) -> None:
        """การ์ดแม่ hover อยู่ไหม — เปลี่ยนสีขีดให้อ่านออกบนพื้นม่วง"""
        self._hover = hover
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)

        on_col  = QColor("white") if self._hover else QColor(self._accent)
        off_col = QColor(255, 255, 255, 70) if self._hover else QColor(BAR_OFF)

        step = (BAR_H_MAX - BAR_H_MIN) / max(self._total - 1, 1)
        for i in range(self._total):
            h = BAR_H_MIN + step * i
            x = i * (BAR_W + BAR_GAP)
            p.setBrush(QBrush(on_col if i < self._level else off_col))
            p.drawRoundedRect(QRectF(x, BARS_H - h, BAR_W, h), 4, 4)

        if self._badge:
            f = p.font()
            f.setPointSize(8)
            f.setBold(True)
            p.setFont(f)
            cw = p.fontMetrics().horizontalAdvance(self._badge) + 18
            chip = QRectF((self.width() - cw) / 2, BARS_H + 5, cw, BADGE_H - 6)
            p.setBrush(QBrush(QColor(255, 255, 255, 60) if self._hover else QColor(self._accent)))
            p.drawRoundedRect(chip, chip.height() / 2, chip.height() / 2)
            p.setPen(QColor("white"))
            p.drawText(chip, Qt.AlignCenter, self._badge)
        p.end()


class Card(QFrame):
    """การ์ดตัวเลือก — คลิกแล้วเรียก on_pick(key) ทันที; hover = ม่วงเรืองแสง

    icon: str = emoji (เฟสเลือกโหมด/ฝั่ง) | dict = kwargs ของ DifficultyBars (เฟสเลือกโมเดล)
    """

    def __init__(self, key: str, label: str, icon, on_pick, parent=None):
        super().__init__(parent)
        self._key = key
        self._on_pick = on_pick
        self.setFixedSize(172, 158)
        self.setCursor(Qt.PointingHandCursor)

        if isinstance(icon, dict):
            self.icon_lbl = None
            self.bars = DifficultyBars(**icon)
            icon_widget = self.bars
        else:
            self.bars = None
            self.icon_lbl = QLabel(icon)
            self.icon_lbl.setAlignment(Qt.AlignCenter)
            icon_widget = self.icon_lbl

        self.text_lbl = QLabel(label)
        self.text_lbl.setAlignment(Qt.AlignCenter)
        self.text_lbl.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 18, 12, 16)
        lay.setSpacing(8)
        lay.addStretch()
        lay.addWidget(icon_widget, alignment=Qt.AlignHCenter)
        lay.addWidget(self.text_lbl)
        lay.addStretch()

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(26)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 90))
        self.setGraphicsEffect(shadow)

        self._apply(False)

    def _apply(self, hover: bool):
        if hover:
            self.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0, y1:0, x2:0.8, y2:1,
                                stop:0 {CARD_PURPLE_A}, stop:1 {CARD_PURPLE_B});
                    border-radius: 22px;
                    border: none;
                }}
            """)
            if self.icon_lbl is not None:
                self.icon_lbl.setStyleSheet("background: transparent; font-size: 46px; color: white;")
            self.text_lbl.setStyleSheet("background: transparent; color: white; font-size: 15px; font-weight: bold;")
        else:
            self.setStyleSheet("""
                QFrame {
                    background: white;
                    border-radius: 22px;
                    border: none;
                }
            """)
            if self.icon_lbl is not None:
                self.icon_lbl.setStyleSheet(f"background: transparent; font-size: 46px; color: {BAR_ON};")
            self.text_lbl.setStyleSheet(f"background: transparent; color: {CARD_TEXT_DARK}; font-size: 15px; font-weight: bold;")
        if self.bars is not None:
            self.bars.set_hover(hover)

    def enterEvent(self, event):
        self._apply(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._on_pick(self._key)
        super().mousePressEvent(event)


class SelectScreen(QWidget):
    """หน้าเลือก (โหมด/โมเดล/ฝั่ง) แบบการ์ด บนพื้น gradient เดียวกับ Hub"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._back_cb = None

        self.title_lbl = QLabel()
        self.title_lbl.setAlignment(Qt.AlignCenter)
        self.title_lbl.setStyleSheet("background: transparent; color: white; font-size: 30px; font-weight: bold;")

        self.underline = QFrame()
        self.underline.setFixedSize(72, 5)
        self.underline.setStyleSheet(f"background: {CARD_PURPLE_A}; border-radius: 2px;")

        self.subtitle_lbl = QLabel()
        self.subtitle_lbl.setAlignment(Qt.AlignCenter)
        self.subtitle_lbl.setStyleSheet("background: transparent; color: #B9B4D6; font-size: 13px;")

        self._cards_row = QHBoxLayout()
        self._cards_row.setSpacing(26)
        cards_host = QWidget()
        cards_host.setStyleSheet("background: transparent;")   # กัน WINDOW_BG แผ่เทาทับ gradient
        cards_host.setLayout(self._cards_row)

        self.btn_back = QPushButton("←  Back")
        self.btn_back.setFixedHeight(40)
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.12);
                color: #EDEAFB;
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 20px;
                font-size: 13px;
                font-weight: bold;
                padding: 0 30px;
            }
            QPushButton:hover  { background: rgba(255,255,255,0.20); }
            QPushButton:pressed{ background: rgba(255,255,255,0.30); }
        """)
        self.btn_back.clicked.connect(self._on_back)

        ul_row = QHBoxLayout()
        ul_row.addStretch(); ul_row.addWidget(self.underline); ul_row.addStretch()
        cards_center = QHBoxLayout()
        cards_center.addStretch(); cards_center.addWidget(cards_host); cards_center.addStretch()
        back_row = QHBoxLayout()
        back_row.addStretch(); back_row.addWidget(self.btn_back); back_row.addStretch()

        root = QVBoxLayout(self)
        root.setContentsMargins(60, 50, 60, 50)
        root.addStretch(2)
        root.addWidget(self.title_lbl)
        root.addSpacing(10)
        root.addLayout(ul_row)
        root.addSpacing(8)
        root.addWidget(self.subtitle_lbl)
        root.addSpacing(38)
        root.addLayout(cards_center)
        root.addSpacing(40)
        root.addLayout(back_row)
        root.addStretch(3)

    def configure(self, title: str, options: list, on_pick, on_back, subtitle: str = ""):
        """ตั้งค่าเฟส: หัวข้อ + การ์ด + callback (subtitle ไม่ใส่ = ซ่อน)"""
        self.title_lbl.setText(title)
        self.subtitle_lbl.setText(subtitle)
        self.subtitle_lbl.setVisible(bool(subtitle))
        self._back_cb = on_back
        # ล้างการ์ดเดิม แล้วสร้างใหม่
        while self._cards_row.count():
            item = self._cards_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        for key, label, icon in options:
            self._cards_row.addWidget(Card(key, label, icon, on_pick))

    def _on_back(self):
        if self._back_cb is not None:
            self._back_cb()

    def paintEvent(self, event):
        p = QPainter(self)
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0.0, QColor(HUB_BG_TOP))
        g.setColorAt(1.0, QColor(HUB_BG_BOTTOM))
        p.fillRect(self.rect(), QBrush(g))
        p.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-O Drawing Game")
        self.setStyleSheet(WINDOW_BG)

        # ── Game state ──
        self.game = TicTacToe()         # logic เกม (X ลงก่อนเสมอ)
        self.player_name = ""           # ชื่อ User1 — กรอกในเฟสแรกสุด, ใช้ตอนส่งเข้า Hub
        self.mode = None                # "HUMAN" (คนกับคน) หรือ "AI" (คนกับโมเดล)
        self.difficulty = None          # "EASY"/"HARD" (SNN) หรือ "DNN" — เฉพาะโหมด AI (เลือกโมเดล .pt)
        self.user1_sym = None           # ฝั่งที่ User1 เลือก ("X"/"O")
        self.ai_symbol = None           # ฝั่งที่ AI เล่น (ตรงข้าม user1_sym, เฉพาะโหมด AI)
        self.in_game = False            # อยู่ระหว่างเล่นมั้ย
        self._auto = False              # preview ตรวจจับสดตอนวาด
        self._detect_win = None
        self._committed_img = None      # checkpoint ลายมือถึงตาที่ commit แล้ว
        self._snn_energy_j = 0.0        # พลังงาน SNN สะสมทั้งเกม (จูล) — ส่งเข้า Hub ตอนเกมจบ
        self._sent = False              # กันส่งซ้ำ: 1 เกมจบ = 1 record

        self.canvas = Canvas()
        self.canvas.changed.connect(self._on_canvas_changed)

        # (ปุ่มเลือกโหมด/โมเดล/ฝั่ง เดิม ย้ายไปเป็นการ์ดใน SelectScreen แล้ว)

        # ── ปุ่มระหว่างเล่น ──
        self.btn_confirm = make_button("ยืนยันตา", "confirm", self._on_confirm)
        self.btn_confirm.setVisible(False)

        self.btn_undo = make_button("แก้ตานี้", "undo", self._on_undo)
        self.btn_undo.setVisible(False)

        self.btn_break = make_button("Break (ล้มกระดาน)", "break", self._on_break)
        self.btn_break.setVisible(False)

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(STATUS_STYLE)

        self.canvas_frame = QWidget()
        self.canvas_frame.setStyleSheet(CANVAS_FRAME_STYLE)
        QVBoxLayout(self.canvas_frame).addWidget(self.canvas)
        self.canvas_frame.setVisible(False)   # ซ่อนกระดานจนกว่าจะเลือกโหมด/ฝั่งครบ

        # แผงกราฟพลังงาน SNN เรียลไทม์ (โชว์เฉพาะโหมด AI, อัปเดตทุกตาที่ AI เดิน)
        self.energy_panel = EnergyPanel()
        self.energy_panel.setVisible(False)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        for b in self._all_buttons():
            btn_row.addWidget(b)
        btn_row.addStretch()

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        # แถวกลาง: canvas ซ้าย + กราฟพลังงานผอมๆ ขวา
        content_row = QHBoxLayout()
        content_row.setSpacing(16)
        content_row.addStretch()
        content_row.addWidget(self.canvas_frame)
        content_row.addWidget(self.energy_panel)
        content_row.addStretch()

        layout.addWidget(self.status_label)
        layout.addLayout(content_row, stretch=1)   # กินที่ว่างที่เหลือ → ปุ่มติดล่างเสมอทุกเฟส
        layout.addLayout(btn_row)

        # ── หน้าเล่นจริง — พื้น gradient เดียวกับ Hub ──
        self.game_page = GradientPage()
        self.game_page.setLayout(layout)

        self.start_page = StartPage()
        self.start_page.started.connect(self._on_started)

        # หน้าเลือกโหมด/โมเดล/ฝั่ง แบบการ์ด (คั่นระหว่าง sign-in กับหน้าเล่นจริง)
        self.select_screen = SelectScreen()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.start_page)     # index 0
        self.stack.addWidget(self.select_screen)  # index 1
        self.stack.addWidget(self.game_page)      # index 2
        self.setCentralWidget(self.stack)

        # ไม่ล็อคขนาดแล้ว — ขยาย/เต็มจอได้ แต่คง 1100×780 เป็นขั้นต่ำ (เท่ากับ Alphabet_read เป๊ะ)
        self.setMinimumSize(WINDOW_W, WINDOW_H)
        self.resize(WINDOW_W, WINDOW_H)

        self._show_name_phase()         # เริ่มที่หน้าจอใส่ชื่อ

    def _all_buttons(self):
        # เหลือเฉพาะปุ่มระหว่างเล่น — เลือกโหมด/โมเดล/ฝั่ง ใช้การ์ดใน SelectScreen แทน
        return (self.btn_confirm, self.btn_undo, self.btn_break)

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

    # ── ชื่อฝ่ายตรงข้าม User1 (โมเดลที่เลือก หรือ User2) ──
    def _opponent_name(self) -> str:
        if self.mode == "AI":
            return AI_NAMES[self.difficulty]
        return "User2"

    # ── ใครเป็นเจ้าของ symbol นี้ ──
    def _whose(self, sym: str) -> str:
        if sym == self.user1_sym:
            return "User1"
        return self._opponent_name()

    def _prompt_turn(self):
        cur = self.game.current
        self.status_label.setText(
            f"ตาของ {self._whose(cur)} ({cur}) — วาด {cur} ลงช่องว่าง (ให้อยู่กลางช่อง) แล้วกด 'ยืนยันตา'"
        )

    # ── โชว์เฉพาะปุ่มที่ส่งมา ซ่อนที่เหลือทั้งหมด (กันปุ่มค้างข้ามเฟส) ──
    def _only_show(self, *buttons):
        for b in self._all_buttons():
            b.setVisible(b in buttons)

    # ── ซ่อน/โชว์กระดาน (ขนาดหน้าต่างล็อคไว้ ไม่ขยับตามอีกแล้ว) ──
    def _show_board(self, show: bool):
        self.canvas_frame.setVisible(show)

    # ── เฟส 1: ใส่ชื่อผู้เล่น (แรกสุด) — กลับไปหน้า StartPage ──
    def _show_name_phase(self, message: str = DEFAULT_SUBTITLE):
        self.in_game = False
        self._auto = False
        self.mode = None
        self.difficulty = None
        self.ai_symbol = None
        self.player_name = ""
        self.energy_panel.setVisible(False)
        self._show_board(False)
        self.start_page.reset(message)
        self.stack.setCurrentWidget(self.start_page)

    def _on_started(self, name: str):
        """StartPage ส่งชื่อมา (ปุ่มจะกดไม่ได้ถ้าชื่อว่าง จึงไม่ต้องเช็คซ้ำ)"""
        self.player_name = name
        self._show_mode_phase()

    # ── เฟส 2: เลือกโหมด (การ์ด) ──
    def _show_mode_phase(self):
        self.in_game = False
        self._auto = False
        self.mode = None
        self.difficulty = None
        self.ai_symbol = None
        self.energy_panel.setVisible(False)
        self._show_board(False)
        self.select_screen.configure(
            title="SELECT MODE",
            options=[
                ("HUMAN", "เล่นกับคน", "👥"),
                ("AI",    "เล่นกับ AI", "🤖"),
            ],
            on_pick=self._on_mode,
            on_back=self._show_name_phase,   # กลับหน้า sign-in
        )
        self.stack.setCurrentWidget(self.select_screen)

    def _on_mode(self, mode: str):
        self.mode = mode
        if mode == "AI":
            self._show_difficulty_phase()   # เล่นกับ AI → เลือกความยากก่อน
        else:
            self._show_select_phase()       # เล่นกับคน → ข้ามไปเลือกฝั่งเลย

    # ── เฟส 3: เลือกโมเดล AI (การ์ด, เฉพาะโหมด AI) ──
    def _show_difficulty_phase(self):
        self.in_game = False
        self._auto = False
        self._show_board(False)
        self.select_screen.configure(
            title="SELECT AI MODEL",
            options=[
                ("EASY", "SNN ง่าย", {"level": 1}),
                ("HARD", "SNN ยาก", {"level": 3}),
                ("DNN",  "DNN",      {"level": 3}),
            ],
            on_pick=self._on_difficulty,
            on_back=self._show_mode_phase,
        )
        self.stack.setCurrentWidget(self.select_screen)

    def _on_difficulty(self, level: str):
        self.difficulty = level
        if level != "DNN":
            set_difficulty(level)           # สลับ weight ที่ snn_inference จะโหลด/ใช้
        self._show_select_phase()

    # ── เฟส 4: เลือกฝั่ง (การ์ด) ──
    def _show_select_phase(self):
        self.in_game = False
        self._auto = False
        self._show_board(False)
        # เล่นกับคน → Back กลับไปเลือกโหมด; เล่นกับ AI → Back กลับไปเลือกโมเดล
        back = self._show_difficulty_phase if self.mode == "AI" else self._show_mode_phase
        self.select_screen.configure(
            title="SELECT SIDE",
            options=[
                ("X", "User1 เป็น X", "✕"),
                ("O", "User1 เป็น O", "◯"),
            ],
            on_pick=self._on_select,
            on_back=back,
        )
        self.stack.setCurrentWidget(self.select_screen)

    def _on_select(self, sym: str):
        self.stack.setCurrentWidget(self.game_page)   # จากการ์ด → เข้าหน้าเล่นจริง
        self.user1_sym = sym
        self.ai_symbol = ("O" if sym == "X" else "X") if self.mode == "AI" else None
        self.game.reset()                       # X ลงก่อนเสมอ
        self._snn_energy_j = 0.0                # เกมใหม่ = นับพลังงานใหม่
        self._sent = False
        # เตรียมแผงพลังงาน: โหมด AI โชว์+ล้าง (ก่อน AI เดินตาแรก), โหมดคนซ่อน
        self.energy_panel.reset()
        if self.mode == "AI":
            self.energy_panel.set_model_name(self._opponent_name())
        self.energy_panel.setVisible(self.mode == "AI")
        self.in_game = True
        self._auto = True
        self.canvas.clear()
        self._committed_img = self.canvas.snapshot()

        self._only_show(self.btn_confirm, self.btn_undo, self.btn_break)
        self._show_board(True)                  # เข้าหน้าเกม → กระดานโผล่ตอนนี้
        self.btn_confirm.setEnabled(True)
        self.btn_undo.setEnabled(True)

        # ไม่โชว์หน้าต่าง "ผลตรวจจับ 9 ช่อง" ให้ user เห็นแล้ว — classify ยังทำงานปกติใน _detect()
        # (_detect_win คงเป็น None ตลอด, update_cells มี guard skip เอง)

        opp_sym = "O" if sym == "X" else "X"
        self.status_label.setText(f"เริ่มเกม! User1 = {sym}, {self._opponent_name()} = {opp_sym}")

        # โหมด AI + AI ได้เป็น X → AI ต้องเปิดเกมก่อน (X ลงก่อนเสมอ)
        if self.mode == "AI" and self.ai_symbol == "X":
            self._ai_move()
        else:
            self._prompt_turn()

    # ── ตรวจจับ canvas → (tensors, filled, results) + อัปเดต preview (ไม่แตะ status) ──
    def _detect(self):
        pil = self.canvas.to_pil()
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        tensors = segment_grid(bgr)               # cell_transform รอบเดียวต่อ cell
        filled = detect_filled(tensors)           # ช่องไหนมีการเขียน
        results = classify_grid(tensors, filled)  # classify X/O เฉพาะช่องที่เขียน
        if self._detect_win is not None:
            self._detect_win.update_cells(tensors, filled, results)
        return tensors, filled, results

    # ── ยืนยันตา: ตรวจ + commit ลง logic + สลับตา/หาผู้ชนะ ──
    def _on_confirm(self):
        if not self.in_game or self.game.over:
            return

        _, filled, results = self._detect()
        cur = self.game.current

        # ช่องที่ "เพิ่งวาดตานี้" = filled แต่ logic ยังว่าง (ที่ commit แล้วจะ filled อยู่ด้วย)
        new_cells = [i for i in range(GRID * GRID)
                     if filled[i] and self.game.board[i] == EMPTY]

        if len(new_cells) == 0:
            self.status_label.setText(
                f"ยังไม่ได้วาด (หรือวาดทับช่องที่มีคนลงแล้ว) — วาด {cur} ลงช่องว่าง แล้วกด 'ยืนยันตา'"
            )
            return
        if len(new_cells) > 1:
            self.status_label.setText(
                "วาดได้ทีละช่อง — วาดให้อยู่กลางช่อง อย่าล้ำเส้น แล้วกด 'แก้ตานี้' เพื่อวาดใหม่"
            )
            return

        idx = new_cells[0]
        label = results[idx]["label"]
        if label != cur:
            self.status_label.setText(
                f"ตานี้เป็นของ {self._whose(cur)} ({cur}) ห้ามวาด {label} — กด 'แก้ตานี้' แล้ววาด {cur} ใหม่"
            )
            return

        # ✅ ถูกต้อง → commit ลง logic
        r = self.game.play(idx)
        self._committed_img = self.canvas.snapshot()   # ล็อกลายมือถึงตานี้

        if self._handle_result(r):
            return
        # ยังไม่จบ → โหมด AI ให้ AI เดินต่อทันที, โหมดคนรอ User อีกคน
        if self.mode == "AI":
            self._ai_move()
        else:
            self._prompt_turn()

    # ── ผลหลัง commit 1 ตา: ชนะ/เสมอ → จบเกม. คืน True ถ้าเกมจบแล้ว ──
    def _handle_result(self, r: dict) -> bool:
        if r["winner"]:
            head = f"🎉 {self._whose(r['winner'])} ({r['winner']}) ชนะ!"
        elif r["draw"]:
            head = "🤝 เสมอ! กระดานเต็มแล้ว"
        else:
            return False

        self._lock_board()
        # รู้ผลแพ้/ชนะ/เสมอแล้วเท่านั้น ถึงบันทึกลง Hub
        parts = [head, self._submit_energy(), "— กด 'Break' เพื่อเริ่มใหม่"]
        self.status_label.setText(" ".join(p for p in parts if p))
        return True

    # ── ส่งพลังงาน SNN ทั้งเกมเข้า Hub → คืนข้อความสรุปไปต่อท้าย status ──
    def _submit_energy(self) -> str:
        """ส่งเฉพาะโหมด AI + โมเดล SNN + ยังไม่เคยส่งเกมนี้ (1 เกมจบ = 1 record)"""
        if self._sent:
            return ""
        if self.mode != "AI" or self.difficulty not in SNN_LEVELS:
            return ""

        self._sent = True                       # ตั้งก่อนส่ง กันกดรัวจนส่งซ้ำ
        try:
            # ส่งเป็นจูลดิบ — energy_client เป็นคนแปลงเป็น pJ ที่เดียว
            record_id = send_energy(GAME_NAME, self.player_name, self._snn_energy_j)
        except Exception as exc:                # ไม่กลืน error เงียบ ๆ — โชว์ให้ผู้เล่นเห็น
            self._sent = False                  # ส่งไม่ผ่าน → ให้เกมหน้ามีสิทธิ์ลองใหม่
            return f"| ⚠️ ส่ง energy ไม่สำเร็จ: {exc}"

        energy_uj = self._snn_energy_j * 1e6    # จูล → µJ (หน่วยเดียวกับ leaderboard ของ Hub)
        return (
            f"| ✓ บันทึกแล้ว: {self.player_name} "
            f"| SNN {energy_uj:,.3f} µJ (record #{record_id})"
        )

    # ── ตา AI (SNN หรือ DNN): เลือกช่อง → วาดลง canvas → commit logic ──
    def _ai_move(self):
        if not self.in_game or self.game.over:
            return
        # ส่ง difficulty ตรงๆ ทุกตา → เลือกโมเดล/weight จาก self.difficulty เป็นแหล่งความจริงเดียว
        if self.difficulty == "DNN":
            idx, energy_j = dnn_choose_move(self.game.board, self.ai_symbol)
        else:
            idx, energy_j = ai_choose_move(self.game.board, self.ai_symbol, self.difficulty)
            self._snn_energy_j += energy_j                      # สะสมเฉพาะ SNN (DNN ไม่ส่งเข้า Hub)
        self.energy_panel.add_move(energy_j)                    # อัปเดตกราฟพลังงานเรียลไทม์ทันที
        self.canvas.draw_symbol(idx, self.ai_symbol)            # วาด (ไม่ emit changed)
        r = self.game.play(idx)                                 # commit ตรง (authoritative)
        self._committed_img = self.canvas.snapshot()            # ล็อกตา AI ไว้ (กัน undo ลบ)
        self._detect()                                          # อัปเดต preview ให้เห็นตา AI
        if self._handle_result(r):
            return
        self._prompt_turn()

    def _lock_board(self):
        """จบเกม (ชนะ/เสมอ) — ปิดยืนยัน/แก้ตา เหลือแค่ Break"""
        self.game.over = True
        self._auto = False
        self.btn_confirm.setEnabled(False)
        self.btn_undo.setEnabled(False)

    # ── แก้ตานี้: คืน canvas กลับ checkpoint ล่าสุด (ไม่แตะตาที่ commit แล้ว) ──
    def _on_undo(self):
        if not self.in_game or self.game.over:
            return
        self.canvas.restore(self._committed_img)
        self._detect()                  # refresh preview ให้ตรงกับที่ commit ไว้
        self._prompt_turn()

    # ── Break: จบเกม + จบ user คนนี้ → กลับไปหน้าใส่ชื่อ (ผู้เล่นคนใหม่) ──
    # หมายเหตุ: กด Break กลางเกม (ยังไม่รู้แพ้ชนะ) = ไม่บันทึกอะไรลง Hub
    def _on_break(self):
        self.game.reset()
        self.canvas.clear()
        self._committed_img = self.canvas.snapshot()
        self.btn_confirm.setEnabled(True)
        self.btn_undo.setEnabled(True)
        self.energy_panel.reset()
        self._snn_energy_j = 0.0
        self._sent = False
        self._close_detect_win()
        # กลับหน้าใส่ชื่อ (ล้างชื่อผู้เล่นให้ด้วย) — ข้อความไปโชว์แทน subtitle ของหน้าแรก
        self._show_name_phase("ล้มกระดาน จบเกม — ใส่ชื่อผู้เล่นคนใหม่เพื่อเริ่มรอบต่อไป")

    def _close_detect_win(self):
        if self._detect_win is not None:
            self._detect_win.close()
            self._detect_win = None

    # auto-preview: ทุกครั้งที่วาดเสร็จ (ปล่อยเมาส์) อัปเดตหน้าต่างตรวจจับสด (ยังไม่ commit)
    def _on_canvas_changed(self):
        if self._auto and self.in_game and not self.game.over:
            self._detect()


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
