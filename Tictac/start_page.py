"""
หน้า Sign-In ก่อนเข้าเกม (ดีไซน์ตาม Stunning Sign-In) — แยกออกจาก main.py ให้ไม่รก
ใช้ร่วมกับ MainWindow ผ่าน signal `started(str)` และ method `reset(message)`
"""
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QColor, QBrush, QPen, QLinearGradient, QPainterPath,
)
from PyQt5.QtWidgets import (
    QWidget, QFrame, QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,
)

DEFAULT_SUBTITLE = "ใส่ชื่อผู้เล่นเพื่อเริ่มเกม"

# ---------------------------------------------------------------- SIGN-IN THEME
SIGNIN_BG_TOP    = "#3B1D5E"
SIGNIN_BG_BOTTOM = "#1A0E33"
SIGNIN_TEXT      = "#EDEAFB"
SIGNIN_MUTED     = "#B9B0D0"

HERO_TITLE = "Tic Tac Toe"
HERO_DESC  = ("วาดหมาก X ในช่องที่ต้องการ แล้วสู้กับ AI (SNN / DNN)\n"
              "พร้อมวัดพลังงานที่โมเดลใช้จริง")


class LogoMark(QWidget):
    """โลโก้เล็กมุมซ้ายบน — แท่งขาวโค้งมนสามอัน"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(46, 30)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("white"))
        p.drawRoundedRect(0, 4, 12, 26, 4, 4)
        p.drawRoundedRect(17, 0, 12, 30, 4, 4)
        p.drawRoundedRect(34, 8, 12, 22, 4, 4)
        p.end()


class StartPage(QWidget):
    """หน้า Sign-In ก่อนเข้าเกม — hero ซ้าย + การ์ด Sign in ขวา"""

    started = pyqtSignal(str)   # ส่งชื่อผู้เล่นออกไป

    def __init__(self):
        super().__init__()
        self._build_ui()

    # ── UI ──
    def _build_ui(self):
        logo = LogoMark()

        heading = QLabel(HERO_TITLE)
        heading.setStyleSheet("font-size: 54px; font-weight: bold; color: white; background: transparent;")

        underline = QFrame()
        underline.setFixedSize(64, 4)
        underline.setStyleSheet("background: rgba(255,255,255,0.55); border-radius: 2px;")

        desc = QLabel(HERO_DESC)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size: 13px; color: {SIGNIN_MUTED}; background: transparent;")

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        left.addWidget(logo)
        left.addStretch(2)
        left.addWidget(heading)
        left.addSpacing(16)
        left.addWidget(underline)
        left.addSpacing(22)
        left.addWidget(desc)
        left.addStretch(3)

        left_host = QWidget()
        # โปร่งใส ไม่งั้น WINDOW_BG (#f0f0f0) ของ MainWindow จะแผ่ทับ gradient
        left_host.setStyleSheet("background: transparent;")
        left_host.setLayout(left)

        card = self._build_card()
        card_host = QVBoxLayout()
        card_host.addStretch()
        card_host.addWidget(card)
        card_host.addStretch()

        root = QHBoxLayout(self)
        root.setContentsMargins(70, 60, 70, 60)
        root.setSpacing(40)
        root.addWidget(left_host, stretch=1)
        root.addLayout(card_host)

    def _build_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        card.setFixedWidth(360)
        card.setStyleSheet("""
            QFrame#card {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 16px;
            }
            QLabel { background: transparent; color: #EDEAFB; }
        """)

        signin_title = QLabel("Sign in")
        signin_title.setStyleSheet("font-size: 26px; font-weight: bold; color: white;")
        title_underline = QFrame()
        title_underline.setFixedSize(44, 3)
        title_underline.setStyleSheet("background: rgba(255,255,255,0.7); border-radius: 2px;")

        tcol = QVBoxLayout()
        tcol.setSpacing(6)
        tcol.addWidget(signin_title)
        tcol.addWidget(title_underline)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addStretch()
        title_row.addLayout(tcol)
        title_row.addStretch()

        user_label = QLabel("User Name")
        user_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #EDEAFB;")

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("ชื่อผู้เล่น")
        self.name_input.setMaxLength(32)
        self.name_input.setFixedHeight(44)
        self.name_input.setStyleSheet("""
            QLineEdit {
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.18);
                border-radius: 8px;
                padding: 0 14px;
                font-size: 15px;
                color: white;
            }
            QLineEdit:focus { border: 1px solid #FBB040; }
        """)
        self.name_input.textChanged.connect(self._on_text_changed)
        self.name_input.returnPressed.connect(self._on_start)

        self.btn_start = QPushButton("Start Game")
        self.btn_start.setFixedHeight(46)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setEnabled(False)        # ชื่อว่างเริ่มไม่ได้
        self.btn_start.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 #FBB040, stop:1 #F0533F);
                color: white;
                font-size: 15px;
                font-weight: bold;
                border: none;
                border-radius: 23px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 #FFC15A, stop:1 #FF6552);
            }
            QPushButton:disabled {
                background: rgba(255,255,255,0.14);
                color: rgba(255,255,255,0.40);
            }
        """)
        self.btn_start.clicked.connect(self._on_start)

        self.msg_label = QLabel(DEFAULT_SUBTITLE)
        self.msg_label.setWordWrap(True)
        self.msg_label.setAlignment(Qt.AlignCenter)
        self.msg_label.setStyleSheet(f"font-size: 12px; color: {SIGNIN_MUTED};")

        lay = QVBoxLayout(card)
        lay.setContentsMargins(34, 34, 34, 34)
        lay.setSpacing(10)
        lay.addLayout(title_row)
        lay.addSpacing(14)
        lay.addWidget(user_label)
        lay.addWidget(self.name_input)
        lay.addSpacing(18)
        lay.addWidget(self.btn_start)
        lay.addSpacing(6)
        lay.addWidget(self.msg_label)
        return card

    # ── background ──
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        g = QLinearGradient(0, 0, w, h)
        g.setColorAt(0.0, QColor(SIGNIN_BG_TOP))
        g.setColorAt(1.0, QColor(SIGNIN_BG_BOTTOM))
        p.fillRect(self.rect(), QBrush(g))

        pen = QPen(QColor(255, 255, 255, 20))
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(w * 0.04, h * 0.92)
        path.cubicTo(w * 0.32, h * 0.72, w * 0.18, h * 0.28, w * 0.52, h * 0.10)
        p.drawPath(path)
        path2 = QPainterPath()
        path2.moveTo(w * 0.62, h * 1.02)
        path2.cubicTo(w * 0.78, h * 0.62, w * 0.92, h * 0.52, w * 1.06, h * 0.18)
        p.drawPath(path2)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 12))
        p.drawEllipse(QPointF(w * 0.86, h * 0.24), 130, 130)
        p.end()

    # ── logic ──
    def _on_text_changed(self, text: str):
        self.btn_start.setEnabled(bool(text.strip()))

    def _on_start(self):
        name = self.name_input.text().strip()
        if name:
            self.started.emit(name)

    def reset(self, message: str = DEFAULT_SUBTITLE):
        """กลับมาหน้านี้ — ล้างชื่อคนก่อน แล้วโชว์ข้อความ (เช่น สรุปเกมที่เพิ่งล้มกระดาน)"""
        self.msg_label.setText(message)
        self.name_input.clear()
        self.name_input.setFocus()
