"""
หน้า Leaderboard ของ Hub — สไตล์การ์ด gradient (มงกุฎ + avatar แมว + rank badge + pill)
วาง 2 เกมเป็นคอลัมน์ข้างกัน:  ซ้าย = Tic-Tac-Toe | ขวา = Alphabet

โชว์ TOP 5 คนที่ใช้ energy น้อยที่สุด (ยิ่งน้อย = ยิ่งดี → อันดับ 1 ได้มงกุฎ)
ถ้ายังมีไม่ครบ 5 ช่องที่เหลือขึ้น UNKNOWN / "-" ไว้ก่อน

ห้ามเขียน SQL ที่นี่ — เรียกผ่าน hub.db เท่านั้น
"""
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QPainter, QColor, QBrush, QPolygonF, QFont, QPainterPath,
    QLinearGradient, QFontDatabase,
)
from PyQt5.QtWidgets import (
    QWidget, QLabel, QFrame, QPushButton,
    QVBoxLayout, QHBoxLayout, QSizePolicy,
)

from hub import db

# ---------------------------------------------------------------- THEME
BG_TOP, BG_BOTTOM = "#3A3168", "#1E1943"     # gradient พื้นหลังแนวตั้ง
GOLD_A, GOLD_B = "#EFDC72", "#F0F4CC"        # อันดับ 1
MINT_A, MINT_B = "#3EDCC4", "#A9F1D0"        # อันดับ 2–5
TEXT_DARK = "#101119"
TEXT_LIGHT = "#EDEAFB"
TEXT_MUTED = "#B9B4D6"

ROW_H   = 60     # ความสูงแต่ละแถว (badge / avatar / pill)
GAP     = 14     # ระยะห่างแนวตั้งระหว่างแถว
SPACING = 16     # ระยะห่างแนวนอน badge / avatar / pill
CROWN_W, CROWN_H = 42, 32

# สีประจำเกม (ใช้กับหัวคอลัมน์)
GAME_COLORS = {
    "tictac":   "#2980b9",   # ฟ้า
    "alphabet": "#8e44ad",   # ม่วง
}

# ชื่อเกมใน db.GAMES → หัวข้อที่โชว์ (เรียงซ้าย → ขวา)
GAMES: list[tuple[str, str]] = [
    ("tictac", "🎮 Tic-Tac-Toe"),
    ("alphabet", "🔤 Alphabet"),
]

# DB เก็บเป็น pJ (CLAUDE.md §5) — แปลงหน่วยตอนโชว์เท่านั้น ไม่แตะค่าใน DB
# แต่ละเกมสเกลต่างกัน: tictac ระดับ nJ, alphabet ระดับ µJ
GAME_UNITS: dict[str, tuple[str, float]] = {
    "tictac":   ("nJ", 1e3),    # 1 nJ = 1e3 pJ
    "alphabet": ("µJ", 1e6),    # 1 µJ = 1e6 pJ
}

PLACEHOLDER_NAME = "UNKNOWN"
PLACEHOLDER_SCORE = "-"


def _fmt_energy(energy_pj: float, pj_per_unit: float) -> str:
    """pJ → หน่วยที่เกมนั้นใช้ สำหรับโชว์บน pill"""
    return f"{energy_pj / pj_per_unit:,.3f}"


def bold_font(size: int, spacing: float = 1.5) -> QFont:
    """ฟอนต์หนา + letter-spacing (QSS ไม่รองรับ letter-spacing ต้องทำผ่าน QFont)"""
    f = QFont("Poppins")                       # ไม่มีจะ fallback ให้เอง
    if f.family() not in QFontDatabase().families():
        f = QFont("Arial")
    f.setPointSize(size)
    f.setWeight(QFont.Black)
    f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return f


# ---------------------------------------------------------------- WIDGETS
class CatAvatar(QWidget):
    """วงกลม avatar หน้าแมว"""

    def __init__(self, circle_color: str, cat_color: str, size: int = ROW_H, parent=None):
        super().__init__(parent)
        self.circle_color = QColor(circle_color)
        self.cat_color = QColor(cat_color)
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self.circle_color))
        p.drawEllipse(0, 0, w, h)

        p.setBrush(QBrush(self.cat_color))
        cx, cy = w / 2, h / 2 + h * 0.07
        head_w, head_h = w * 0.58, h * 0.50
        ear_h = h * 0.24

        p.drawPolygon(QPolygonF([
            QPointF(cx - head_w / 2, cy - head_h * 0.30),
            QPointF(cx - head_w / 2 + w * 0.015, cy - head_h / 2 - ear_h),
            QPointF(cx - head_w * 0.10, cy - head_h * 0.40),
        ]))
        p.drawPolygon(QPolygonF([
            QPointF(cx + head_w / 2, cy - head_h * 0.30),
            QPointF(cx + head_w / 2 - w * 0.015, cy - head_h / 2 - ear_h),
            QPointF(cx + head_w * 0.10, cy - head_h * 0.40),
        ]))

        path = QPainterPath()
        path.addRoundedRect(
            QRectF(cx - head_w / 2, cy - head_h / 2, head_w, head_h),
            head_w * 0.38, head_h * 0.38
        )
        p.drawPath(path)

        p.setBrush(QBrush(self.circle_color))
        eye_r = w * 0.052
        p.drawEllipse(QPointF(cx - head_w * 0.21, cy), eye_r, eye_r)
        p.drawEllipse(QPointF(cx + head_w * 0.21, cy), eye_r, eye_r)
        p.end()


class Crown(QWidget):
    def __init__(self, w: int = CROWN_W, h: int = CROWN_H, parent=None):
        super().__init__(parent)
        self.setFixedSize(w, h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor("#FFE85C")))
        w, h = self.width(), self.height()
        p.drawPolygon(QPolygonF([
            QPointF(w * 0.08, h * 0.88), QPointF(w * 0.00, h * 0.20),
            QPointF(w * 0.26, h * 0.55), QPointF(w * 0.50, h * 0.02),
            QPointF(w * 0.74, h * 0.55), QPointF(w * 1.00, h * 0.20),
            QPointF(w * 0.92, h * 0.88),
        ]))
        p.end()


class RankBadge(QLabel):
    def __init__(self, number: int, c1: str, c2: str, size: int = ROW_H, parent=None):
        super().__init__(str(number), parent)
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setFont(bold_font(16, 0.5))
        self.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(x1:0, y1:0, x2:0.9, y2:1,
                            stop:0 {c1}, stop:1 {c2});
                border-radius: {size // 2}px;
                color: {TEXT_DARK};
            }}
        """)


class PlayerPill(QFrame):
    """แถบชื่อ + คะแนน ทรงแคปซูล — filled=มีข้อมูลจริง, ไม่ filled=ช่องว่าง (สีจาง)"""

    def __init__(self, name: str, score: str, c1: str, c2: str,
                 filled: bool = True, parent=None):
        super().__init__(parent)
        self.setFixedHeight(ROW_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        if filled:
            bg = (f"qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                  f"stop:0 {c1}, stop:1 {c2})")
            text_color = TEXT_DARK
        else:
            # ช่องว่าง — โปร่งแสงจาง ๆ ให้ดูออกว่ายังไม่มีคนครอง
            bg = ("qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                  "stop:0 rgba(255,255,255,0.10), stop:1 rgba(255,255,255,0.04))")
            text_color = TEXT_MUTED

        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border-radius: {ROW_H // 2}px;
            }}
            QLabel {{ background: transparent; color: {text_color}; }}
        """)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(28, 0, 28, 0)

        name_lbl = QLabel(name)
        name_lbl.setFont(bold_font(13, 1.0))

        score_lbl = QLabel(score)
        score_lbl.setFont(bold_font(15, 0.5))
        score_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lay.addWidget(name_lbl)
        lay.addStretch()
        lay.addWidget(score_lbl)


def _clear_layout(layout) -> None:
    """ลบ widget ทุกตัวออกจาก layout (ใช้ตอน refresh สร้างแถวใหม่)"""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()


# ---------------------------------------------------------------- COLUMN
class GameColumn(QWidget):
    """คอลัมน์ leaderboard ของเกมเดียว: หัวข้อ + มงกุฎ + 5 แถว"""

    def __init__(self, game: str, label: str) -> None:
        super().__init__()
        self.game = game
        self.color = GAME_COLORS[game]
        self.unit_name, self.pj_per_unit = GAME_UNITS[game]
        self._build_ui(label)

    def _build_ui(self, label: str) -> None:
        header = QLabel(f"{label}  ·  {self.unit_name}")
        header.setAlignment(Qt.AlignCenter)
        header.setFixedHeight(40)
        header.setFont(bold_font(13, 1.0))
        header.setStyleSheet(f"""
            QLabel {{
                background-color: {self.color};
                color: white;
                border-radius: 20px;
            }}
        """)

        # host ของแถว (มงกุฎ + 5 แถว) — สร้างใหม่ทุก refresh
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(GAP)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(bold_font(9, 0.5))
        self.status_label.setStyleSheet(f"color: {TEXT_MUTED};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(GAP)
        root.addWidget(header)
        root.addWidget(self._rows_host)
        root.addWidget(self.status_label)
        root.addStretch()

    def _make_crown_row(self) -> QWidget:
        """แถวมงกุฎ จัดให้อยู่กึ่งกลาง avatar ของอันดับ 1"""
        crown = Crown()
        host = QWidget()
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        # badge(ROW_H) + spacing + ครึ่ง avatar - ครึ่ง crown
        offset = ROW_H + SPACING + ROW_H // 2 - crown.width() // 2
        lay.addSpacing(offset)
        lay.addWidget(crown)
        lay.addStretch()
        host.setFixedHeight(CROWN_H)
        return host

    def _make_row(self, rank: int, name: str, score: str, filled: bool) -> QWidget:
        first = (rank == 1)
        if filled:
            c1, c2 = (GOLD_A, GOLD_B) if first else (MINT_A, MINT_B)
            circle = "#5E5E5E" if first else "#3F5666"
            cat = "#FFF8DA" if first else MINT_A
        else:
            c1 = c2 = "#4A4470"            # badge จาง ๆ
            circle, cat = "#4A4470", "#6E688F"

        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING)
        row.addWidget(RankBadge(rank, c1, c2))
        row.addWidget(CatAvatar(circle, cat))
        row.addWidget(PlayerPill(name, score, c1, c2, filled=filled), 1)
        return host

    def refresh(self) -> None:
        """ดึง TOP 5 ของเกมนี้จาก DB แล้ววาดแถวใหม่ (เติม UNKNOWN ให้ครบ 5)"""
        rows = db.get_top5(self.game)
        _clear_layout(self._rows_layout)

        # มงกุฎขึ้นเฉพาะตอนอันดับ 1 มีคนจริง
        if rows:
            self._rows_layout.addWidget(self._make_crown_row())
        else:
            spacer = QWidget()
            spacer.setFixedHeight(CROWN_H)
            self._rows_layout.addWidget(spacer)

        for i in range(5):
            rank = i + 1
            if i < len(rows):
                r = rows[i]
                name = r["name"]
                score = _fmt_energy(r["best_energy"], self.pj_per_unit)
                filled = True
            else:
                name, score, filled = PLACEHOLDER_NAME, PLACEHOLDER_SCORE, False
            self._rows_layout.addWidget(self._make_row(rank, name, score, filled))

        self.status_label.setText(f"{len(rows)} / 5 รายการ")


# ---------------------------------------------------------------- MAIN
class LeaderboardWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._columns: list[GameColumn] = []
        self.setMinimumSize(1040, 660)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        title = QLabel("🏆 ENERGY LEADERBOARD")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(bold_font(20, 2.0))
        title.setStyleSheet(f"color: {TEXT_LIGHT};")

        subtitle = QLabel("TOP 5 ผู้เล่นที่ใช้พลังงานน้อยที่สุดของแต่ละเกม")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 12px;")

        # 2 คอลัมน์ข้างกัน
        cols = QHBoxLayout()
        cols.setSpacing(40)
        for game, label in GAMES:
            col = GameColumn(game, label)
            self._columns.append(col)
            cols.addWidget(col, stretch=1)

        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.setFixedHeight(38)
        btn_refresh.setCursor(Qt.PointingHandCursor)
        btn_refresh.setFont(bold_font(11, 0.5))
        btn_refresh.setStyleSheet("""
            QPushButton {
                background: rgba(255,255,255,0.12);
                color: #EDEAFB;
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 19px;
                padding: 0 26px;
            }
            QPushButton:hover  { background: rgba(255,255,255,0.20); }
            QPushButton:pressed{ background: rgba(255,255,255,0.30); }
        """)
        btn_refresh.clicked.connect(self.refresh)

        bottom = QHBoxLayout()
        bottom.addStretch()
        bottom.addWidget(btn_refresh)
        bottom.addStretch()

        root = QVBoxLayout(self)
        root.setContentsMargins(50, 28, 50, 32)
        root.setSpacing(6)
        root.addWidget(title)
        root.addWidget(subtitle)
        root.addSpacing(18)
        root.addLayout(cols, stretch=1)
        root.addSpacing(14)
        root.addLayout(bottom)

    def paintEvent(self, event):
        """พื้นหลัง gradient แนวตั้ง"""
        p = QPainter(self)
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0.0, QColor(BG_TOP))
        g.setColorAt(1.0, QColor(BG_BOTTOM))
        p.fillRect(self.rect(), QBrush(g))
        p.end()

    def refresh(self) -> None:
        """รีเฟรชทั้ง 2 คอลัมน์พร้อมกัน"""
        for col in self._columns:
            col.refresh()
