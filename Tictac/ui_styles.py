from PyQt5.QtWidgets import QPushButton

# ── สีปุ่มแต่ละแบบ: (bg, hover, pressed) ──
BTN_PALETTES = {
    "x":       ("#2980b9", "#2471a3", "#1f618d"),
    "o":       ("#8e44ad", "#7d3c98", "#6c3483"),
    "confirm": ("#27ae60", "#1e8449", "#196f3d"),
    "undo":    ("#f39c12", "#d68910", "#b9770e"),
    "break":   ("#e74c3c", "#BD0E00", "#c0392b"),
    "human":   ("#16a085", "#13876f", "#0e6655"),
    "ai":      ("#34495e", "#2c3e50", "#212f3d"),
    "easy":    ("#2ecc71", "#28b463", "#239b56"),
    "hard":    ("#c0392b", "#a93226", "#922b21"),
    "dnn":     ("#2980b9", "#2471a3", "#1f618d"),
    "start":   ("#27ae60", "#219150", "#1b7742"),   # ปุ่ม "เริ่มเกม" หน้าแรก — เขียวชุดเดียวกับ Alphabet_read
}

WINDOW_BG = "background-color: #f0f0f0;"
# โปร่งใส + สว่าง อ่านออกบนพื้น gradient (ไม่งั้น WINDOW_BG แผ่เทาทับเป็นบาร์ขาว)
STATUS_STYLE = "font-size: 14px; font-weight: bold; color: #EDEAFB; background: transparent;"
CANVAS_FRAME_STYLE = "background: white; border: 2px solid #ccc; border-radius: 4px;"

# หน้าแรก (StartPage) — สไตล์เดียวกับ Alphabet_read เป๊ะ ๆ
START_TITLE_STYLE    = "font-size: 28px; font-weight: bold; color: #2768F5;"
START_SUBTITLE_STYLE = "font-size: 14px; color: #555;"

# ช่องกรอกชื่อผู้เล่นในเฟสแรก — ให้หน้าตาเข้าชุดกับ Alphabet_read
NAME_INPUT_STYLE = """
    QLineEdit {
        background: white;
        color: #333;
        border: 2px solid #ccc;
        border-radius: 6px;
        padding: 0 12px;
        font-size: 16px;
    }
    QLineEdit:focus { border: 2px solid #2768F5; }
"""


def btn_style(bg: str, hover: str, pressed: str) -> str:
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
    """


def make_button(text: str, palette: str, on_click=None, height: int = 40) -> QPushButton:
    """สร้างปุ่มสำเร็จรูป: ตั้งความสูง + สีตาม palette + ต่อ signal ในที่เดียว"""
    btn = QPushButton(text)
    btn.setFixedHeight(height)
    btn.setStyleSheet(btn_style(*BTN_PALETTES[palette]))
    if on_click is not None:
        btn.clicked.connect(on_click)
    return btn
