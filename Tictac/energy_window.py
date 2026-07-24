# energy_window.py
# ── Panel กราฟพลังงาน AI แบบเรียลไทม์ (ฝังในหน้าต่างเกม ฝั่งขวาข้าง canvas) ──
# เพิ่มแท่ง 1 อันทุกครั้งที่ AI เดิน (add_move) → โชว์พลังงานราย "ตา AI" หน่วย nJ
# พลังงานมาจาก SOP (snn_inference) หรือ MACs (dnn_inference) แล้วแต่โมเดลที่ user เลือก

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# หมายเหตุ: matplotlib จัด "สระ/วรรณยุกต์" ไทยซ้อนกับพยัญชนะไม่ได้ (ไม่มี text shaping)
# → label ในกราฟใช้อังกฤษสั้นๆ ส่วนข้อความไทยเต็มๆ อยู่ที่ QLabel ด้านบน (Qt render ได้สวย)

J_TO_NJ = 1e9           # จูล → นาโนจูล (nJ)


class EnergyPanel(QWidget):
    """แผงกราฟพลังงาน AI เรียลไทม์ ฝังในหน้าต่างเกม

    วิธีใช้ใน GUI:
        panel = EnergyPanel()
        panel.set_model_name("DNN")  # ชื่อโมเดลที่จะโชว์บนป้าย (default "SNN")
        panel.reset()               # เริ่มเกมใหม่ → ล้างแท่งเดิม
        panel.add_move(energy_j)    # ทุกตาที่ AI เดิน → เพิ่มแท่ง + วาดใหม่ทันที
    """

    def __init__(self):
        super().__init__()
        self._nj = []               # พลังงานราย AI move สะสมไว้ (หน่วย nJ)
        self._model_name = "SNN"    # โชว์บนป้ายไทย — เปลี่ยนตามโมเดลที่ user เลือก
        self.setFixedWidth(240)     # คอลัมน์ผอมๆ ฝั่งขวา ข้าง canvas

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ป้ายไทยด้านบน (Qt render ภาษาไทยได้ชัด) — ตัดหลายบรรทัดให้พอดีคอลัมน์แคบ
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #196f3d; font-size: 13px; font-weight: bold; border: none;")
        layout.addWidget(self.label)

        # กราฟแท่งทรงสูงผอม (สีเข้ากับธีมสว่างของหน้าต่างหลัก #f0f0f0)
        self.fig = Figure(figsize=(2.3, 5.4), facecolor="#f0f0f0")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)      # ยืดสูงเต็มแถวให้เท่า canvas เอง

        self.reset()

    def set_model_name(self, name: str):
        """ตั้งชื่อโมเดลบนป้าย ('SNN ยาก', 'DNN', ...) — GUI เรียกตอนเริ่มเกม"""
        self._model_name = name
        self._redraw()

    def reset(self):
        """เริ่มเกมใหม่ — ล้างแท่งทั้งหมด"""
        self._nj = []
        self._redraw()

    def add_move(self, energy_j: float):
        """เพิ่มพลังงานของตา AI ล่าสุด (จูล) แล้ววาดกราฟใหม่ทันที (เรียลไทม์)"""
        self._nj.append(energy_j * J_TO_NJ)     # จูล → nJ
        self._redraw()

    def _redraw(self):
        total = sum(self._nj)
        n     = len(self._nj)

        # ── ป้ายไทยด้านบน (หลายบรรทัด พอดีคอลัมน์แคบ) ──
        if n == 0:
            self.label.setText(f"⚡ พลังงาน {self._model_name}\nรอ AI เดินตาแรก...")
        else:
            self.label.setText(
                f"⚡ พลังงาน {self._model_name}\n"
                f"ตาที่ {n}: +{self._nj[-1]:.2f} nJ\n"
                f"รวมทั้งเกม {total:.2f} nJ"
            )

        # ── กราฟแท่งทรงสูงผอม ──
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#ffffff")

        moves = list(range(1, n + 1))
        if n > 0:
            ax.bar(moves, self._nj, color="#2ecc71", width=0.55, zorder=3)
            for x, y in zip(moves, self._nj):
                ax.text(x, y, f"{y:.1f}", ha="center", va="bottom",
                        color="#333", fontsize=8, zorder=4)
            ax.set_xticks(moves)
            ax.set_xlim(0.4, n + 0.6)           # เว้นขอบซ้ายขวาให้แท่งลอยกลาง
        else:
            ax.set_xticks([])

        ax.set_title("Energy (nJ)", color="#333", fontsize=10, pad=8)
        ax.set_xlabel("AI move #", color="#555", fontsize=9)
        ax.tick_params(colors="#555", labelsize=8)
        ax.margins(y=0.20)                       # เผื่อที่ให้ตัวเลขบนหัวแท่ง
        ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#ddd", zorder=0)
        ax.spines["top"].set_visible(False)      # เก็บกวาดเส้นกรอบให้ดูโปร่ง
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ccc")
        ax.spines["bottom"].set_color("#ccc")

        self.fig.tight_layout()
        self.canvas.draw()
