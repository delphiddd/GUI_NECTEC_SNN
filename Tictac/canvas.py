from PIL import Image
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

# ── Grid config (3x3 X-O board) ──
GRID      = 3        # 3x3 = 9 ช่อง
CELL_PX   = 28       # ขนาด output ต่อช่อง (ตรงกับตอนเทรนโมเดล)
CELL_DISP = 196      # ขนาดช่องบนจอ = 7 * 28 → ย่อลง 28 ลงตัว
GRID_LINE = 4        # ความหนาเส้นตาราง (สีดำ)
BOARD_PX  = GRID * CELL_DISP   # 588


class Canvas(QWidget):
    changed = pyqtSignal()

    def __init__(self, width: int = BOARD_PX, height: int = BOARD_PX):
        super().__init__()
        self.canvas_w = width
        self.canvas_h = height
        self.setFixedSize(width, height)
        self.setCursor(Qt.CrossCursor)

        self.pen_size = 12
        self.pen_color = QColor(0, 0, 0)

        # เก็บเฉพาะลายมือ (พื้นขาว) — เส้นตารางเป็น overlay ไม่ปนลงในรูปนี้
        self._image = QImage(width, height, QImage.Format_RGB32)
        self.clear()

        self._last_pos = None
        self._drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)
        self._draw_grid(painter)

    def _draw_grid(self, painter: QPainter):
        """วาดตาราง 3×3 สีดำทับบนรูป (overlay เท่านั้น)"""
        pen = QPen(QColor(0, 0, 0), GRID_LINE)
        painter.setPen(pen)
        # กรอบนอก
        painter.drawRect(0, 0, self.canvas_w - 1, self.canvas_h - 1)
        # เส้นแบ่งภายใน
        for i in range(1, GRID):
            x = i * CELL_DISP
            painter.drawLine(x, 0, x, self.canvas_h)
            painter.drawLine(0, x, self.canvas_w, x)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = True
            self._last_pos = event.pos()
            self._draw_point(event.pos())

    def mouseMoveEvent(self, event):
        if self._drawing and event.buttons() & Qt.LeftButton:
            self._draw_line(self._last_pos, event.pos())
            self._last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drawing = False
            self._last_pos = None
            self.changed.emit()

    def _draw_point(self, pos: QPoint):
        painter = QPainter(self._image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.pen_color)
        painter.drawEllipse(pos, self.pen_size // 2, self.pen_size // 2)
        painter.end()
        self.update()

    def _draw_line(self, p1: QPoint, p2: QPoint):
        painter = QPainter(self._image)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.pen_color, self.pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)
        painter.end()
        self.update()

    def draw_symbol(self, cell_idx: int, symbol: str):
        """วาด X หรือ O กลางช่อง cell_idx (0-8) ลงบน canvas
        ใช้ตอน AI เดิน — จงใจ 'ไม่' emit changed เพื่อไม่ให้ trigger auto-detect กลางตา AI"""
        r, c = cell_idx // GRID, cell_idx % GRID
        cx = c * CELL_DISP + CELL_DISP // 2
        cy = r * CELL_DISP + CELL_DISP // 2
        half = int(CELL_DISP * 0.28)            # เว้นขอบช่อง ให้อยู่กลางพอดี

        painter = QPainter(self._image)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self.pen_color, self.pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        if symbol == "O":
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPoint(cx, cy), half, half)
        else:                                    # "X"
            painter.drawLine(cx - half, cy - half, cx + half, cy + half)
            painter.drawLine(cx + half, cy - half, cx - half, cy + half)
        painter.end()
        self.update()

    def clear(self):
        self._image.fill(QColor(255, 255, 255))
        self.update()

    def snapshot(self) -> QImage:
        """copy รูปปัจจุบัน (ลายมือล้วน ไม่รวมเส้นตาราง) ไว้เป็น checkpoint"""
        return self._image.copy()

    def restore(self, image: QImage):
        """คืนรูปกลับไปที่ checkpoint — ใช้ลบเฉพาะตาที่เพิ่งวาด ไม่แตะของที่ commit แล้ว"""
        self._image = image.copy()
        self.update()

    def to_pil(self) -> Image.Image:
        self._image = self._image.convertToFormat(QImage.Format_RGB32)
        ptr = self._image.bits()
        ptr.setsize(self._image.byteCount())
        return Image.frombytes("RGBX", (self.canvas_w, self.canvas_h), bytes(ptr)).convert("RGB")
