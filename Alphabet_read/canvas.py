from PIL import Image
import torch
from torchvision import transforms
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

pre_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
])


class Canvas(QWidget):
    changed = pyqtSignal()

    def __init__(self, width, height):
        super().__init__()
        self.canvas_w = width
        self.canvas_h = height
        self.setFixedSize(width, height)
        self.setCursor(Qt.CrossCursor)

        # ปากกาสเกลตามขนาด canvas (อิงของเดิม 13px @ 1000px)
        # เส้นต้องหนา "เท่าเดิมเชิงสัดส่วน" ไม่งั้นตอน crop ย่อลง 28x28 ภาพจะต่างจากที่โมเดลเคยเห็น
        self.pen_size = max(1, round(13 * width / 1000))
        self.pen_color = QColor(0, 0 , 0)

        self._image = QImage(width, height, QImage.Format_RGB32)
        self.clear()

        self._last_pos = None
        self._drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self._image)

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

    def clear(self):
        self._image.fill(QColor(255, 255, 255))
        self.update()

    def to_pil(self) -> Image.Image:
        self._image = self._image.convertToFormat(QImage.Format_RGB32)
        ptr = self._image.bits()
        ptr.setsize(self._image.byteCount())
        return Image.frombytes("RGBX", (self.canvas_w, self.canvas_h), bytes(ptr)).convert("RGB")

    def to_tensor(self) -> torch.Tensor:
        self._image = self._image.convertToFormat(QImage.Format_RGB32)
        ptr = self._image.bits()
        ptr.setsize(self._image.byteCount())
        pil_img = Image.frombytes("RGBX", (self.canvas_w, self.canvas_h), bytes(ptr)).convert("RGB")
        tensor = pre_transform(pil_img)    # [1, 64, 64] (invert อยู่ใน transform แล้ว)
        return tensor.unsqueeze(0)         # [1, 1, 64, 64]

