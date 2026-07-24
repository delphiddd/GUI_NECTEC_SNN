import torch
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

_THUMB = 112   # แสดงแต่ละ crop ที่ 112×112 (4× upscale จาก 28×28)


class CropPreviewWindow(QWidget):
    """popup window แสดง crop ตัวอักษรทั้งหมดที่ segment ได้"""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Segmented Characters")
        self.setStyleSheet("background-color: #1e1e2e;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._title = QLabel("—")
        self._title.setStyleSheet(
            "font-size: 13px; color: #ccc; font-weight: bold;"
        )
        layout.addWidget(self._title)

        self._crop_widget = QWidget()
        self._crop_widget.setStyleSheet("background: transparent;")
        self._crop_row = QHBoxLayout(self._crop_widget)
        self._crop_row.setContentsMargins(0, 0, 0, 0)
        self._crop_row.setSpacing(8)
        self._crop_row.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self._crop_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(_THUMB + 32)
        scroll.setStyleSheet(
            "border: 1px solid #444; border-radius: 6px; background: #12121e;"
        )
        layout.addWidget(scroll)
        layout.addStretch()

        self._labels: list[QLabel] = []

    def show_crops(self, tensors: list):
        for lbl in self._labels:
            self._crop_row.removeWidget(lbl)
            lbl.deleteLater()
        self._labels.clear()

        self._title.setText(f"พบ {len(tensors)} ตัวอักษร")

        for t in tensors:
            arr = (t.squeeze().numpy() * 255).astype(np.uint8)
            h, w = arr.shape
            qimg = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(
                _THUMB, _THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            lbl = QLabel()
            lbl.setFixedSize(_THUMB, _THUMB)
            lbl.setPixmap(pixmap)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                "border: 1px solid #555; border-radius: 4px; background: #0d0d1a;"
            )
            self._crop_row.insertWidget(self._crop_row.count() - 1, lbl)
            self._labels.append(lbl)

        n = len(tensors)
        w = max(300, min(n * (_THUMB + 16) + 64, 900))
        self.resize(w, _THUMB + 100)
        self.show()
        self.raise_()
