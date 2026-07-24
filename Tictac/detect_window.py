import numpy as np
from PyQt5.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from canvas import GRID


class DetectWindow(QWidget):
    """หน้าต่างแยกแสดงผลตรวจจับ 9 ช่อง (ภาพ 28×28 หลัง preprocess + ช่องไหนมีการเขียน)
    สร้าง 9 ช่องครั้งเดียว แล้ว update_cells() อัปเดตในที่เดิม (รองรับ auto-detect)"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ผลตรวจจับ 9 ช่อง (preprocess 28×28)")
        self.setStyleSheet("background-color: #1a1a2e;")

        grid = QGridLayout(self)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setSpacing(10)

        self._img_labels = []
        self._txt_labels = []
        for i in range(GRID * GRID):
            r, c = i // GRID, i % GRID
            img_lbl = QLabel()
            img_lbl.setFixedSize(120, 120)
            img_lbl.setStyleSheet("border: 2px solid #555; background: #000;")

            txt_lbl = QLabel(f"cell {i}  ({r},{c})  —")
            txt_lbl.setAlignment(Qt.AlignCenter)
            txt_lbl.setStyleSheet("color: #888; font-size: 11px; border: none;")

            holder = QVBoxLayout()
            holder.addWidget(img_lbl, alignment=Qt.AlignCenter)
            holder.addWidget(txt_lbl)
            cell = QWidget()
            cell.setLayout(holder)
            grid.addWidget(cell, r, c)

            self._img_labels.append(img_lbl)
            self._txt_labels.append(txt_lbl)

    def update_cells(self, tensors: list, filled: list, results: list = None):
        if results is None:
            results = [None] * len(tensors)
        for i, (t, is_filled, res) in enumerate(zip(tensors, filled, results)):
            arr = (t.squeeze().numpy() * 255).astype(np.uint8)
            h, w = arr.shape
            qimg = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(120, 120, Qt.KeepAspectRatio, Qt.FastTransformation)

            border = "#27ae60" if is_filled else "#555"
            self._img_labels[i].setPixmap(pix)
            self._img_labels[i].setStyleSheet(f"border: 2px solid {border}; background: #000;")

            r, c = i // GRID, i % GRID
            if is_filled and res is not None:
                # แสดงผล classify X / O + confidence
                tag = f"{res['label']}  {res['conf']:.0f}%"
                color = "#2ecc71"
            elif is_filled:
                tag = "✏️ เขียน"
                color = "#2ecc71"
            else:
                tag = "ว่าง"
                color = "#888"
            self._txt_labels[i].setText(f"cell {i}  ({r},{c})  {tag}")
            self._txt_labels[i].setStyleSheet(f"color: {color}; font-size: 11px; border: none;")
