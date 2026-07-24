import torch
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


class PreprocessPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: white; border: 2px solid #ccc; border-radius: 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Preprocess")
        title.setStyleSheet("font-size: 14px; font-weight: bold; border: none; color: #333;")
        layout.addWidget(title)

        self.image_label = QLabel("กด Submit เพื่อดูรูป 64×64")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(200, 200)
        self.image_label.setStyleSheet(
            "background-color: #f9f9f9; border: 1px dashed #999; color: #999; font-size: 12px; border-radius: 4px;"
        )
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # แถวแสดง crop แต่ละตัวอักษร (Debug)
        self._crop_widget = QWidget()
        self._crop_row = QHBoxLayout(self._crop_widget)
        self._crop_row.setContentsMargins(0, 0, 0, 0)
        self._crop_row.setSpacing(4)
        self._crop_row.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self._crop_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(76)
        scroll.setStyleSheet("border: 1px dashed #ccc; border-radius: 4px; background: #f9f9f9;")
        layout.addWidget(scroll)

        self._crop_labels = []
        layout.addStretch()

    def clear(self):
        self.image_label.clear()
        self.image_label.setText("กด Submit เพื่อดูรูป 64×64")
        self.update_crops([])

    def update_image(self, tensor: torch.Tensor):
        img_np = (tensor.squeeze().numpy() * 255).astype(np.uint8)
        h, w = img_np.shape
        qimg = QImage(img_np.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimg).scaled(200, 200, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")

    def update_crops(self, tensors: list):
        """แสดง preprocessed crop แต่ละตัวอักษร (จาก Debug)"""
        for lbl in self._crop_labels:
            self._crop_row.removeWidget(lbl)
            lbl.deleteLater()
        self._crop_labels.clear()

        for t in tensors:
            arr = (t.squeeze().numpy() * 255).astype(np.uint8)
            h, w = arr.shape
            qimg = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(64, 64, Qt.KeepAspectRatio, Qt.FastTransformation)
            lbl = QLabel()
            lbl.setFixedSize(64, 64)
            lbl.setPixmap(pixmap)
            lbl.setStyleSheet("border: 1px solid #ccc;")
            self._crop_row.insertWidget(self._crop_row.count() - 1, lbl)
            self._crop_labels.append(lbl)
