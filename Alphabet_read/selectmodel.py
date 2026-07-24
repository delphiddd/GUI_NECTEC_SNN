from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
from PyQt5.QtCore import Qt


def _fmt_energy(j: float) -> str:
    """แสดง energy ในหน่วย SI suffix + J"""
    abs_j = abs(j)
    if abs_j == 0:
        return "0 J"
    if abs_j < 1e-9:
        return f"{j * 1e12:.1f} pJ"
    if abs_j < 1e-6:
        return f"{j * 1e9:.2f} nJ"
    if abs_j < 1e-3:
        return f"{j * 1e6:.2f} µJ"
    return f"{j:.2e} J"


class _ModelCard(QFrame):
    def __init__(self, name: str, active: bool = True):
        super().__init__()
        self._active = active

        border_color = "#ccc" if active else "#e8e8e8"
        bg_color     = "white" if active else "#f7f7f7"
        header_color = "#2768F5" if active else "#c0c0c0"
        text_color   = "#333"   if active else "#bbb"

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
            }}
        """)
        self.setMinimumWidth(110)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        name_lbl = QLabel(name)
        name_lbl.setAlignment(Qt.AlignCenter)
        name_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {header_color}; border: none;"
        )
        layout.addWidget(name_lbl)

        self._pred_lbl = QLabel("--")
        self._pred_lbl.setAlignment(Qt.AlignCenter)
        self._pred_lbl.setStyleSheet(
            f"font-size: 30px; font-weight: bold; color: {text_color}; border: none;"
        )
        layout.addWidget(self._pred_lbl)

        if active:
            self._conf_lbl = QLabel("-- %")
            self._conf_lbl.setAlignment(Qt.AlignCenter)
            self._conf_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #27ae60; border: none;")
            layout.addWidget(self._conf_lbl)

            self._time_lbl = QLabel("-- ms")
            self._time_lbl.setAlignment(Qt.AlignCenter)
            self._time_lbl.setStyleSheet("font-size: 11px; color: #777; border: none;")
            layout.addWidget(self._time_lbl)

            self._energy_lbl = QLabel("--")
            self._energy_lbl.setAlignment(Qt.AlignCenter)
            self._energy_lbl.setStyleSheet("font-size: 11px; color: #777; border: none;")
            layout.addWidget(self._energy_lbl)
        else:
            soon_lbl = QLabel("coming soon")
            soon_lbl.setAlignment(Qt.AlignCenter)
            soon_lbl.setStyleSheet("font-size: 10px; color: #c0c0c0; border: none;")
            layout.addWidget(soon_lbl)

    def set_result(self, label: str, elapsed_ms: float, energy_j: float, confidence: float = 0.0):
        if not self._active:
            return
        self._pred_lbl.setText(label)
        pct = confidence * 100
        self._conf_lbl.setText(f"{pct:.1f} %")
        color = "#27ae60" if pct >= 70 else "#e67e22" if pct >= 40 else "#e74c3c"
        self._conf_lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color}; border: none;")
        self._time_lbl.setText(f"{elapsed_ms:.1f} ms")
        self._energy_lbl.setText(_fmt_energy(energy_j))

    def reset_result(self):
        self._pred_lbl.setText("--")
        if self._active:
            self._conf_lbl.setText("-- %")
            self._conf_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #27ae60; border: none;")
            self._time_lbl.setText("-- ms")
            self._energy_lbl.setText("--")


class ModelsResultPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            "background-color: white; border: 2px solid #ccc; border-radius: 4px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        title = QLabel("Model Results")
        title.setStyleSheet(
            "font-size: 14px; font-weight: bold; border: none; color: #333;"
        )
        layout.addWidget(title)

        card_row = QVBoxLayout()
        card_row.setSpacing(8)

        self._snn_card = _ModelCard("SNN", active=True)
        self._dnn_card = _ModelCard("DNN", active=True)
        self._svc_card = _ModelCard("SVC", active=False)

        card_row.addWidget(self._snn_card)
        card_row.addWidget(self._dnn_card)
        card_row.addWidget(self._svc_card)

        layout.addLayout(card_row)
        layout.addStretch()

    def update_snn(self, label: str, elapsed_ms: float, energy_j: float, confidence: float = 0.0):
        self._snn_card.set_result(label, elapsed_ms, energy_j, confidence)

    def update_dnn(self, label: str, elapsed_ms: float, energy_j: float, confidence: float = 0.0):
        self._dnn_card.set_result(label, elapsed_ms, energy_j, confidence)

    def clear(self):
        self._snn_card.reset_result()
        self._dnn_card.reset_result()
        self._svc_card.reset_result()
