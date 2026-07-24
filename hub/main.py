"""
Hub — แอพกลาง entry point

รัน:  python hub/main.py
"""
import sys
from pathlib import Path

# ให้ import hub.* ได้ตอนรันไฟล์นี้ตรง ๆ
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMainWindow  # noqa: E402

from hub import db  # noqa: E402
from hub.gui.leaderboard import LeaderboardWidget  # noqa: E402


class HubWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Energy Hub — Leaderboard")
        # ขั้นต่ำต้องใหญ่พอไม่ให้เนื้อหาโดนตัด (widget ต้องการ ~1040×660)
        self.setMinimumSize(1080, 700)
        self.resize(1080, 700)
        self.setCentralWidget(LeaderboardWidget())

    def keyPressEvent(self, event):
        """F11 สลับเต็มจอ, Esc ออกจากเต็มจอ"""
        if event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
        else:
            super().keyPressEvent(event)


def main() -> int:
    db.init_db()  # กันเคสเปิด Hub ก่อนที่จะมีใครเล่นเกม
    app = QApplication(sys.argv)
    win = HubWindow()
    win.showFullScreen()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
