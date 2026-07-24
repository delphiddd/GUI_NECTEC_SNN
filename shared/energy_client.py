import sys
from pathlib import Path

# ให้ import hub.db ได้โดยไม่ต้องติดตั้งเป็น package
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hub import db  # noqa: E402

J_TO_PJ = 1e12


def send_energy(game: str, name: str, energy_joules: float) -> int:
    """ส่งผลพลังงานของ 1 เกมเข้า Hub → คืน record id

    game          : 'tictac' หรือ 'alphabet'
    energy_joules : ค่าที่ inference คืนมา (หน่วย J) — แปลงเป็น pJ ให้เอง
    """
    db.init_db()
    energy_pj = float(energy_joules) * J_TO_PJ
    return db.add_record(game, name, energy_pj)


def send_energy_pj(game: str, name: str, energy_pj: float) -> int:
    """เผื่อกรณีที่ผู้เรียกมีค่าเป็น pJ อยู่แล้ว ไม่ต้องแปลงซ้ำ"""
    db.init_db()
    return db.add_record(game, name, energy_pj)
