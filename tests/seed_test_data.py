"""
Seed ข้อมูลทดสอบเข้า Hub — ตารางละ 100 row (tictac + alphabet = 200 row)

จุดประสงค์: ยัดเคสจริง + เคสขอบ เพื่อล่าบั๊กของ leaderboard
(GROUP BY name, MIN(energy), tie-break, float formatting, layout ล้นเมื่อชื่อยาว)

ทุกแถวเขียนผ่าน db.add_record() เท่านั้น — ไม่มี SQL ในไฟล์นี้ (ตาม CLAUDE.md)
default เขียนลง DB ทดสอบแยก ไม่แตะ energy.db ตัวจริง

ใช้งาน:
    python tests/seed_test_data.py                     # → hub/data/energy_test.db
    python tests/seed_test_data.py --db hub/data/energy.db   # ลง DB จริง (ตั้งใจเท่านั้น)
    python tests/seed_test_data.py --clear             # ลบข้อมูลเดิมก่อน (ถามยืนยัน)
    python tests/seed_test_data.py --seed 7            # เปลี่ยน random seed
"""
import argparse
import math
import random
import sqlite3
import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from hub import db  # noqa: E402

ROWS_PER_TABLE = 100
TEST_DB_PATH = _ROOT / "hub" / "data" / "energy.db"

# ── ช่วงพลังงานจริง (pJ) ประเมินจาก inference code ──
# tictac  : SNN 9→64→64→9, 25 step, สะสมเฉพาะตาที่ AI เดิน (~3-5 ตา/เกม)
#           ≈ หลักหมื่น–แสน pJ (0.01–0.2 µJ)
# alphabet: conv SNN fan-out 576/256/26 สะสมทุกตัวอักษรในคำ → ใหญ่กว่ามาก
TICTAC_RANGE = (1.5e4, 2.5e5)
ALPHABET_RANGE = (8.0e5, 6.0e7)

# ผู้เล่นจริง: (name, skill 0.0=เก่งสุด/พลังงานต่ำ 1.0=เปลืองสุด, น้ำหนัก=เล่นบ่อยแค่ไหน)
PLAYERS: list[tuple[str, float, int]] = [
    ("davis", 0.05, 15),
    ("bank", 0.15, 12),
    ("mint", 0.25, 10),
    ("pond", 0.35, 9),
    ("ploy", 0.45, 8),
    ("nano", 0.55, 8),
    ("earth", 0.65, 7),
    ("beam", 0.75, 6),
    ("guitar", 0.85, 5),
    ("newbie", 0.95, 4),
    ("โดม", 0.40, 6),                                  # ชื่อไทย → encoding + layout
    ("ผู้เล่นที่ชื่อยาวมากจนตารางอาจจะล้น", 0.60, 3),   # ชื่อยาว → layout GUI
]


def _lognormal(lo: float, hi: float, skill: float, rng: random.Random) -> float:
    """สุ่มพลังงานแบบ log-normal ในช่วง [lo, hi] เลื่อนตาม skill

    เกมจริงพลังงานเบ้ขวา (เล่นจบเร็วบ่อย, เล่นยาว ๆ นาน ๆ ที)
    skill 0 → เกาะขอบล่าง, skill 1 → เกาะขอบบน, ฟอร์มแกว่งรายเกมด้วย gauss
    """
    center = skill + rng.gauss(0.0, 0.12)
    center = min(max(center, 0.0), 1.0)
    log_lo, log_hi = math.log(lo), math.log(hi)
    return float(math.exp(log_lo + center * (log_hi - log_lo)))


def _edge_cases(game: str) -> list[tuple[str, float]]:
    """เคสขอบที่ตั้งใจยัดให้บั๊กโผล่ → คืน (name, energy_pj)"""
    lo, hi = TICTAC_RANGE if game == "tictac" else ALPHABET_RANGE
    return [
        # 1. ชื่อซ้ำต่างตัวพิมพ์ → GROUP BY name มองเป็นคนละคนกับ 'davis' หรือเปล่า?
        ("Davis", lo * 0.9),
        ("DAVIS", lo * 0.95),
        # 2. ชื่อมี space นำ/ตาม → add_record strip ให้ แต่รวมกลุ่มกับ 'bank' ถูกไหม
        ("  bank  ", lo * 1.1),
        # 3. energy = 0 → ควรขึ้นอันดับ 1 หรือเป็นข้อมูลเสียที่ต้องกรอง?
        ("zero_hero", 0.0),
        # 4. tie เป๊ะ ๆ → tie-break เสถียรไหม (รันซ้ำได้ลำดับเดิมหรือเปล่า)
        ("tie_a", lo * 1.5),
        ("tie_b", lo * 1.5),
        # 5. ต่างกันนิดเดียว → GUI ปัดเลขจนดูเท่ากันไหม
        ("near_a", lo * 2.0),
        ("near_b", lo * 2.0 + 1e-6),
        # 6. ค่าใหญ่มาก (แพ้ยาว ๆ) → ห้ามหลุดเข้า TOP 5
        ("marathon", hi * 20),
        # 7. เล่นครั้งเดียวแต่ดีมาก → ต้องแทรกขึ้น TOP 5 ได้
        ("one_shot", lo * 0.5),
    ]


def build_rows(game: str, rng: random.Random) -> list[tuple[str, float]]:
    """สร้าง ROWS_PER_TABLE แถวของเกมนั้น = เคสขอบ + เคสจริงเติมจนครบ"""
    lo, hi = TICTAC_RANGE if game == "tictac" else ALPHABET_RANGE

    rows: list[tuple[str, float]] = list(_edge_cases(game))

    names = [p[0] for p in PLAYERS]
    skills = {p[0]: p[1] for p in PLAYERS}
    weights = [p[2] for p in PLAYERS]

    while len(rows) < ROWS_PER_TABLE:
        name = rng.choices(names, weights=weights, k=1)[0]
        rows.append((name, _lognormal(lo, hi, skills[name], rng)))

    rows = rows[:ROWS_PER_TABLE]
    rng.shuffle(rows)       # คละลำดับ id ไม่ให้เรียงตามพลังงาน (กันบั๊กที่ซ่อนอยู่ใน ORDER BY)
    return rows


def clear_db(db_path: Path) -> None:
    """ลบข้อมูลเดิมของทั้ง 2 ตาราง — เรียกเมื่อผู้ใช้ยืนยันแล้วเท่านั้น"""
    with sqlite3.connect(db_path) as conn:
        for table in db.GAMES.values():          # whitelist จาก db.GAMES เท่านั้น
            conn.execute(f"DELETE FROM {table}")
    print(f"🗑  ลบข้อมูลเดิมใน {db_path} แล้ว")


def seed(db_path: Optional[Path] = None, seed_value: int = 42, clear: bool = False) -> None:
    path = db_path or TEST_DB_PATH
    db.init_db(path)

    if clear:
        if input(f"จะลบข้อมูลเดิมทั้งหมดใน {path} — ยืนยัน? [yes/N] ").strip().lower() != "yes":
            print("ยกเลิก")
            return
        clear_db(path)

    rng = random.Random(seed_value)

    for game in db.GAMES:
        rows = build_rows(game, rng)
        for name, energy_pj in rows:
            db.add_record(game, name, energy_pj, db_path=path)
        print(f"✅ {game:9s} +{len(rows)} rows")

    print(f"\n📦 DB: {path}")
    for game in db.GAMES:
        print(f"\n── TOP 5 [{game}] (energy น้อยสุด = ดีสุด) ──")
        for rank, row in enumerate(db.get_top5(game, db_path=path), 1):
            print(f"  {rank}. {row['name']:<40s} {row['best_energy']:>18,.6f} pJ  ({row['plays']} plays)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ข้อมูลทดสอบเข้า Hub (ตารางละ 100 row)")
    parser.add_argument("--db", type=Path, default=None, help=f"path ของ DB (default: {TEST_DB_PATH})")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--clear", action="store_true", help="ลบข้อมูลเดิมก่อน (ถามยืนยัน)")
    args = parser.parse_args()

    seed(args.db, args.seed, args.clear)


if __name__ == "__main__":
    main()
