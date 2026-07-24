"""
จัดการ SQLite ทั้งหมดของ Hub — SQL อยู่ที่ไฟล์นี้ที่เดียวเท่านั้น
game app ห้ามเขียน SQL เอง ให้เรียกผ่าน shared/energy_client.py

หน่วยพลังงานใน DB = pJ (picojoule) ตาม CLAUDE.md §5
"""
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "energy.db"

# ชื่อตารางเอาไป parameterize (?) ไม่ได้ ต้อง whitelist เท่านั้น
GAMES: dict[str, str] = {
    "tictac": "tictac",
    "alphabet": "alphabet",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tictac (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    energy  REAL NOT NULL          -- pJ
);

CREATE TABLE IF NOT EXISTS alphabet (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    energy  REAL NOT NULL          -- pJ
);

CREATE INDEX IF NOT EXISTS idx_tictac_energy   ON tictac(energy);
CREATE INDEX IF NOT EXISTS idx_alphabet_energy ON alphabet(energy);
"""


def _table(game: str) -> str:
    """แปลงชื่อเกม → ชื่อตาราง พร้อมกัน SQL injection"""
    try:
        return GAMES[game]
    except KeyError:
        raise ValueError(
            f"ไม่รู้จักเกม {game!r} — ต้องเป็นหนึ่งใน {sorted(GAMES)}"
        ) from None


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """สร้างตารางถ้ายังไม่มี — เรียกซ้ำได้ปลอดภัย"""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def add_record(
    game: str,
    name: str,
    energy_pj: float,
    db_path: Optional[Path] = None,
) -> int:
    """บันทึก 1 เกม (append ทุกครั้ง ไม่ทับของเดิม) → คืน id ที่เพิ่งสร้าง"""
    table = _table(game)
    name = name.strip()
    if not name:
        raise ValueError("name ว่างไม่ได้")
    if energy_pj < 0:
        raise ValueError(f"energy ติดลบไม่ได้: {energy_pj}")

    with _connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO {table} (name, energy) VALUES (?, ?)",
            (name, float(energy_pj)),
        )
        return int(cur.lastrowid)


def get_top5(game: str, db_path: Optional[Path] = None) -> list[sqlite3.Row]:
    """TOP 5 คนที่ใช้ energy น้อยที่สุดของเกมนั้น (ยิ่งน้อย = ยิ่งดี)

    นับจาก best score ของแต่ละคน (MIN) ไม่ใช่ค่าเฉลี่ยหรือยอดสะสม
    """
    table = _table(game)
    with _connect(db_path) as conn:
        return conn.execute(
            f"""
            SELECT name, MIN(energy) AS best_energy, COUNT(*) AS plays
            FROM {table}
            GROUP BY name
            ORDER BY best_energy ASC
            LIMIT 5
            """
        ).fetchall()


def get_records(game: str, db_path: Optional[Path] = None) -> list[sqlite3.Row]:
    """ดึงทุกแถวของเกมนั้น เรียงตาม id — ไว้ debug"""
    table = _table(game)
    with _connect(db_path) as conn:
        return conn.execute(
            f"SELECT id, name, energy FROM {table} ORDER BY id"
        ).fetchall()
