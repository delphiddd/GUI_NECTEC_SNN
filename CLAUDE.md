# CLAUDE.md

เอกสารนี้เป็น context หลักสำหรับ Claude Code ในโปรเจกต์นี้
**อ่านไฟล์นี้ให้จบก่อนทำอะไรทุกครั้ง**

---

## ⚠️ กฎเหล็ก (READ FIRST — NON-NEGOTIABLE)

1. **ห้ามแก้ไข / สร้าง / ลบไฟล์โค้ดใด ๆ ก่อนได้รับอนุญาตจากเจ้าของโปรเจกต์**
2. **ก่อนลงมือทุกครั้ง ต้องเสนอแผนก่อน** ในรูปแบบนี้:
   - จะแตะไฟล์ไหนบ้าง (list path ให้ครบ)
   - จะแก้อะไรในแต่ละไฟล์ (สรุปเป็นข้อ ๆ)
   - มีผลกระทบกับส่วนไหนอีกบ้าง (breaking change?)
   - ถ้ามีหลายวิธี → เสนอ 2–3 ทางเลือก พร้อมข้อดี/ข้อเสีย
3. **รอคำว่า "โอเค" / "ทำเลย" ก่อนเสมอ** ไม่มีการเดาใจแล้วลุยเอง
4. อ่านโค้ดได้เสรี (`read`, `grep`, `ls`) — ไม่ต้องขออนุญาต
5. รันคำสั่งที่ไม่ทำลายข้อมูลได้ (เช่น `python -m pytest`, `sqlite3 ... "SELECT ..."`) — แต่ถ้าเป็นคำสั่งที่ **เขียน/ลบ** ข้อมูล ต้องถามก่อน
6. ถ้าไม่ชัวร์ → **ถาม อย่าเดา** โดยเฉพาะเรื่อง schema, หน่วยพลังงาน, และ interface ระหว่างแอพ

---

## 1. ภาพรวมโปรเจกต์

ระบบประกอบด้วย **3 แอพที่ทำงานร่วมกัน** โดยมีแอพกลางเป็นศูนย์รวมข้อมูลพลังงาน (energy) ที่โมเดลใช้ตอนเล่นเกม

```
┌──────────────┐        ┌──────────────┐
│  Game App A  │        │  Game App B  │
│  (Tic-Tac-Toe│        │  (Digit /    │
│   SNN vs DNN)│        │   EMNIST)    │
└──────┬───────┘        └──────┬───────┘
       │  ส่ง energy หลังเกมจบ  │
       └────────────┬───────────┘
                    ▼
          ┌────────────────────┐
          │    HUB (แอพกลาง)   │
          │  - SQLite DB       │
          │  - GUI Leaderboard │
          │  - TOP 5 พลังงาน   │
          │    "น้อยที่สุด"    │
          └────────────────────┘
```

### เป้าหมายหลัก
- Hub เก็บสถิติพลังงานของผู้เล่นทุกคน
- Leaderboard แสดง **TOP 5 คนที่ใช้ energy น้อยที่สุด** (ยิ่งน้อย = ยิ่งดี = อันดับสูง)
- Game app ทั้ง 2 ตัว **ไม่ต้องรู้จักกัน** รู้จักแค่ Hub

---

## 2. โครงสร้าง Repo (proposed — ยังปรับได้)

```
project-root/
├── CLAUDE.md                 ← ไฟล์นี้
├── hub/                      ← แอพกลาง
│   ├── main.py               ← entry point (GUI)
│   ├── db.py                 ← จัดการ SQLite ทั้งหมด (ที่เดียว!)
│   ├── gui/
│   │   └── leaderboard.py    ← หน้า TOP 5
│   └── data/
│       └── energy.db         ← SQLite file (gitignore)
│
├── app_tictactoe/            ← Game App A
│   └── main.py
│
├── app_digit/                ← Game App B
│   └── main.py
│
└── shared/                   ← โค้ดที่ใช้ร่วมกัน
    ├── energy_client.py      ← ตัวส่ง energy ไป Hub
    └── schema.py             ← dataclass ของ payload (single source of truth)
```

**หลักการ:** logic ของ DB อยู่ใน `hub/db.py` ที่เดียวเท่านั้น game app ห้ามเขียน SQL เอง

---

## 3. Data Contract (สำคัญที่สุด — ห้ามแก้เองเด็ดขาด)

### 3.1 SQLite Schema

```sql
-- ตารางผู้เล่น
CREATE TABLE IF NOT EXISTS players (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ตารางบันทึกพลังงานต่อ 1 เกม
CREATE TABLE IF NOT EXISTS energy_records (
    record_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   INTEGER NOT NULL,
    app_name    TEXT NOT NULL,          -- 'tictactoe' | 'digit'
    model_type  TEXT NOT NULL,          -- 'SNN' | 'DNN'
    energy      REAL NOT NULL,          -- หน่วย pJ (picojoule)
    sop_count   INTEGER,                -- สำหรับ SNN
    mac_count   INTEGER,                -- สำหรับ DNN
    played_at   TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

CREATE INDEX IF NOT EXISTS idx_energy ON energy_records(energy);
```

### 3.2 Query สำหรับ Leaderboard (TOP 5 น้อยสุด)

```sql
SELECT p.name, MIN(e.energy) AS best_energy, e.model_type
FROM energy_records e
JOIN players p ON p.id = e.player_id
GROUP BY p.id
ORDER BY best_energy ASC
LIMIT 5;
```

> ⚠️ **ต้องเคลียร์ก่อนเขียนโค้ด:** TOP 5 นับจาก
> (a) energy ต่ำสุดของแต่ละคน, (b) ค่าเฉลี่ยของแต่ละคน, หรือ (c) energy รวมสะสม?
> ตอนนี้ default ไว้ที่ **(a)** — ถ้าไม่ใช่ให้บอกก่อน

### 3.3 Payload ที่ Game App ส่งมา

```python
{
    "player_name": "davis",
    "app_name":    "tictactoe",   # หรือ "digit"
    "model_type":  "SNN",         # หรือ "DNN"
    "energy":      1234.56,       # pJ
    "sop_count":   45000,         # optional
    "mac_count":   None           # optional
}
```

---

## 4. วิธี Integrate (ต้องเลือก 1 — ยังไม่ตัดสินใจ)

| ทางเลือก | ข้อดี | ข้อเสีย |
|---|---|---|
| **A. เขียน SQLite ตรง** ผ่าน `shared/energy_client.py` | ง่ายสุด ไม่ต้องรัน server | เขียนพร้อมกันหลายแอพอาจ lock ได้ |
| **B. HTTP local API** (FastAPI/Flask ใน Hub) | สะอาด แยกชั้นชัด แตกเป็น network ได้ในอนาคต | ต้องเปิด Hub ค้างไว้ตลอด |
| **C. เขียน JSON ลง queue folder** แล้ว Hub คอย poll | ทนต่อการที่ Hub ไม่ได้เปิด | ข้อมูลไม่ real-time |

**ข้อเสนอของผม: A** ก่อน (เพราะ single-machine, จำนวนคนน้อย) แล้วค่อยอัปเกรดเป็น B ถ้าจำเป็น
→ **รอคำตัดสินจากเจ้าของโปรเจกต์**

---

## 5. Energy Convention (ต้องยึดให้ตรงกันทุกแอพ)

- หน่วยมาตรฐาน: **pJ (picojoule)** — game app ทุกตัวต้องแปลงเป็น pJ ก่อนส่ง
- SNN: `E = SOP × E_ac`
- DNN: `E = MAC × E_mac`
- ค่าอ้างอิง (Horowitz, 2014 — 45nm):
  - `E_ac (32-bit FP add)` = 0.9 pJ
  - `E_mac (32-bit FP MAC)` = 4.6 pJ
- SOP counting ต้องนับจาก **spike ที่เกิดขึ้นจริง** × fan-out ไม่ใช่นับจาก layer size
- ถ้าจะเปลี่ยนสมมติฐานพลังงาน → **ถามก่อน** เพราะกระทบผลวิจัยโดยตรง

---

## 6. Tech Stack

- Python 3.10+
- GUI: **PyQt6** (ให้สอดคล้องกับงานเดิม)
- DB: SQLite3 (stdlib)
- ML: PyTorch + snnTorch
- ห้ามเพิ่ม dependency ใหม่โดยไม่ถาม

---

## 7. Coding Style

- ตั้งชื่อ variable/function เป็นภาษาอังกฤษ, comment เป็นไทยได้
- Type hints ทุก public function
- `db.py` ใช้ context manager (`with sqlite3.connect(...)`) เสมอ
- ห้าม hardcode path — ใช้ `pathlib.Path(__file__).parent`
- ห้าม `except: pass` เงียบ ๆ

---

## 8. Open Questions (ยังไม่มีคำตอบ — ต้องถามก่อนเขียนโค้ด)

1. Leaderboard นับ energy แบบไหน? (min / avg / total ต่อคน)
2. เลือก integration แบบ A, B หรือ C?
3. Game App B คือแอพอะไรแน่? (EMNIST digit recognition ใช่ไหม)
4. ผู้เล่นระบุตัวตนยังไง? พิมพ์ชื่อเอง หรือมี login?
5. ถ้าคนเดิมเล่นซ้ำ → เก็บทุกครั้ง (append) หรือทับของเดิม?
6. Leaderboard แยกตาม app / model_type ไหม หรือรวมทุกอย่างเป็นตารางเดียว?