# GUI_NECTEC_SNN

ระบบ 3 แอพที่ทำงานร่วมกัน — เกม 2 ตัวส่งค่าพลังงาน (energy) ที่โมเดลใช้ตอน inference
ไปเก็บที่ Hub กลาง แล้วโชว์ Leaderboard ของคนที่ใช้พลังงาน **น้อยที่สุด**

| แอพ | โฟลเดอร์ | คำอธิบาย |
|---|---|---|
| Hub | `hub/` | ศูนย์กลาง SQLite + GUI Leaderboard |
| Tic-Tac-Toe | `Tictac/` | เขียน X/O บน canvas เล่นกับ AI (SNN / DNN) |
| Alphabet Read | `Alphabet_read/` | เขียนตัวอักษรให้โมเดลทาย (EMNIST, SNN / DNN) |

หน่วยพลังงานมาตรฐานคือ **pJ (picojoule)** — SNN: `E = SOP × 0.9`, DNN: `E = MAC × 4.6`
(อ้างอิง Horowitz, 2014 — 45nm)

---

## Setup (Windows)

ต้องมี **Python 3.11** (แนะนำ ไม่ควรใช้ 3.13 ขึ้นไป เพราะ torch 2.3.0 ยังไม่รองรับ)

```powershell
git clone https://github.com/delphiddd/GUI_NECTEC_SNN.git
cd GUI_NECTEC_SNN

py -3.11 -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

> torch จาก PyPI บน Windows เป็น **CPU build** ซึ่งพอสำหรับ inference ในโปรเจกต์นี้
> ถ้าเครื่องมี NVIDIA GPU และอยากใช้ CUDA ให้ลง torch แยกตาม https://pytorch.org/get-started/locally/

### Setup (macOS / Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## วิธีรัน

รันจาก **root ของโปรเจกต์** ทั้งหมด (path ของ weight อิง `__file__` ไม่ใช่ cwd)

```bash
python hub/main.py             # Hub — Leaderboard
python Tictac/main.py          # เกม Tic-Tac-Toe
python Alphabet_read/main.py   # เกม Alphabet
```

ไม่ต้องเปิด Hub ค้างไว้ก็เล่นเกมได้ — เกมเขียนลง SQLite ตรงผ่าน `shared/energy_client.py`
เปิด Hub เมื่อไหร่ก็เห็นสถิติล่าสุด

---

## ฐานข้อมูล

`hub/data/energy.db` **ไม่ถูก commit ขึ้น git** (อยู่ใน `.gitignore`)
แต่ละเครื่องจะมี DB ของตัวเอง — `hub/db.py` สร้างไฟล์ + ตารางให้อัตโนมัติในการรันครั้งแรก

---

## โครงสร้าง

```
├── hub/
│   ├── main.py              entry point (GUI)
│   ├── db.py                จัดการ SQLite ทั้งหมด (ที่เดียว!)
│   ├── gui/leaderboard.py   หน้า TOP 5
│   └── data/energy.db       (gitignored)
├── Tictac/                  Game App A
├── Alphabet_read/           Game App B
├── shared/
│   └── energy_client.py     ตัวส่ง energy ไป Hub
└── tests/seed_test_data.py  สร้างข้อมูลตัวอย่างไว้เทส Leaderboard
```

> logic ของ DB อยู่ใน `hub/db.py` ที่เดียวเท่านั้น — game app ห้ามเขียน SQL เอง
