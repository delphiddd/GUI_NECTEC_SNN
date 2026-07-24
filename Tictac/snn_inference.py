# snn_inference.py
# ── SNN inference สำหรับให้ AI เลือกช่องเดิน XO (เรียลไทม์ใน GUI) ──
# โมเดล: xo_snn_model.pt — SNN Q-network เล่น tic-tac-toe (บอร์ด 9 ช่อง → Q-value 9 ช่อง)
#
# ไฟล์นี้เป็น inference-only ที่ดึงเฉพาะส่วนที่ GUI ต้องใช้จาก snn_test.py มาเขียนเอง:
#   - ตัด training / self-play / smoke test ออกหมด (GUI ไม่เทรน)
#   - forward รับบอร์ดทีละกระดาน [9] อย่างเดียว (GUI เล่นทีละตา ไม่มี batch)
#   - ตัด surrogate gradient ออก — ใช้แค่ตอน backward ตอนเทรน ไม่กระทบ forward
#   - รวม board→numeric + เลือกช่อง ให้เหลือฟังก์ชันเดียว ai_choose_move()
#
# จุดที่ "ห้ามแก้" (ไม่งั้น weight ที่โหลดมาเพี้ยน):
#   - สถาปัตยกรรม fc1(9,64)→lif1→fc2(64,64)→lif2→fc3(64,9)→lif3 ต้องตรงตอนเทรน
#   - NUM_STEPS=25, BETA=0.9 ต้องเท่าตอนเทรน
#   - lif3 ตั้ง reset_mechanism="none" → เป็น leaky integrator ล้วน, membrane = Q-value (มีค่าลบได้)

import os
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn

from checkwin import EMPTY          # " " — ช่องว่างบนบอร์ด GUI

_HERE = os.path.dirname(__file__)

# ── weight ของแต่ละระดับความยาก (สถาปัตยกรรมเดียวกัน ต่างแค่ weight) ──
WEIGHT_PATHS = {
    "HARD": os.path.join(_HERE, "xo_snn_smart_model.pt"),    # AI ยาก — เทรนจนเล่นเป็น
    "EASY": os.path.join(_HERE, "xo_snn_stupid_model.pt"),   # AI ง่าย — เทรนน้อย เดินมั่ว
}

# ── SNN config (ต้องตรงกับตอนเทรน) ──
NUM_STEPS = 25                      # จำนวน time step ต่อ 1 forward
BETA      = 0.7                 # decay rate ของ membrane potential

# ── Energy model (SOP + MAC) ──
# SOP = spike ที่วิ่งข้าม synapse ไปชั้นถัดไป
# fan-out ของแต่ละ Leaky layer = จำนวน output ของ Linear ชั้นถัดไป
#   lif1 (64) → fc2(64→64) : 64
#   lif2 (64) → fc3(64→9)  : 9
#   lif3 (output)          : 0  (ไม่มีชั้นถัดไป)
#
# fc1 ไม่ใช่ SOP เพราะ input เป็น "กระแส" ป้อนตรง ไม่ใช่ spike — แต่ไม่ใช่ของฟรี
# มันคือ dense matmul จริง → นับเป็น MAC: 9×64 = 576 ต่อ 1 step
# และเพราะป้อนบอร์ดเดิมซ้ำทุก step โค้ดจึงรัน fc1 ใหม่ทุก step → นับ × NUM_STEPS ตามจริง
FAN_OUTS       = [64, 9, 0]
FC1_MACS       = 9 * 64             # MAC ของ fc1 ต่อ 1 time step
ENERGY_PER_SOP = 0.9e-12            # 0.9 pJ ต่อ 1 synaptic operation
ENERGY_PER_MAC = 4.6e-12            # 4.6 pJ ต่อ 1 MAC (FP32 @45nm) — ฐานเดียวกับฝั่ง DNN

# map สัญลักษณ์บนบอร์ด → ค่าตัวเลขที่โมเดลเข้าใจ
_SYM2NUM = {"X": 1.0, "O": -1.0, EMPTY: 0.0}


class SNN_QNet(nn.Module):
    """SNN Q-network (inference-only): board [9] → Q-value [9]

    เข้ารหัส input เป็น "กระแส" ป้อนซ้ำทุก step (current injection) ไม่ใช้ rate coding
    เพราะบอร์ดมีค่า -1 (ฝ่าย O) ที่ rate coding [0,1] เข้ารหัสไม่ได้
    อ่าน output จาก membrane ของชั้นสุดท้าย (lif3) ที่ไม่รีเซ็ต → เป็น Q-value ที่มีค่าลบได้
    """
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(9, 64)
        self.lif1 = snn.Leaky(beta=BETA)
        self.fc2  = nn.Linear(64, 64)
        self.lif2 = snn.Leaky(beta=BETA)
        self.fc3  = nn.Linear(64, 9)
        self.lif3 = snn.Leaky(beta=BETA, reset_mechanism="none")   # leaky integrator ล้วน

    def forward(self, x):
        # x: FloatTensor [9] (บอร์ดเดียว) → (Q [9], total_sops, total_macs)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem3_sum = torch.zeros_like(self.fc3.bias)     # สะสม membrane ของ output ทุก step
        total_sops = 0.0                               # สะสม synaptic operation ทุก step
        total_macs = 0.0                               # สะสม MAC ของ fc1 ทุก step

        for _ in range(NUM_STEPS):
            spk1, mem1 = self.lif1(self.fc1(x),    mem1)   # ป้อนบอร์ดเดิมซ้ำทุก step
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            _,    mem3 = self.lif3(self.fc3(spk2), mem3)
            mem3_sum   = mem3_sum + mem3

            # นับ SOP: จำนวน spike ในชั้นนั้น × fan-out ไปชั้นถัดไป (บวกอย่างเดียว ไม่แตะ Q)
            total_sops += spk1.sum().item() * FAN_OUTS[0]
            total_sops += spk2.sum().item() * FAN_OUTS[1]

            # นับ MAC: fc1 เป็น dense matmul จาก input ที่เป็นค่า analog รันใหม่ทุก step
            total_macs += FC1_MACS

        # (เฉลี่ย membrane = Q-value [9], SOP รวม, MAC รวม)
        return mem3_sum / NUM_STEPS, total_sops, total_macs


# ── cache โมเดลตาม path (โหลดไฟล์ละครั้ง ใช้ซ้ำทุกตา) ──
# key ด้วย path ไม่ใช่ singleton เดียว เพราะ user สลับความยากได้ระหว่าง session
_models: dict[str, SNN_QNet] = {}
_difficulty = "HARD"                # ค่าเริ่มต้น ถ้า GUI ไม่ได้ตั้ง


def set_difficulty(level: str) -> None:
    """level: 'HARD' (smart model) หรือ 'EASY' (stupid model) — GUI เรียกตอน user เลือก"""
    global _difficulty
    if level not in WEIGHT_PATHS:
        raise ValueError(f"ระดับความยากไม่รู้จัก: {level}")
    _difficulty = level


def get_model(level: str | None = None) -> SNN_QNet:
    path = WEIGHT_PATHS[level or _difficulty]
    if path not in _models:
        net = SNN_QNet()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        _models[path] = net
    return _models[path]


def ai_choose_move(board: list, ai_symbol: str, level: str | None = None) -> tuple[int, float]:
    """
    board     : list ของ 'X'/'O'/' ' (len 9) — game.board ปัจจุบัน
    ai_symbol : 'X' หรือ 'O' — ฝั่งที่ AI เล่น
    level     : 'HARD'/'EASY' — ถ้าไม่ส่ง ใช้ค่าที่ตั้งไว้ด้วย set_difficulty()
    returns   : (index ช่องที่ AI เลือกเดิน 0-8 เป็นช่องว่างเสมอ, พลังงานที่ใช้ตานี้ หน่วยจูล)

    perspective: คูณ ai_value ให้ AI เห็นหมากตัวเองเป็น +1 เสมอ (self-play convention ตอนเทรน)
    แล้วเลือกช่องว่างที่ Q สูงสุด
    """
    net      = get_model(level)
    ai_value = 1.0 if ai_symbol == "X" else -1.0
    state    = np.array([_SYM2NUM[c] for c in board], dtype=np.float32) * ai_value
    valid    = [i for i in range(9) if board[i] == EMPTY]

    with torch.no_grad():
        q, sops, macs = net(torch.from_numpy(state))    # Q-value 9 ช่อง + SOP/MAC รวมตานี้
        q = q.numpy()

    idx      = max(valid, key=lambda i: q[i])           # ช่องว่างที่ Q สูงสุด
    energy_j = sops * ENERGY_PER_SOP + macs * ENERGY_PER_MAC   # SOP (ชั้น spike) + MAC (fc1)
    return idx, energy_j


# ── smoke test: python snn_inference.py ──
if __name__ == "__main__":
    empty = [EMPTY] * 9
    for lv in ("HARD", "EASY"):
        print(f"\n── {lv} ({os.path.basename(WEIGHT_PATHS[lv])}) ──")
        print("โหลดโมเดล...", "OK" if get_model(lv) else "FAIL")
        idx, e = ai_choose_move(empty, "X", lv)
        print(f"บอร์ดว่าง AI(X) เลือกช่อง: {idx}  (พลังงาน {e*1e9:.3f} nJ)")

        b = empty.copy(); b[4] = "X"           # คนเล่น X กลางกระดาน
        idx, e = ai_choose_move(b, "O", lv)
        print(f"คน X ลงกลาง, AI(O) ตอบช่อง: {idx}  (พลังงาน {e*1e9:.3f} nJ)")
