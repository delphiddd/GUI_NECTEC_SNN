# dnn_inference.py
# ── DNN inference สำหรับให้ AI เลือกช่องเดิน XO (เรียลไทม์ใน GUI) ──
# โมเดล: xo_dnn_rl_model.pt — DNN Q-network เล่น tic-tac-toe (บอร์ด 9 ช่อง → Q-value 9 ช่อง)
#
# ไฟล์นี้เป็นคู่เทียบของ snn_inference.py: API เหมือนกันเป๊ะ (ai_choose_move → (idx, energy_j))
# ต่างกันแค่ตัวโมเดลและวิธีนับพลังงาน
#   - SNN นับ SOP  (spike ที่วิ่งข้าม synapse — sparse, นับเฉพาะที่ spike จริง)
#   - DNN นับ MACs (dense matmul — คำนวณทุก connection เสมอ ไม่ว่า activation จะเป็น 0 หรือไม่)
#
# จุดที่ "ห้ามแก้" (ไม่งั้น weight ที่โหลดมาเพี้ยน):
#   - สถาปัตยกรรม fc1(9,64)→ReLU→fc2(64,64)→ReLU→fc3(64,9) ต้องตรงตอนเทรน
#   - fc3 ไม่มี activation — output = Q-value ตรงๆ (มีค่าลบได้)

import os
import numpy as np
import torch
import torch.nn as nn

from checkwin import EMPTY          # " " — ช่องว่างบนบอร์ด GUI

_HERE = os.path.dirname(__file__)

WEIGHT_PATH = os.path.join(_HERE, "xo_dnn_rl_model.pt")

# ── Net config (ต้องตรงกับตอนเทรน) ──
INPUT_SIZE  = 9                     # บอร์ด 9 ช่อง
HIDDEN_SIZE = 64
OUTPUT_SIZE = 9                     # Q-value ช่องละค่า

# ── Energy model (นับ MACs = multiply-accumulate ของ fully-connected layer) ──
# 4.6 pJ ต่อ 1 MAC (FP32 @45nm: mult 3.7 pJ + add 0.9 pJ) — ฐานเดียวกับ 0.9 pJ/SOP ของ SNN
ENERGY_PER_MAC = 4.6e-12

# map สัญลักษณ์บนบอร์ด → ค่าตัวเลขที่โมเดลเข้าใจ
_SYM2NUM = {"X": 1.0, "O": -1.0, EMPTY: 0.0}


class Net(nn.Module):
    """DNN Q-network: board [9] หรือ [batch, 9] → Q-value รูปเดียวกัน
    การวัดพลังงาน: สะสมจำนวน MACs ไว้ใน self.mac_count ทุกครั้งที่ forward
    MACs = Multiply-Accumulate operations ของแต่ละ fully-connected layer
      fc1 : INPUT_SIZE  × HIDDEN_SIZE
      fc2 : HIDDEN_SIZE × HIDDEN_SIZE
      fc3 : HIDDEN_SIZE × OUTPUT_SIZE
    นับทุก layer เพราะ dense matmul คำนวณทุก connection เสมอ ไม่ว่า activation จะเป็น 0 หรือไม่
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)   # output = Q-value ตรงๆ ไม่มี activation
        self.mac_count = 0.0    # ตัวนับ MACs สะสม — อ่าน/รีเซ็ตจากภายนอกได้

    def reset_macs(self):
        self.mac_count = 0.0

    def energy_joules(self):
        return self.mac_count * ENERGY_PER_MAC

    def forward(self, x):
        squeeze = (x.dim() == 1)
        if squeeze:
            x = x.unsqueeze(0)                # [9] → [1, 9]

        batch_size = x.shape[0]

        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        q  = self.fc3(h2)

        # นับ MACs
        self.mac_count += batch_size * INPUT_SIZE  * HIDDEN_SIZE   # fc1
        self.mac_count += batch_size * HIDDEN_SIZE * HIDDEN_SIZE   # fc2
        self.mac_count += batch_size * HIDDEN_SIZE * OUTPUT_SIZE   # fc3

        return q.squeeze(0) if squeeze else q


# ── cache โมเดล (โหลดไฟล์ครั้งเดียว ใช้ซ้ำทุกตา) ──
_model: Net | None = None


def get_model() -> Net:
    global _model
    if _model is None:
        net = Net()
        net.load_state_dict(torch.load(WEIGHT_PATH, map_location="cpu"))
        net.eval()
        _model = net
    return _model


def ai_choose_move(board: list, ai_symbol: str) -> tuple[int, float]:
    """
    board     : list ของ 'X'/'O'/' ' (len 9) — game.board ปัจจุบัน
    ai_symbol : 'X' หรือ 'O' — ฝั่งที่ AI เล่น
    returns   : (index ช่องที่ AI เลือกเดิน 0-8 เป็นช่องว่างเสมอ, พลังงานที่ใช้ตานี้ หน่วยจูล)

    perspective: คูณ ai_value ให้ AI เห็นหมากตัวเองเป็น +1 เสมอ (self-play convention ตอนเทรน)
    แล้วเลือกช่องว่างที่ Q สูงสุด
    """
    net      = get_model()
    ai_value = 1.0 if ai_symbol == "X" else -1.0
    state    = np.array([_SYM2NUM[c] for c in board], dtype=np.float32) * ai_value
    valid    = [i for i in range(9) if board[i] == EMPTY]

    net.reset_macs()                                    # นับพลังงานเฉพาะตานี้
    with torch.no_grad():
        q = net(torch.from_numpy(state)).numpy()        # Q-value ทั้ง 9 ช่อง

    idx      = max(valid, key=lambda i: q[i])           # ช่องว่างที่ Q สูงสุด
    energy_j = net.energy_joules()                      # แปลง MACs → พลังงาน (จูล)
    return idx, energy_j


# ── smoke test: python dnn_inference.py ──
if __name__ == "__main__":
    empty = [EMPTY] * 9
    print(f"── DNN ({os.path.basename(WEIGHT_PATH)}) ──")
    print("โหลดโมเดล...", "OK" if get_model() else "FAIL")
    idx, e = ai_choose_move(empty, "X")
    print(f"บอร์ดว่าง AI(X) เลือกช่อง: {idx}  (พลังงาน {e*1e9:.3f} nJ)")

    b = empty.copy(); b[4] = "X"           # คนเล่น X กลางกระดาน
    idx, e = ai_choose_move(b, "O")
    print(f"คน X ลงกลาง, AI(O) ตอบช่อง: {idx}  (พลังงาน {e*1e9:.3f} nJ)")
