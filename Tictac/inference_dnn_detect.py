# inference_dnn_detect.py
# ── DNN inference สำหรับจำแนก X / O ต่อช่องที่ user วาด (เรียลไทม์) ──
# Architecture + weight มาจาก x-o-inference.ipynb (Net + cnn3d_emnist_e20_valAcc-1.00.pth)
#
# หมายเหตุสำคัญเรื่อง invert:
#   tensor ที่ได้จาก img_segment.segment_grid() ผ่าน `1 - x` มาแล้ว (หมึก≈1, พื้น≈0)
#   ซึ่งตรงกับ convention ที่โมเดลถูกเทรน (ภาพ EMNIST ตัวอักษรขาวบนพื้นดำ)
#   → ป้อน tensor เข้าโมเดลได้เลย "ห้าม invert ซ้ำ"

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config (ต้องตรงกับตอนเทรนทุกอย่าง) ──
NUM_OUTPUTS = 2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_PATH = os.path.join(os.path.dirname(__file__), "cnn3d_emnist_e20_valAcc-1.00.pth")

# map index → ตัวอักษร (ตรงกับ notebook: 0=O, 1=X)
alphabets = {0: "O", 1: "X"}


# ── Architecture (คัดลอกเป๊ะจาก x-o-inference.ipynb) ──
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Block 1: 28x28 → MaxPool → 14x14
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(32)

        # Block 2: 14x14 → MaxPool → 7x7
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        # Block 3: 7x7 → MaxPool → 3x3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)

        # Pool
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)

        # Fully connected
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, NUM_OUTPUTS)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.dropout(x)

        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ── Build + load (lazy singleton: โหลด weight ครั้งเดียว ใช้ซ้ำทุกครั้งที่วาด) ──
_model = None


def build_model() -> Net:
    return Net().to(DEVICE)


def load_model(weight_path: str = WEIGHT_PATH) -> Net:
    net = build_model()
    net.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    net.eval()
    return net


def get_model() -> Net:
    """คืนโมเดล singleton — โหลด weight แค่ครั้งแรก กันโหลดซ้ำตอน auto-detect"""
    global _model
    if _model is None:
        _model = load_model()
    return _model


# ── Predict ช่องเดียว ──
def classify_cell(tensor: torch.Tensor, net: Net = None) -> tuple[str, float, int]:
    """
    tensor: [1, 1, 28, 28] (invert มาแล้วจาก segment_grid: หมึก≈1, พื้น≈0)
    returns: (ตัวอักษร 'X'/'O', confidence เป็น %, class index)
    """
    if net is None:
        net = get_model()

    x = tensor.to(DEVICE)
    if x.dim() == 3:          # เผื่อรับ [1,28,28] → เพิ่ม batch dim
        x = x.unsqueeze(0)

    with torch.no_grad():
        output = net(x)
        probs  = F.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)

    idx        = predicted.item()
    confidence = conf.item() * 100.0
    return alphabets[idx], confidence, idx


# ── Predict ทั้งกระดาน — classify เฉพาะช่องที่มีการเขียน ──
def classify_grid(tensors: list, filled: list, net: Net = None) -> list:
    """
    tensors: list ของ [1,1,28,28] ต่อช่อง (จาก segment_grid)
    filled : list ของ bool ต่อช่อง (จาก detect_filled) — True = มีการเขียน
    returns: list ต่อช่อง เป็น dict {'label', 'conf', 'idx'} ถ้ามีการเขียน
             ช่องว่างคืน None (ไม่ส่งเข้าโมเดล)
    """
    if net is None:
        net = get_model()

    results = []
    for t, is_filled in zip(tensors, filled):
        if not is_filled:
            results.append(None)
            continue
        label, conf, idx = classify_cell(t, net)
        results.append({"label": label, "conf": conf, "idx": idx})
    return results


# ── ทดสอบเร็วๆ ──
if __name__ == "__main__":
    net = get_model()
    print(f"โหลดโมเดลสำเร็จ ({sum(p.numel() for p in net.parameters()):,} params) บน {DEVICE}")
    # ป้อนช่องว่าง (ศูนย์) เพื่อเช็คว่า forward ทำงาน
    label, conf, idx = classify_cell(torch.zeros(1, 1, 28, 28))
    print(f"ช่องว่าง → {label} ({conf:.2f}%)")
