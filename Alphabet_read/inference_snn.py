import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils

NUM_OUTPUTS = 26       # a-z
NUM_STEPS   = 15
BETA        = 0.6
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fan-out ของแต่ละ Leaky layer → layer ถัดไป
# Leaky #0 → Conv2d(32,64,3×3)    : 64×3×3 = 576
# Leaky #1 → LazyLinear(→256)     : 256
# Leaky #2 → Linear(256→26)       : 26
# Leaky #3 (output)                : 0
FAN_OUTS = [576, 256, 26, 0]

# ── Energy model ──
# ชั้นแรก (Conv1) รับ pixel ดิบ (direct encoding) ไม่ใช่ spike → เป็น MAC ไม่ใช่ SOP
#   MACs/step = H×W×out_ch×(in_ch×k×k) = 28×28×32×(1×3×3) = 225,792
#   BatchNorm พับเข้า conv ได้ตอน inference / MaxPool เป็นการเทียบค่า → ไม่นับเป็น MAC
# direct encoding ป้อนภาพเดิมซ้ำทุก step และโค้ดรัน Conv1 ใหม่ทุก step → นับ × NUM_STEPS ตามจริง
STEM_MACS = 28 * 28 * 32 * (1 * 3 * 3)
E_SOP = 0.9e-12   # 0.9 pJ ต่อ synaptic operation (spike-driven accumulate)
E_MAC = 4.6e-12   # 4.6 pJ ต่อ MAC (FP32 @45nm)

ENG_MAP = {i: chr(ord('a') + i) for i in range(NUM_OUTPUTS)}


def build_model():
    spike_grad = surrogate.fast_sigmoid(slope=25)
    net = nn.Sequential(
        # Block 1: 28×28 → 14×14
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),
        snn.Leaky(beta=BETA, spike_grad=spike_grad, init_hidden=True),

        # Block 2: 14×14 → 7×7
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),
        snn.Leaky(beta=BETA, spike_grad=spike_grad, init_hidden=True),

        # FC: Flatten → LazyLinear (PyTorch หา input size ให้เอง)
        nn.Flatten(),
        nn.LazyLinear(256),
        nn.BatchNorm1d(256),
        snn.Leaky(beta=BETA, spike_grad=spike_grad, init_hidden=True),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_OUTPUTS),
        snn.Leaky(beta=BETA, spike_grad=spike_grad, init_hidden=True, output=True),
    ).to(DEVICE)
    return net


def load_model(pth_path: str):
    net = build_model()
    # eval ก่อน dummy forward เพราะ BatchNorm1d ต้องการ eval mode เมื่อ batch=1
    net.eval()
    dummy = torch.zeros(1, 1, 28, 28).to(DEVICE)
    utils.reset(net)
    with torch.no_grad():
        net(dummy)
    net.load_state_dict(torch.load(pth_path, map_location=DEVICE))
    net.eval()
    return net


def predict(net, img_tensor: torch.Tensor) -> tuple[int, str, list[float], float]:
    """returns (predicted_idx, eng_label, confidence_per_class, energy_joules)"""
    img_tensor = img_tensor.to(DEVICE)

    utils.reset(net)

    spk_rec    = []
    total_sops = 0.0
    total_macs = 0.0

    with torch.no_grad():
        for _ in range(NUM_STEPS):
            x = img_tensor   # direct encoding: ส่ง pixel ตรงๆ ทุก step (ตรงกับ training)
            total_macs += STEM_MACS   # Conv1 รันใหม่ทุก step → บวก MAC ทุก step
            leaky_idx = 0
            for layer in net:
                x = layer(x)
                if isinstance(layer, snn.Leaky): #เช็คว่า layer ที่เพิ่งผ่านไปเป็น Leaky neuron (LIF) 
                    """
                    Leaky layer อาจ return ได้ 2 แบบ ขึ้นกับ config:
                    - ตัวธรรมดา (init_hidden=True) → return spike อย่างเดียว = tensor
                    - ตัว output (output=True) → return tuple (spk, mem) = (spike, membrane potential)
                    """
                    spk = x[0] if isinstance(x, tuple) else x
                    total_sops += spk.detach().float().sum().item() * FAN_OUTS[leaky_idx]
                    leaky_idx += 1
            spk_out = x[0] if isinstance(x, tuple) else x
            spk_rec.append(spk_out)

    spike_counts = torch.stack(spk_rec).sum(dim=0).squeeze(0)  # [26]
    predicted    = spike_counts.argmax().item()
    total        = spike_counts.sum()
    confidence   = (spike_counts / total).tolist() if total > 0 else [0.0] * NUM_OUTPUTS
    energy_j     = total_sops * E_SOP + total_macs * E_MAC  # SOP (ชั้น spike) + MAC (Conv1)

    return predicted, ENG_MAP[predicted], confidence, energy_j
