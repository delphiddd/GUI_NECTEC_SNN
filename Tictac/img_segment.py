import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from canvas import GRID, CELL_PX

# transform ต่อช่อง → 28×28 (ตรงกับขนาดที่เทรนโมเดล) — ทรานส์ฟอร์มรอบเดียวต่อ cell
cell_transform = transforms.Compose([
    transforms.Resize((CELL_PX, CELL_PX)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),
])


INK_THRESHOLD = 8.0   # ช่องว่างอ่านได้ 0.0 เป๊ะ (เส้นตารางเป็น overlay ไม่ถูกจับ) → ตั้งต่ำได้


def detect_filled(tensors: list, threshold: float = INK_THRESHOLD) -> list:
    """
    ตรวจแต่ละช่องว่ามีการเขียนมั้ย จากผลรวมหมึก (tensor invert: หมึก≈1, พื้น≈0)
    คืน list ของ bool ต่อช่อง (True = มีเครื่องหมาย)
    หมายเหตุ: ยังไม่แยก X/O — ส่วนนั้นรอเสียบโมเดล (classify) ทีหลัง
    """
    return [float(t.sum()) > threshold for t in tensors]


def segment_grid(bgr: np.ndarray, inset: int = 4) -> list:
    """
    แยกรูป board (สี่เหลี่ยมจัตุรัส) เป็น GRID×GRID ช่องเท่าๆ กัน
    เรียงซ้าย→ขวา บน→ล่าง (row-major) → list ของ tensor [1,1,28,28] ต่อช่อง

    inset: ตัดขอบในแต่ละช่องทิ้งกันเส้นตาราง/ลายมือช่องข้างเคียงเล็กเข้ามา
    """
    h, w = bgr.shape[:2]
    cell_h, cell_w = h // GRID, w // GRID

    tensors = []
    for r in range(GRID):
        for c in range(GRID):
            y0 = r * cell_h + inset
            x0 = c * cell_w + inset
            cell = bgr[y0:y0 + cell_h - 2 * inset, x0:x0 + cell_w - 2 * inset]
            rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
            tensors.append(cell_transform(Image.fromarray(rgb)).unsqueeze(0))
    return tensors
