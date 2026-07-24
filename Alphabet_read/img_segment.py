import math

import cv2
import numpy as np

from PIL import Image
from canvas import pre_transform

def binarize(bgr: np.ndarray, threshold: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_components(binary: np.ndarray, min_area: int) -> tuple[np.ndarray, list]:
    '''
        หา components ที่มีในภาพว่ามีกี่ components และเก็บข้อมูลของแต่ละ component ไว้ใน list
    '''
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    components = []
    for label_id in range(1, num_labels):
        x    = stats[label_id, cv2.CC_STAT_LEFT]
        y    = stats[label_id, cv2.CC_STAT_TOP]
        w    = stats[label_id, cv2.CC_STAT_WIDTH]
        h    = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[label_id]
            current = [x, y, w, h, area, [label_id], cx, cy]
            components.append(current)

        print(f"Label ID: {label_id}, Area: {area}")

    return label_map, components

def is_solid(component: list, min_solidity: float) -> bool:
    """
        เช็คว่า pixel เติมเต็ม bbox แค่ไหน (solidity = pixel_area / bbox_area)
        component = [x, y, w, h, area, [label_id], cx, cy]
    """
    w, h, area = component[2], component[3], component[4]
    bbox_area = w * h
    if bbox_area <= 0:
        return False
    solidity = area / bbox_area
    return solidity >= min_solidity

def is_square(component: list, max_aspect_ratio: float) -> bool:
    """
        เช็คว่า bbox เป็น square หรือเกือบ square
        max_aspect_ratio = 1.3 -> ด้านยาวสุดห้ามเกิน 1.3 เท่าของด้านสั้นสุด
        component = [x, y, w, h, area, [label_id], cx, cy]
    """
    w, h = component[2], component[3]
    if w <= 0 or h <= 0:
        return False
    aspect = max(w, h) / min(w, h)
    return aspect <= max_aspect_ratio

def is_dot(
    component: list,
    absolute_area_cap: float = 1500.00,
    max_aspect_ratio: float = 1.3,
    min_solidity: float = 0.75,
) -> bool:
    """
    Component มีสิทธิ์เป็น dot ก็ต่อเมื่อผ่านทั้ง 3 เงื่อนไข:
      1. area <= absolute_area_cap
      2. boundary box เกือบเป็น square
      3. pixel เติมเต็ม bbox ไม่ใช่กลวง/โค้ง
    """
    area_ok     = component[4] <= absolute_area_cap
    shape_ok    = is_square(component, max_aspect_ratio)
    solidity_ok = is_solid(component, min_solidity)

    # Debugging output
    # print(f'area_ok: {area_ok}')
    # print(f'shape_ok: {shape_ok}')
    # print(f'solidity_ok: {solidity_ok}')
    return area_ok and shape_ok and solidity_ok

def bbox_gap(component1: list, component2: list) -> float:
    """ระยะห่างระหว่าง bbox 2 อัน"""
    x1, y1, w1, h1 = component1[0], component1[1], component1[2], component1[3]
    x2, y2, w2, h2 = component2[0], component2[1], component2[2], component2[3]
    dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
    dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)
    return math.hypot(dx, dy)

def merge_pair(comp: list, dot: list) -> list:
    """รวม dot เข้ากับ component หลัก คืน component ใหม่ (bbox union + area รวม + label_ids รวม)"""
    x1  = min(comp[0], dot[0])
    y1  = min(comp[1], dot[1])
    x2 = max(comp[0] + comp[2], dot[0] + dot[2])
    y2 = max(comp[1] + comp[3], dot[1] + dot[3])
    area      = comp[4] + dot[4]
    label_ids = comp[5] + dot[5]
    cx = (comp[6] * comp[4] + dot[6] * dot[4]) / area
    cy = (comp[7] * comp[4] + dot[7] * dot[4]) / area
    return [x1, y1, x2 - x1, y2 - y1, area, label_ids, cx, cy]

def merge_diacritics(
    components: list,
    max_distance: float,
    absolute_area_cap: float = 1500.00
) -> list:
    """
    แยก components เป็น dots / non-dots ด้วย is_dot()
    เช็ค gap ของแต่ละ dot กับทุก non-dot component
    ถ้า gap <= max_distance และ area ratio (dot/comp) อยู่ในช่วง [min_ratio, max_ratio]
    -> เป็น candidate ให้ merge, เลือก non-dot ที่ gap น้อยสุดเป็นคู่
    """
    """
    แยก components เป็น dots / non-dots ด้วย is_dot()
    เช็ค gap ของแต่ละ dot กับทุก non-dot component
    ถ้า gap <= max_distance -> เป็น candidate ให้ merge
    เลือก non-dot ที่มี bbox area น้อยสุดเป็นคู่
    """
    dots, non_dots = [], []
    for c in components:
        if is_dot(c, absolute_area_cap):
            dots.append(c)
            print(f'dots: {c[5]}')
        else:
            non_dots.append(c)
            print(f'non-dots: {c[5]}')

    if not dots or not non_dots:
        return components

    used_non_dots = set()
    matched_pairs = []  # (dot_idx, non_dot_idx)

    for di, dot in enumerate(dots):
        best_idx, best_bbox_area = None, None
        for ci, comp in enumerate(non_dots):
            if ci in used_non_dots:
                continue

            gap = bbox_gap(comp, dot)
            if gap > max_distance:
                continue

            bbox_area = comp[2] * comp[3]
            if best_bbox_area is None or bbox_area < best_bbox_area:
                best_idx, best_bbox_area = ci, bbox_area

        if best_idx is not None:
            used_non_dots.add(best_idx)
            matched_pairs.append((di, best_idx))

    used_dots = set()
    for di, ci in matched_pairs:
        non_dots[ci] = merge_pair(non_dots[ci], dots[di])
        used_dots.add(di)

    print('--------------------------------------------')

    combined = []
    combined_dot_nondot = non_dots + [d for i, d in enumerate(dots) if i not in used_dots]
    for i, component in enumerate(combined_dot_nondot):
        combined.append(component[5])

    print(f"component: {combined}")

    return combined_dot_nondot

def component_time(component, strokes: list, label_map: np.ndarray, tool_filter="pen") -> float:
    """
    หา timestamp เร็วสุดของ stroke ที่มี point ตกบน pixel ที่เป็นของ component นี้จริง ๆ
    เช็คผ่าน label_map (connected component id) ไม่ใช่ bbox -> กัน false-positive ตอน bbox overlap
    """
    label_ids = set(component[5])
    img_h, img_w = label_map.shape[:2]

    best_t = None
    for s in strokes:
        if s.get("tool", "pen") != tool_filter:
            continue
        for p in s["points"]:
            px, py = p.x(), p.y()
            if 0 <= px < img_w and 0 <= py < img_h and label_map[py, px] in label_ids:
                if best_t is None or s["t_start"] < best_t:
                    best_t = s["t_start"]
                break
    return best_t if best_t is not None else float("inf")


def group_by_time(components: list, strokes: list, label_map: np.ndarray) -> list:
    timed = [(component_time(c, strokes, label_map), c) for c in components]
    timed.sort(key=lambda x: x[0])
    return [c for _, c in timed]

# ---------------------------------------------------------------------------
# Row grouping: บนลงล่างก่อน แล้วซ้ายไปขวาในแต่ละแถว
# ---------------------------------------------------------------------------

def group_into_rows(components: list, row_thresh: int) -> list:
    """
    จัดกลุ่มตัวอักษรเป็นแถวจาก cy (vertical center)
    component ที่ cy ห่างจาก running average ของแถวไม่เกิน row_thresh
    ถือว่าอยู่แถวเดียวกัน จากนั้นเรียงแถวบน->ล่าง และในแถวเดียวกันเรียงซ้าย->ขวา
    """
    if not components:
        return []

    sorted_by_y = sorted(components, key=lambda c: c[7])  # cy

    rows = []
    for comp in sorted_by_y:
        cy = comp[7]
        placed = False
        for row in rows:
            if abs(cy - row["avg_cy"]) <= row_thresh:
                row["members"].append(comp)
                row["avg_cy"] = float(np.mean([m[7] for m in row["members"]]))
                placed = True
                break
        if not placed:
            rows.append({"avg_cy": cy, "members": [comp]})

    rows.sort(key=lambda r: r["avg_cy"])

    ordered = []
    for row in rows:
        row["members"].sort(key=lambda c: c[0])  # x ซ้าย->ขวา
        ordered.extend(row["members"])
    return ordered

# ---------------------------------------------------------------------------
# Crop / clean / pad
# ---------------------------------------------------------------------------

def crop_and_clean(bgr, label_map, component, pad):
    x, y, w, h, _, label_ids, _, _ = component
    img_h, img_w = bgr.shape[:2]
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, img_w), min(y + h + pad, img_h)
    crop_bgr    = bgr[y0:y1, x0:x1].copy()
    crop_labels = label_map[y0:y1, x0:x1]
    belongs_to_char = np.isin(crop_labels, label_ids)
    crop_bgr[~belongs_to_char] = [255, 255, 255]
    return crop_bgr

def pad_to_square(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    side = max(h, w)
    square = np.full((side, side, 3), 255, dtype=np.uint8)
    square[(side - h) // 2:(side - h) // 2 + h,
           (side - w) // 2:(side - w) // 2 + w] = crop_bgr
    return square

def compute_angle(square_bgr: np.ndarray, min_nu02: float = 1e-3, max_skew: float = 1.0) -> float:
    """
    คำนวณมุม slant (องศา) ของตัวอักษรจาก 2nd-order image moments
    ใช้ shear ratio (mu11/mu02) แปลงเป็นองศาด้วย atan() เพื่อ apply เป็น true rotation

    ทำไมไม่ใช้ atan2(2*mu11, mu20-mu02):
      - สูตรนั้นให้ "มุมแกนหลัก" (principal axis) = 90° สำหรับเส้นตรงแนวตั้ง
        -> clamp เป็น max_angle ทันที (ผิด)
    ทำไมใช้ mu11/mu02:
      - mu11/mu02 คือ shear ratio ตาม axis แนวตั้ง
      - เส้นตรงแนวตั้ง: mu11 ≈ 0 -> skew ≈ 0 -> angle ≈ 0° (ถูกต้อง)
      - ตัวอักษรเอียง: skew = ค่าบวก/ลบ -> atan(skew) = มุมแก้ slant ที่ถูกต้อง

    min_nu02 : ถ้า nu02 ต่ำมาก (ตัวอักษรแบนราบ เช่น "-", ".") ถือว่าไม่มี slant
    max_skew : clamp shear ratio ก่อน atan (1.0 ~ 45°, ครอบคลุมลายมือเอียงสุด)
    """
    gray = cv2.cvtColor(square_bgr, cv2.COLOR_BGR2GRAY)
    ink  = 255 - gray   # invert: เส้นหมึก (pixel ต่ำเดิม) -> ค่าสูง = มวลของ moment

    m = cv2.moments(ink)
    if m['m00'] == 0 or abs(m['nu02']) < min_nu02:
        return 0.0

    skew = m['mu11'] / m['mu02']
    skew = max(-max_skew, min(max_skew, skew))   # clamp shear ratio
    return math.degrees(math.atan(skew))          # แปลงเป็นองศาหมุนจริง


def deskew(square_bgr: np.ndarray, min_nu02: float = 1e-3, max_skew: float = 1.0) -> np.ndarray:
    """
    แก้ orientation ของตัวอักษรเดี่ยวด้วย true rotation (ไม่ใช่ shear)
    ใช้ cv2.getRotationMatrix2D() สร้าง rotation matrix จริง กัน distortion รูปทรง

    ทำงานกับ crop ที่ pad เป็นสี่เหลี่ยมจัตุรัสแล้วเท่านั้น (input/output shape เท่ากัน)
    max_skew : ส่งต่อให้ compute_angle() เป็น clamp ของ shear ratio (1.0 ~ 45°)
    """
    angle = compute_angle(square_bgr, min_nu02, max_skew)
    if abs(angle) < 3.0:
        return square_bgr.copy()

    h, w = square_bgr.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -(angle), scale=1.0)
    rotated = cv2.warpAffine(
        square_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    return rotated



# ---------------------------------------------------------------------------
# Debug visualization: แสดงผลแยกทีละ step ของ pipeline
# ---------------------------------------------------------------------------

def _tile_grid(images: list, cols: int = 6, cell: int = 130, labels: list | None = None) -> np.ndarray:
    """เรียง crop หลายรูปเป็น grid รูปเดียว สำหรับ debug
    labels: ถ้าให้มา จะแทนที่ label "#i" ด้วยข้อความนี้ต่อรูป (เช่น ค่ามุม deskew)"""
    if not images:
        return np.full((cell, cell, 3), 255, dtype=np.uint8)

    rows = (len(images) + cols - 1) // cols
    grid = np.full((rows * cell, cols * cell, 3), 220, dtype=np.uint8)

    for i, img in enumerate(images):
        h, w = img.shape[:2]
        scale  = (cell - 10) / max(h, w)
        resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        r, c = divmod(i, cols)
        y0 = r * cell + (cell - resized.shape[0]) // 2
        x0 = c * cell + (cell - resized.shape[1]) // 2
        grid[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        tag = labels[i] if labels else f"#{i}"
        cv2.putText(grid, tag, (c * cell + 4, r * cell + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)

    return grid

def debug_pipeline(
    bgr: np.ndarray,
    binary: np.ndarray,
    label_map: np.ndarray,
    components: list,
    crop_pad: int,
    output_prefix: str = "debug",
) -> dict:
    """
    บันทึกภาพ debug แยกทีละ step (ยังไม่ deskew):
      step1_binary      : ผล threshold (binarize)
      step2_components  : bounding box + ลำดับ component หลัง merge_diacritics + group_into_rows
      step3_crops       : clean crop + pad เป็นสี่เหลี่ยมจัตุรัส ต่อตัว (input ที่จะเข้าโมเดล)
    คืนค่า dict ของภาพทั้งหมด และเซฟไฟล์ {output_prefix}_stepN_*.png
    """
    outputs = {}

    # step 1: binary threshold
    outputs["step1_binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # step 2: bounding box + ลำดับหลัง merge + group_into_rows
    step2 = bgr.copy()
    for i, component in enumerate(components):
        x, y, w, h, _, _, _, _ = component
        cv2.rectangle(step2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(step2, f"#{i}", (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    outputs["step2_components"] = step2

    # step 3: crop + pad ต่อ component (ยังไม่ deskew)
    crops = []
    for component in components:
        clean_crop  = crop_and_clean(bgr, label_map, component, pad=crop_pad)
        square_crop = pad_to_square(clean_crop)
        crops.append(square_crop)
    outputs["step3_crops"] = _tile_grid(crops)

    # step 4: crop เดียวกัน หลังผ่าน deskew (moment-based shear correction)
    deskewed_crops = [deskew(c) for c in crops]

    angle_labels = []
    for crop in crops:
        angle_deg = compute_angle(crop)
        angle_labels.append(f"{angle_deg:+.1f}")
        print(f"deskew angle: {angle_deg:+.1f} deg")

    outputs["step4_deskewed"] = _tile_grid(deskewed_crops, labels=angle_labels)

    for name, img in outputs.items():
        cv2.imwrite(f"{output_prefix}_{name}.png", img)

    return outputs

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_chars(
    bgr: np.ndarray,
    strokes: list | None = None,
    order_mode: str = "time",   # "spatial" | "time"
    min_area: int         = 120,
    max_distance: float    = 18.0,
    crop_pad: int         = 20,
    row_thresh: int       = 100,
    debug: bool           = False,
) -> list:
    binary = binarize(bgr)
    print("--------------------------------------------")
    label_map, components = find_components(binary, min_area)
    print("--------------------------------------------")
    if not components:
        return []

    components = merge_diacritics(components, max_distance)

    if order_mode == "time" and strokes:
        components = group_by_time(components, strokes, label_map)
    else:
        components = group_into_rows(components, row_thresh)

    if debug:
        debug_pipeline(bgr, binary, label_map, components, crop_pad)

    tensors = []
    for i, component in enumerate(components):
        clean_crop  = crop_and_clean(bgr, label_map, component, pad=crop_pad)
        square_crop = pad_to_square(clean_crop)
        square_crop = deskew(square_crop)
        pil_image   = Image.fromarray(cv2.cvtColor(square_crop, cv2.COLOR_BGR2RGB))
        tensor      = pre_transform(pil_image).unsqueeze(0)
        tensors.append(tensor)

    return tensors
