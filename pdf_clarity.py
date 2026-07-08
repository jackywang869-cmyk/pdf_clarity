#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_clarity.py — 让扫描版 PDF 变清晰的工具

原理
----
逐页处理一份 PDF：
1. 先判断该页是不是"原生文字页"（PDF 里本来就带可提取文字/矢量内容，不是扫描图片）。
   这类页面本身就是清晰的，直接原样保留，不做任何"增强"，避免画质变差。
2. 对"扫描图片页"，用 OCR 置信度 + 图像统计（饱和度、直方图双峰性、边缘密度等）
   自动判断这一页更像"文字扫描件"还是"照片/图片"：
   - 文字扫描件：去噪 → 自动纠偏（deskew）→ CLAHE 对比度增强 → 锐化 →
     （可选）Otsu 二值化，让文字边缘锐利、背景干净。
   - 照片/图片：彩色去噪 → LAB 空间 CLAHE 提升层次 → USM 锐化 → 轻微饱和度提升，
     不做纠偏和二值化，保留照片的自然色彩层次。
3. 可选：用 Tesseract 给处理后的每一页重新生成一层"看不见的可搜索文字层"（--ocr），
   这样清晰化之后的 PDF 依然可以被复制/搜索。
4. 把所有页面（原生页 + 处理后的图片页）按原始顺序重新拼接成一份新 PDF，
   页面物理尺寸与原文档保持一致。

依赖
----
pip install pypdf pdfplumber pypdfium2 opencv-python-headless numpy pillow img2pdf pytesseract
并需要系统安装 tesseract-ocr（--ocr 时才需要）。

用法
----
    python3 pdf_clarity.py input.pdf -o output.pdf
    python3 pdf_clarity.py input.pdf -o output.pdf --dpi 400 --ocr --lang eng
    python3 pdf_clarity.py input.pdf -o output.pdf --mode text          # 强制全部按文字页处理
    python3 pdf_clarity.py input.pdf -o output.pdf --binarize           # 文字页做纯黑白二值化
"""

import argparse
import io
import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
import pdfplumber
import img2pdf

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


# --------------------------------------------------------------------------
# 第 1 步：判断某一页是"原生文字页"还是"扫描图片页"
# --------------------------------------------------------------------------

def is_native_text_page(pdfplumber_page, image_area_ratio_threshold=0.85):
    """
    判断该页是否是"原生"页面（本身带可提取文字、不是整页大图扫描件）。
    规则：
      - 页面里能提取到有意义的文字（长度 > 20 个字符），并且
      - 页面中"图片"覆盖的面积占比没有大到接近整页（说明不是整页扫描图 + 隐藏文字层）。
    满足以上条件则认为是原生页面，直接保留、不做图像增强。
    """
    try:
        text = pdfplumber_page.extract_text() or ""
    except Exception:
        text = ""

    page_area = float(pdfplumber_page.width) * float(pdfplumber_page.height)
    image_area = 0.0
    for img in pdfplumber_page.images:
        w = max(0.0, float(img.get("x1", 0)) - float(img.get("x0", 0)))
        h = max(0.0, float(img.get("bottom", 0)) - float(img.get("top", 0)))
        image_area += w * h

    image_ratio = (image_area / page_area) if page_area > 0 else 0.0

    has_real_text = len(text.strip()) > 20
    is_full_page_scan_image = image_ratio >= image_area_ratio_threshold

    return has_real_text and not is_full_page_scan_image


# --------------------------------------------------------------------------
# 第 2 步：渲染页面为高分辨率图像
# --------------------------------------------------------------------------

def render_page_to_bgr(pdf_doc, page_index, dpi):
    """用 pypdfium2 把某一页渲染成 OpenCV 格式（BGR）的 numpy 数组。"""
    page = pdf_doc[page_index]
    scale = dpi / 72.0  # pdfium 的 1.0 缩放对应 72 dpi
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil().convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return bgr


# --------------------------------------------------------------------------
# 第 3 步：自动分类 —— 文字扫描件 vs 照片/图片
# --------------------------------------------------------------------------

def classify_scanned_page(bgr_image, fast=False):
    """
    返回 'text' 或 'photo'。
    先算"免费"的图像统计特征（饱和度、直方图双峰性、边缘密度），
    只有在统计特征给出的信号模棱两可时，才动用较慢的 OCR 置信度做二次确认。
    这样绝大多数页面（信号明确的）可以完全跳过 OCR，速度快很多。
    fast=True 时无论如何都不跑 OCR，只用图像统计判断（最快，牺牲一点点准确率）。
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # --- 特征 A：颜色饱和度。文字扫描件基本是灰阶/近似黑白，照片饱和度更高 ---
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    # --- 特征 B：灰度直方图的双峰性。文字页 = 大片白底 + 少量深色文字 ---
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-6)
    bright_ratio = float(hist_norm[200:].sum())   # 接近白色的像素占比
    mid_ratio = float(hist_norm[80:200].sum())    # 中间调占比（照片中间调通常较多）

    # --- 特征 C：边缘密度 ---
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(np.mean(edges > 0))

    text_like_score = 0
    if mean_saturation < 25:
        text_like_score += 1
    if bright_ratio > 0.45 and mid_ratio < 0.35:
        text_like_score += 1
    if 0.02 < edge_density < 0.25:
        text_like_score += 1

    # 统计特征已经给出明确信号（0 或 3）时，直接采用，跳过较慢的 OCR
    if fast or text_like_score in (0, 3) or not HAS_TESSERACT:
        return "text" if text_like_score >= 2 else "photo"

    # 信号模棱两可（1 或 2）时，才用 OCR 置信度做二次确认
    try:
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        data = pytesseract.image_to_data(
            small, output_type=pytesseract.Output.DICT,
            config="--psm 3"
        )
        confidences = [float(c) for c in data.get("conf", []) if c not in ("-1", -1)]
        word_count = sum(1 for w in data.get("text", []) if w.strip())
        if confidences and word_count >= 5:
            mean_conf = np.mean(confidences)
            if mean_conf >= 55 and word_count >= 8:
                return "text"
    except Exception:
        pass

    return "text" if text_like_score >= 2 else "photo"


# --------------------------------------------------------------------------
# 第 4 步 a：文字扫描件增强流程
# --------------------------------------------------------------------------

def deskew_image(gray_image):
    """通过文本区域的最小外接矩形估计倾斜角度并纠正。"""
    inverted = cv2.bitwise_not(gray_image)
    _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return gray_image, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # 角度过大通常是误判（比如整页几乎无文字），忽略避免把页面转歪
    if abs(angle) < 0.1 or abs(angle) > 15:
        return gray_image, 0.0

    (h, w) = gray_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_image, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def enhance_text_page(bgr_image, binarize=False, strength="normal", keep_color=True, fast=False, deskew=True):
    """
    文字扫描件增强：去噪 -> 纠偏 -> CLAHE 对比度增强 -> 锐化 -> (可选)二值化。

    keep_color=True（默认）时全程保留原始颜色：只在 LAB 色彩空间的亮度(L)
    通道上做去噪/对比度/锐化，色彩(a,b)通道原样保留，纠偏时对整张彩色图一起旋转。
    这样即使页面被判定为"文字页"，页面里嵌的彩色图案、颜色标注、高亮等也不会丢色。

    只有 binarize=True（用户显式要求纯黑白）时才会转灰阶再二值化，
    因为二值化这个操作本身就是"只保留黑/白"，无法保留颜色。

    fast=True 时用更小的去噪搜索窗口（searchWindowSize 21->9），速度明显更快，
    画质略有下降但通常察觉不出来，适合页数很多、追求速度的场景。

    deskew=False 时跳过内部纠偏步骤——用于"同页混合处理"流程：纠偏已经在
    更上层对整张页面统一做过一次，这里就不用再重复纠偏（否则文字区域和
    照片区域会各自转出不同角度，拼接时对不上）。
    """
    search_window = 9 if fast else 21
    template_window = 7

    if binarize or not keep_color:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(
            gray, h=10, templateWindowSize=template_window, searchWindowSize=search_window
        )
        if deskew:
            deskewed, _angle = deskew_image(denoised)
        else:
            deskewed = denoised
        clip_limit = 3.0 if strength == "strong" else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        contrasted = clahe.apply(deskewed)
        blurred = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(contrasted, 1.5, blurred, -0.5, 0)

        if binarize:
            result = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=31, C=10
            )
            result = cv2.medianBlur(result, 3)
        else:
            result = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # --- 保留颜色的文字增强流程 ---
    if deskew:
        gray_for_skew = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        denoised_gray = cv2.fastNlMeansDenoising(
            gray_for_skew, h=10, templateWindowSize=template_window, searchWindowSize=search_window
        )
        _, angle = deskew_image(denoised_gray)

        if abs(angle) > 0:
            (h, w) = bgr_image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            bgr_image = cv2.warpAffine(
                bgr_image, matrix, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

    # 彩色去噪（保留色彩通道细节）
    h_luma = 7 if strength == "strong" else 5
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr_image, h=h_luma, hColor=h_luma,
        templateWindowSize=template_window, searchWindowSize=search_window
    )

    # 只在亮度通道上做对比度增强，颜色通道不动
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clip_limit = 3.0 if strength == "strong" else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    contrasted = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 锐化（Unsharp Mask）
    blurred = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(contrasted, 1.5, blurred, -0.5, 0)

    return sharpened


# --------------------------------------------------------------------------
# 第 4 步 b：照片/图片页增强流程
# --------------------------------------------------------------------------

def enhance_photo_page(bgr_image, strength="normal", fast=False):
    """
    照片页增强：彩色去噪 -> LAB 空间 CLAHE 提亮层次 -> USM 锐化 -> 轻微饱和度提升。
    不做纠偏、不做二值化，保留照片的自然色彩与渐变。
    """
    search_window = 9 if fast else 21
    h_luma = 7 if strength == "strong" else 5
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr_image, h=h_luma, hColor=h_luma,
        templateWindowSize=7, searchWindowSize=search_window
    )

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    contrasted = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # USM 锐化
    blurred = cv2.GaussianBlur(contrasted, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(contrasted, 1.3, blurred, -0.3, 0)

    # 轻微饱和度提升
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.12, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


# --------------------------------------------------------------------------
# 第 4 步 c：同页混合处理 —— 检测页面里的"照片/插图"区域，与文字区域分开处理
# --------------------------------------------------------------------------

def detect_photo_regions(bgr_image, min_area_ratio=0.02, sat_threshold=26, min_fill_ratio=0.42):
    """
    在一张以文字为主的页面图像里，找出"像照片/彩色插图"的矩形区域。

    原理：
    1. 先算局部平均饱和度，用形态学操作把饱和度高的像素合并成完整色块，
       得到候选矩形框（这一步文字标题里的彩色字、logo 也可能被误圈进来，
       因为它们经过较大范围的模糊/闭运算后看起来也是一整块）。
    2. 关键过滤：对每个候选框，在"轻微模糊但不做大范围闭运算"的原始饱和度图上，
       计算框内实际饱和像素的填充率。真正的照片/插图是连续色块，填充率高
       （实测约 0.5-0.9）；彩色文字标题只是笔画着色、大部分区域仍是背景底色，
       填充率低（实测约 0.2-0.3）。以此排除"一整行彩色文字"被误判成插图。

    返回：[(x0, y0, x1, y1), ...] 像素坐标的矩形框列表（已加了一点边距）。
    """
    h, w = bgr_image.shape[:2]
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)

    box_k = max(15, (int(min(h, w) * 0.03) | 1))
    local_sat = cv2.boxFilter(sat, ddepth=-1, ksize=(box_k, box_k))
    mask = (local_sat > sat_threshold).astype(np.uint8) * 255

    close_k = max(21, (int(min(h, w) * 0.025) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 用于填充率检验的"原始"饱和度掩码：只做轻微模糊去噪，不做大范围闭运算
    raw_k = max(5, (int(min(h, w) * 0.006) | 1))
    raw_sat_blur = cv2.boxFilter(sat, ddepth=-1, ksize=(raw_k, raw_k))
    raw_mask = raw_sat_blur > (sat_threshold + 10)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = h * w
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * page_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)

        fill_ratio = float(raw_mask[y:y + bh, x:x + bw].mean())
        if fill_ratio < min_fill_ratio:
            continue  # 大概率是彩色文字/标题被误圈，不是真正的插图

        pad_x = int(bw * 0.02) + 6
        pad_y = int(bh * 0.02) + 6
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
        boxes.append((x0, y0, x1, y1))
    return boxes


def enhance_mixed_page(bgr_image, strength="normal", fast=False):
    """
    同页混合处理：整页先统一纠偏一次，然后检测出"照片/插图"区域，
    该区域用照片流程处理（去噪+提层次+锐化+加饱和度，保留自然色彩），
    其余文字背景区域用文字流程处理（去噪+CLAHE+锐化，颜色保留但更锐利），
    最后用羽化过的蒙版把两者无缝混合在一起。

    返回 (处理后的图像, 检测到的照片区域列表)。
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    search_window = 9 if fast else 21
    denoised_gray = cv2.fastNlMeansDenoising(
        gray, h=10, templateWindowSize=7, searchWindowSize=search_window
    )
    _, angle = deskew_image(denoised_gray)
    if abs(angle) > 0:
        (h, w) = bgr_image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        bgr_image = cv2.warpAffine(
            bgr_image, matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    boxes = detect_photo_regions(bgr_image)
    h, w = bgr_image.shape[:2]
    page_area = h * w

    if not boxes:
        # 没检测到独立插图区域，整页按文字流程处理（已经纠偏过，跳过内部再纠偏）
        result = enhance_text_page(
            bgr_image, binarize=False, strength=strength,
            keep_color=True, fast=fast, deskew=False
        )
        return result, []

    covered_ratio = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes) / page_area
    if covered_ratio > 0.85:
        # 插图几乎占满整页，直接按照片页处理即可
        result = enhance_photo_page(bgr_image, strength=strength, fast=fast)
        return result, boxes

    text_result = enhance_text_page(
        bgr_image, binarize=False, strength=strength,
        keep_color=True, fast=fast, deskew=False
    )
    photo_result = enhance_photo_page(bgr_image, strength=strength, fast=fast)

    mask = np.zeros((h, w), dtype=np.float32)
    for (x0, y0, x1, y1) in boxes:
        mask[y0:y1, x0:x1] = 1.0
    feather = max(9, int(min(h, w) * 0.015) | 1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather)
    mask3 = mask[:, :, np.newaxis]

    blended = (
        photo_result.astype(np.float32) * mask3
        + text_result.astype(np.float32) * (1.0 - mask3)
    )
    return blended.astype(np.uint8), boxes


# --------------------------------------------------------------------------
# 第 5 步：把处理后的图像页打包成 PDF 页（可选带 OCR 文字层）
# --------------------------------------------------------------------------

def image_page_to_pdf_bytes(bgr_image, dpi, ocr=False, lang="eng"):
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    pil_image.info["dpi"] = (dpi, dpi)

    if ocr and HAS_TESSERACT:
        try:
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                pil_image, extension="pdf", lang=lang,
                config=f"--dpi {dpi}"
            )
            return bytes(pdf_bytes)
        except Exception as exc:
            print(f"  ! OCR 生成文字层失败，改用无文字层图片页：{exc}", file=sys.stderr)

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    layout_fun = img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))
    pdf_bytes = img2pdf.convert(buf.read(), layout_fun=layout_fun)
    return pdf_bytes


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 主流程（支持多进程并行处理页面，加速多页文档）
# --------------------------------------------------------------------------

_worker_state = {}


def _worker_init(input_path):
    """每个子进程启动时执行一次：打开自己独立的 pdfium 文档句柄。
    pdfium 的文档对象不能跨进程共享，所以每个 worker 进程各开一份。"""
    _worker_state["pdf_doc"] = pdfium.PdfDocument(input_path)


def _process_one_page(args):
    """
    在子进程里处理单独一页扫描图片页：渲染 -> 分类 -> 增强 -> 打包成单页 PDF 字节。
    返回 (page_index, page_type, pdf_bytes)，由主进程按顺序拼接。
    """
    (page_index, dpi, mode, binarize, strength, ocr, lang,
     grayscale, fast, mixed) = args
    pdf_doc = _worker_state["pdf_doc"]

    bgr = render_page_to_bgr(pdf_doc, page_index, dpi)

    if mode == "text":
        page_type = "text"
    elif mode == "photo":
        page_type = "photo"
    else:
        page_type = classify_scanned_page(bgr, fast=fast)

    # 同页混合处理：只有在"文字页"、不要求二值化/强制灰阶时才启用
    # （二值化/灰阶是用户明确要"整页统一效果"，此时不做区域拆分）
    use_mixed = mixed and page_type == "text" and not binarize and not grayscale

    if use_mixed:
        processed, boxes = enhance_mixed_page(bgr, strength=strength, fast=fast)
        if boxes:
            page_type = f"text+{len(boxes)}处插图"
    elif page_type == "text":
        processed = enhance_text_page(
            bgr, binarize=binarize, strength=strength,
            keep_color=not grayscale, fast=fast
        )
    else:
        processed = enhance_photo_page(bgr, strength=strength, fast=fast)

    page_pdf_bytes = image_page_to_pdf_bytes(processed, dpi, ocr=ocr, lang=lang)
    return page_index, page_type, page_pdf_bytes


def process_pdf(input_path, output_path, dpi=300, mode="auto",
                 binarize=False, strength="normal", ocr=False, lang="eng",
                 grayscale=False, fast=False, workers=None, mixed=True, verbose=True):
    reader = PdfReader(input_path)
    num_pages = len(reader.pages)
    writer = PdfWriter()

    # --- 第一步（快速、单进程）：判断哪些页是原生文字页，不需要再处理 ---
    native_flags = [False] * num_pages
    if mode == "auto":
        with pdfplumber.open(input_path) as plumber_pdf:
            for i in range(num_pages):
                try:
                    native_flags[i] = is_native_text_page(plumber_pdf.pages[i])
                except Exception:
                    native_flags[i] = False

    pages_to_process = [i for i in range(num_pages) if not native_flags[i]]

    if verbose:
        print(f"共 {num_pages} 页：{num_pages - len(pages_to_process)} 页原生文字页（跳过），"
              f"{len(pages_to_process)} 页需要增强处理")

    results = {}

    if pages_to_process:
        if workers is None:
            workers = max(1, min(len(pages_to_process), os.cpu_count() or 1))

        tasks = [
            (i, dpi, mode, binarize, strength, ocr, lang, grayscale, fast, mixed)
            for i in pages_to_process
        ]

        if workers <= 1:
            _worker_init(input_path)
            done = 0
            for task in tasks:
                idx, page_type, pdf_bytes = _process_one_page(task)
                results[idx] = pdf_bytes
                done += 1
                if verbose:
                    print(f"[{done}/{len(tasks)}] 第 {idx + 1} 页判定为「{page_type}」，已完成增强")
        else:
            if verbose:
                print(f"使用 {workers} 个进程并行处理...")
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(input_path,)) as pool:
                done = 0
                for idx, page_type, pdf_bytes in pool.imap(_process_one_page, tasks):
                    results[idx] = pdf_bytes
                    done += 1
                    if verbose:
                        print(f"[{done}/{len(tasks)}] 第 {idx + 1} 页判定为「{page_type}」，已完成增强")

    # --- 按原始页码顺序拼接最终 PDF ---
    for i in range(num_pages):
        if native_flags[i]:
            writer.add_page(reader.pages[i])
        else:
            page_reader = PdfReader(io.BytesIO(results[i]))
            writer.add_page(page_reader.pages[0])

    with open(output_path, "wb") as f:
        writer.write(f)

    if verbose:
        print(f"\n完成！已保存到: {output_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="让扫描版 PDF 变清晰的工具：按每页内容（文字扫描件 / 照片图片）自动采取不同增强手段。"
    )
    parser.add_argument("input", help="输入 PDF 路径")
    parser.add_argument("-o", "--output", help="输出 PDF 路径（默认在原文件名后加 _clear）")
    parser.add_argument("--dpi", type=int, default=300, help="渲染分辨率，默认 300；文字很小可用 400-600")
    parser.add_argument(
        "--mode", choices=["auto", "text", "photo"], default="auto",
        help="auto=逐页自动判断（默认）；text=全部按文字扫描件处理；photo=全部按照片处理"
    )
    parser.add_argument("--binarize", action="store_true", help="文字页做黑白二值化（更锐利，但会丢失灰阶/颜色信息）")
    parser.add_argument("--grayscale", action="store_true", help="文字页转灰阶但不二值化（默认关闭：默认保留颜色）")
    parser.add_argument("--strength", choices=["normal", "strong"], default="normal", help="增强强度")
    parser.add_argument("--ocr", action="store_true", help="为处理后的扫描页重新生成可搜索文字层（需要系统装有 tesseract）")
    parser.add_argument("--lang", default="eng", help="OCR 语言，如 eng / chi_sim / chi_sim+eng（需已安装对应语言包）")
    parser.add_argument("--workers", type=int, default=None,
                         help="并行处理的进程数，默认按 CPU 核心数自动设置。设为 1 则单进程顺序处理（便于调试）")
    parser.add_argument("--fast", action="store_true",
                         help="加速模式：分类时跳过 OCR 二次确认、去噪用更小的搜索窗口。"
                              "画质略降但通常不明显，页数很多时建议开启")
    parser.add_argument("--no-mixed", action="store_true",
                         help="关闭同页混合处理。默认会在文字页里自动检测彩色插图区域，"
                              "插图用照片流程、其余文字用文字流程分开处理再拼接；"
                              "关闭后整页只用一种流程（旧行为）")
    parser.add_argument("--quiet", action="store_true", help="不打印处理进度")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"找不到输入文件: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_clear.pdf"

    if args.ocr and not HAS_TESSERACT:
        print("警告：未安装 pytesseract，--ocr 选项将被忽略。", file=sys.stderr)

    process_pdf(
        args.input, output_path,
        dpi=args.dpi, mode=args.mode,
        binarize=args.binarize, strength=args.strength,
        ocr=args.ocr, lang=args.lang,
        grayscale=args.grayscale,
        fast=args.fast, workers=args.workers,
        mixed=not args.no_mixed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
