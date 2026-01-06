import os
import torch
import tempfile
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3Model

INPUT_FILE = "D:\\JCB_use_case\\data\\invoices\\invoice_10_P.jpg"  
OUTPUT_TXT = "D:\\JCB_use_case\\data\\txt_files\\invoice_10_P.txt"


ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
)

# =========================
# INIT LAYOUTLM
# =========================
processor = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base",
    apply_ocr=False
)
model = LayoutLMv3Model.from_pretrained(
    "microsoft/layoutlmv3-base"
)
model.eval()

# =========================
# LOAD IMAGE(S)
# =========================
def load_images(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return convert_from_path(path, dpi=300)
    else:
        return [Image.open(path).convert("RGB")]


def ocr_to_words_boxes(image):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    result = ocr.predict(image_path)
    os.remove(image_path)

    words, boxes = [], []
    w, h = image.size

    if not result or not result[0]:
        return words, boxes

    for line in result[0]:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue

        bbox, text_info = line

        # Validate bbox
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in bbox)
        ):
            continue

        # Validate text
        if not isinstance(text_info, (list, tuple)) or len(text_info) < 1:
            continue

        text = text_info[0].strip()
        if not text:
            continue

        x = [p[0] for p in bbox]
        y = [p[1] for p in bbox]

        box = [
            int(1000 * min(x) / w),
            int(1000 * min(y) / h),
            int(1000 * max(x) / w),
            int(1000 * max(y) / h),
        ]

        words.append(text)
        boxes.append(box)

    return words, boxes

final_text = []
images = load_images(INPUT_FILE)

for page_no, image in enumerate(images, 1):
    words, boxes = ocr_to_words_boxes(image)

    if not words:
        continue

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        _ = model(**encoding)

    final_text.append(f"\n===== PAGE {page_no} =====\n")
    final_text.extend(words)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(final_text))

print("✅ RAW TEXT extracted successfully")
print("📄 Output file:", OUTPUT_TXT)
print("🧾 Characters written:", len("\n".join(final_text)))
