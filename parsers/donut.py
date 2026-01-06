"""
FINAL INVOICE PARSER
PDF / Image → DONUT → Gemini GST Normalizer → Final JSON

✔ No AutoModelForCausalLM
✔ No accelerate / offload_dir
✔ No local LLM
✔ Safe on CPU / CUDA
"""

import os
import json
import torch
import warnings
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from transformers import DonutProcessor, VisionEncoderDecoderModel
from google import genai

# =========================
# CONFIG
# =========================
MODEL_NAME = "naver-clova-ix/donut-base-finetuned-cord-v2"

INPUT_DIR = r"D:\JCB_use_case\data\invoices"
OUTPUT_DIR = r"D:\JCB_use_case\data\txt_files"

DPI = 300
IMAGE_SIZE = 1280
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# SILENCE LoRA META WARNINGS
# =========================
warnings.filterwarnings(
    "ignore",
    message="copying from a non-meta parameter"
)

# =========================
# LOAD DONUT MODEL
# =========================
processor = DonutProcessor.from_pretrained(
    MODEL_NAME,
    use_fast=False
)

model = VisionEncoderDecoderModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=False
)
model.to(DEVICE)
model.eval()

# =========================
# LOAD GEMINI CLIENT
# =========================
gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

GST_PROMPT = """
You are a strict Indian GST invoice normalization engine.

RULES:
- Use ONLY information present in the input
- Do NOT guess, infer, or fabricate
- If a value is missing or unclear, use null
- Keep numeric values as numbers
- Dates must be YYYY-MM-DD
- Output MUST be valid JSON only
- No explanations, no markdown

TARGET SCHEMA:
{
  "invoice_number": string | null,
  "invoice_date": string | null,
  "vendor_name": string | null,
  "vendor_gstin": string | null,
  "buyer_name": string | null,
  "buyer_gstin": string | null,
  "subtotal": number | null,
  "cgst": number | null,
  "sgst": number | null,
  "igst": number | null,
  "total_tax": number | null,
  "total_amount": number | null,
  "items": [
    {
      "description": string | null,
      "quantity": number | null,
      "unit_price": number | null,
      "amount": number | null
    }
  ]
}

Return JSON only.
"""

# =========================
# HELPERS
# =========================
def load_images(path: str):
    path = path.lower()
    if path.endswith(".pdf"):
        return convert_from_path(path, dpi=DPI)
    elif path.endswith((".jpg", ".jpeg", ".png")):
        return [Image.open(path).convert("RGB")]
    return []

def parse_with_donut(image: Image.Image) -> dict:
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    decoder_input_ids = processor.tokenizer(
        "<s_cord-v2>",
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.to(DEVICE)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=768,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        do_sample=False
    )

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    if not decoded:
        return {}

    return processor.token2json(decoded)

def merge_pages(pages: list) -> dict:
    merged = {}
    for page in pages:
        for k, v in page.items():
            if k not in merged:
                merged[k] = v
            elif isinstance(v, list):
                merged[k].extend(v)
    return merged

def normalize_with_gemini(donut_json: dict) -> dict:
    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            GST_PROMPT,
            json.dumps(donut_json, ensure_ascii=False)
        ],
        generation_config={"temperature": 0}
    )
    return json.loads(response.text)

# =========================
# MAIN
# =========================
def main():
    files = sorted([
        f for f in Path(INPUT_DIR).iterdir()
        if f.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png"]
    ])

    if not files:
        print("❌ No invoice files found")
        return

    for file_path in files:
        print(f"\n📄 Processing: {file_path.name}")

        images = load_images(str(file_path))
        if not images:
            print("  ⏭ Skipped")
            continue

        page_results = []
        for idx, img in enumerate(images):
            print(f"  → Page {idx + 1}")
            page_results.append(parse_with_donut(img))

        raw_invoice = merge_pages(page_results)
        final_invoice = normalize_with_gemini(raw_invoice)

        output_path = Path(OUTPUT_DIR) / f"{file_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_invoice, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Saved: {output_path}")

if __name__ == "__main__":
    main()
