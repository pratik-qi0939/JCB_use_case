import json
import os
import torch
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_MODEL = "manaspros/indian-receipt-parser-v2"

INPUT_FILE = "D:\JCB_use_case\data\invoices\invoice_9_P.pdf"     # .pdf / .png / .jpg
DPI = 300
MAX_NEW_TOKENS = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TESSERACT CONFIG (Windows users: adjust if needed)
# ============================================================
# Example:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ============================================================
# OCR FUNCTIONS
# ============================================================
def ocr_image(image: Image.Image) -> str:
    """
    Extract raw text from a PIL image using Tesseract
    """
    custom_config = r"--oem 3 --psm 6"
    return pytesseract.image_to_string(image, config=custom_config)


def extract_text(path: str) -> str:
    """
    Extract text from PDF or image
    """
    if path.lower().endswith(".pdf"):
        images = convert_from_path(path, dpi=DPI)
        pages_text = []
        for img in images:
            pages_text.append(ocr_image(img))
        return "\n".join(pages_text)
    else:
        image = Image.open(path).convert("RGB")
        return ocr_image(image)

# ============================================================
# LOAD LLM
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print("Loading PEFT adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_MODEL)
model.eval()

# ============================================================
# LLM PARSER
# ============================================================
SYSTEM_PROMPT = """
You are a financial invoice parser specialized in Indian invoices.

Rules:
- Use ONLY the provided text
- Do NOT guess or infer values
- If a value is missing or unclear, use null
- Output ONLY valid JSON
"""

def parse_invoice(text: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=False
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return json.loads(response)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n--- OCR TEXT ---\n")
    ocr_text = extract_text(INPUT_FILE)
    print(ocr_text)

    print("\n--- PARSED JSON ---\n")
    parsed_json = parse_invoice(ocr_text)
    print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
