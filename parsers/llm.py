from google import genai
from dotenv import load_dotenv
import os
import json
import re

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

EXTRACTION_PROMPT = """
You are an information extraction engine.

You will receive OCR-extracted TEXT from EXACTLY ONE invoice or bill.

RULES:
- Use ONLY the information present in the text
- Do NOT infer, guess, or fabricate any values
- If a value is missing or unclear, use null
- Output MUST be valid JSON only
- No explanations, no markdown, no extra text

TASK:
Extract the key invoice details from the given text.

If the text does not contain invoice-related information, output {}.

Output JSON only.
"""

def read_txt_file(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def extract_key_details_from_txt(txt_path: str) -> dict:
    text = read_txt_file(txt_path)

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[EXTRACTION_PROMPT, text],
        temperature=0,
    )

    raw_output = (response.text or "").strip()

    if not raw_output:
        return {}

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw_output)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("LLM returned no valid JSON")

if __name__ == "__main__":
    INPUT_TXT_FILE = "D:/JCB_use_case/data/txt_files/printed/invoice_1_P.txt"
    OUTPUT_JSON_FILE = "D:/JCB_use_case/data/JSON/printed/invoice_1_P.json"

    extracted_json = extract_key_details_from_txt(INPUT_TXT_FILE)

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_json, f, indent=2, ensure_ascii=False)

    print("Extraction completed")
    # print(extracted_json)
