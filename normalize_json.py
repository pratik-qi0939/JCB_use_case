import json
import os
import re
from pathlib import Path
from typing import Dict, Set
from google import genai
from dotenv import load_dotenv

load_dotenv()

GT_DIR = r"D:\JCB_use_case\data\ground_truth"
RAW_PRED_DIR = r"D:\JCB_use_case\data\json\textract"
NORMALIZED_PRED_DIR = r"D:\JCB_use_case\data\normalized_preds\textract"

MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


# normalized the predicted JSON to match the ground truth schema using Gemini

SYSTEM_PROMPT = """
You are a STRICT schema normalization engine.

NON-NEGOTIABLE RULES:
- ONLY rename keys and move existing values
- DO NOT add new fields
- DO NOT infer, guess, or compute values
- DO NOT fix OCR mistakes
- DO NOT modify values
- If a field is missing, keep it null
- Output MUST exactly match the target schema structure
- Output VALID JSON ONLY
"""

def extract_json_strict(text: str) -> Dict:
    if not text or not text.strip():
        raise ValueError("Gemini returned empty response")

    text = text.strip()

    # Fast path: pure JSON
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Fallback: extract JSON substring
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in Gemini response")

    return json.loads(match.group(0))


def normalize_to_schema(predicted_json: Dict, target_schema: Dict) -> Dict:
    prompt = f"""
TARGET SCHEMA (structure only, values may be null):
{json.dumps(target_schema, indent=2)}

PARSER OUTPUT JSON:
{json.dumps(predicted_json, indent=2)}

TASK:
Transform the parser output JSON into the TARGET SCHEMA.
Return JSON only.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
            {"role": "user", "parts": [{"text": prompt}]},
        ],
    )

    raw = response.text

    try:
        return extract_json_strict(raw)
    except Exception as e:
        print("\n--- RAW GEMINI OUTPUT (DEBUG) ---")
        print(raw)
        print("--- END RAW OUTPUT ---\n")
        raise e


def normalize_prediction_folder(gt_dir: str, pred_dir: str, out_dir: str):
    gt_files = {f.stem: f for f in Path(gt_dir).glob("*.json")}
    pred_files = {f.stem: f for f in Path(pred_dir).glob("*.json")}

    os.makedirs(out_dir, exist_ok=True)

    for name, gt_path in gt_files.items():
        if name not in pred_files:
            continue

        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        with open(pred_files[name], encoding="utf-8") as f:
            pred = json.load(f)

        normalized = normalize_to_schema(pred, gt)

        # HARD SAFETY CHECK
        if normalized.keys() != gt.keys():
            raise ValueError(f"Schema mismatch after Gemini normalization: {name}")

        out_path = Path(out_dir) / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)

        print(f"✓ Normalized: {name}")


def flatten_dict(d: Dict, parent: str = "") -> Set[str]:
    keys = set()
    for k, v in d.items():
        path = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            keys |= flatten_dict(v, path)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    keys |= flatten_dict(item, f"{path}[{i}]")
                else:
                    keys.add(f"{path}[{i}]")
        else:
            keys.add(path)
    return keys


def canonical_keys(data: Dict) -> Set[str]:
    return {re.sub(r"\[\d+\]", "[0]", k) for k in flatten_dict(data)}


def f1_score(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def evaluate_schema(ground_truth: Dict, predicted: Dict) -> Dict:
    gt_keys = canonical_keys(ground_truth)
    pred_keys = canonical_keys(predicted)

    intersection = gt_keys & pred_keys
    missing = gt_keys - pred_keys

    recall = len(intersection) / len(gt_keys)
    precision = len(intersection) / max(len(pred_keys), 1)
    f1 = f1_score(precision, recall)
    loss = len(missing) / len(gt_keys)

    return {
        "schema_recall": round(recall, 4),
        "schema_precision": round(precision, 4),
        "schema_f1": round(f1, 4),
        "key_loss": round(loss, 4),
    }


def process_folders(gt_dir: str, pred_dir: str):
    gt_files = {f.stem: f for f in Path(gt_dir).glob("*.json")}
    pred_files = {f.stem: f for f in Path(pred_dir).glob("*.json")}

    common = sorted(set(gt_files) & set(pred_files))

    total_r = total_p = total_f1 = total_loss = 0.0

    for name in common:
        with open(gt_files[name], encoding="utf-8") as f:
            gt = json.load(f)

        with open(pred_files[name], encoding="utf-8") as f:
            pred = json.load(f)

        m = evaluate_schema(gt, pred)

        total_r += m["schema_recall"]
        total_p += m["schema_precision"]
        total_f1 += m["schema_f1"]
        total_loss += m["key_loss"]

        print(
            f"{name} | "
            f"Recall={m['schema_recall']:.2f} "
            f"Precision={m['schema_precision']:.2f} "
            f"F1={m['schema_f1']:.2f} "
            f"Loss={m['key_loss']:.2f}"
        )

    n = len(common)
    print("\nFINAL SUMMARY")
    print("Invoices:", n)
    print("Avg Recall:", total_r / n)
    print("Avg Precision:", total_p / n)
    print("Avg F1:", total_f1 / n)
    print("Avg Loss:", total_loss / n)


if __name__ == "__main__":
    print("\n--- STEP 1: NORMALIZING WITH GEMINI")
    normalize_prediction_folder(GT_DIR, RAW_PRED_DIR, NORMALIZED_PRED_DIR)

    print("\n--- STEP 2: SCHEMA EVALUATION ---")
    process_folders(GT_DIR, NORMALIZED_PRED_DIR)
