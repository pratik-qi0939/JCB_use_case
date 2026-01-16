import json
import re
from pathlib import Path
from typing import Dict, Set

GT_DIR = r"D:\JCB_use_case\data\ground_truth"
NORMALIZED_PRED_DIR = r"D:\JCB_use_case\data\normalized_preds\textract"

# compare the ground truth and normalized predicted JSON schemas
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
    # normalize list indices: items[0] == items[1]
    return {re.sub(r"\[\d+\]", "[0]", k) for k in flatten_dict(data)}


def f1_score(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def evaluate_schema(gt: Dict, pred: Dict) -> Dict:
    gt_keys = canonical_keys(gt)
    pred_keys = canonical_keys(pred)

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
        "schema_loss": round(loss, 4),
    }


def flatten_values(d: Dict, parent: str = "") -> Dict[str, object]:
    out = {}
    for k, v in d.items():
        path = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten_values(v, path))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(flatten_values(item, f"{path}[{i}]"))
                else:
                    out[f"{path}[{i}]"] = item
        else:
            out[path] = v
    return out


def strict_value_metrics(gt: Dict, pred: Dict) -> Dict:
    gt_vals = flatten_values(gt)
    pred_vals = flatten_values(pred)

    comparable = 0
    correct = 0
    mismatches = {}

    for key, gt_val in gt_vals.items():
        if gt_val is None:
            continue  # ignore undefined GT fields

        comparable += 1
        pred_val = pred_vals.get(key)

        if pred_val == gt_val:
            correct += 1
        else:
            mismatches[key] = {
                "gt": gt_val,
                "pred": pred_val
            }

    accuracy = correct / comparable if comparable else 1.0
    loss = 1.0 - accuracy

    return {
        "value_accuracy": round(accuracy, 4),
        "value_loss": round(loss, 4),
        "total_fields": comparable,
        "correct_fields": correct,
        "mismatches": mismatches
    }



def evaluate_invoice(gt: Dict, pred: Dict) -> Dict:
    schema = evaluate_schema(gt, pred)
    value = strict_value_metrics(gt, pred)

    return {
        **schema,
        **value
    }


def process_folders(gt_dir: str, normalized_dir: str):
    gt_files = {f.stem: f for f in Path(gt_dir).glob("*.json")}
    pred_files = {f.stem: f for f in Path(normalized_dir).glob("*.json")}

    common = sorted(set(gt_files) & set(pred_files))

    total_schema_loss = 0.0
    total_value_loss = 0.0

    for name in common:
        gt = json.load(open(gt_files[name], encoding="utf-8"))
        pred = json.load(open(pred_files[name], encoding="utf-8"))

        m = evaluate_invoice(gt, pred)

        total_schema_loss += m["schema_loss"]
        total_value_loss += m["value_loss"]

        print(
            f"{name} | "
            f"SchemaLoss={m['schema_loss']:.2f} "
            f"ValueAcc={m['value_accuracy']:.2f} "
            f"ValueLoss={m['value_loss']:.2f}"
        )

    n = len(common)

    print("\nFINAL SUMMARY")
    print("Invoices:", n)
    print("Avg Schema Loss:", round(total_schema_loss / n, 4))
    print("Avg Value Loss:", round(total_value_loss / n, 4))



if __name__ == "__main__":
    process_folders(GT_DIR, NORMALIZED_PRED_DIR)