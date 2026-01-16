import json
from pathlib import Path
from typing import Dict, Set
import re
# evaluation with key aliases for normalization
KEY_ALIASES = {
    # Seller
    "seller_details.name": "seller.name",
    "seller_information.name": "seller.name",
    "seller.name": "seller.name",

    "seller_details.gstin": "seller.gstin",
    "seller_information.gstin": "seller.gstin",
    "seller.gst_number": "seller.gstin",

    "seller_details.pan": "seller.pan",
    "seller_information.pan_number": "seller.pan",

    "seller_details.address": "seller.address",
    "seller_information.address": "seller.address",

    # Buyer
    "buyer_details.name": "buyer.name",
    "buyer_information.name": "buyer.name",
    "buyer.name": "buyer.name",

    "buyer_details.gstin": "buyer.gstin",
    "buyer_information.gstin": "buyer.gstin",
    "buyer.gst_number": "buyer.gstin",

    # Invoice
    "invoice_details.invoice_number": "invoice.number",
    "invoice.id": "invoice.number",

    "invoice_details.invoice_date": "invoice.date",
    "invoice.date": "invoice.date",

    # Items
    "line_items[0].description": "items[].description",
    "line_items[0].particulars": "items[].description",
    "items[0].item": "items[].description",

    "line_items[0].amount": "items[].amount",
    "items[0].price": "items[].amount",

    # Tax
    "tax_summary.cgst_amount": "tax.cgst",
    "tax_details.cgst": "tax.cgst",
    "invoice.cgst": "tax.cgst",

    "tax_summary.sgst_amount": "tax.sgst",
    "tax_details.sgst": "tax.sgst",
    "invoice.sgst": "tax.sgst",

    # Bank
    "bank_details.ifsc_code": "bank.ifsc",
    "seller.ifsc_code": "bank.ifsc",

    "bank_details.account_number": "bank.account_number",
    "seller.account_number": "bank.account_number",
}


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
    raw_keys = flatten_dict(data)
    canonical = set()

    for k in raw_keys:
        k = re.sub(r"\[\d+\]", "[0]", k) 
        canonical.add(KEY_ALIASES.get(k, k))

    return canonical


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
        "missing_keys": sorted(missing),
    }


def process_folders(gt_dir: str, pred_dir: str) -> Dict:
    gt_files = {f.stem: f for f in Path(gt_dir).glob("*.json")}
    pred_files = {f.stem: f for f in Path(pred_dir).glob("*.json")}

    common = sorted(set(gt_files) & set(pred_files))

    results = []
    total_r = total_p = total_f1 = total_loss = 0.0

    for name in common:
        gt = json.load(open(gt_files[name], encoding="utf-8"))
        pred = json.load(open(pred_files[name], encoding="utf-8"))

        metrics = evaluate_schema(gt, pred)
        metrics["filename"] = name
        results.append(metrics)

        total_r += metrics["schema_recall"]
        total_p += metrics["schema_precision"]
        total_f1 += metrics["schema_f1"]
        total_loss += metrics["key_loss"]

        print(
            f"{name} | "
            f"Recall={metrics['schema_recall']:.2f} "
            f"Precision={metrics['schema_precision']:.2f} "
            f"F1={metrics['schema_f1']:.2f} "
            f"Loss={metrics['key_loss']:.2f}"
        )

    n = len(common)
    return {
        "total_invoices": n,
        "average_schema_recall": total_r / n,
        "average_precision": total_p / n,
        "average_f1": total_f1 / n,
        "average_key_loss": total_loss / n,
        "per_invoice_results": results,
    }

if __name__ == "__main__":
    GT_DIR = r"D:\JCB_use_case\data\ground_truth"
    PRED_DIR = r"D:\JCB_use_case\data\json\textract"

    results = process_folders(GT_DIR, PRED_DIR)

    print("\nFINAL SUMMARY")
    print("Invoices:", results["total_invoices"])
    print("Avg Recall:", results["average_schema_recall"])
    print("Avg Precision:", results["average_precision"])
    print("Avg F1:", results["average_f1"])
    print("Avg Loss:", results["average_key_loss"])
