import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher
import re


class InvoiceLossCalculator:
    """
    Calculate loss between ground truth and predicted invoice JSON.
    Loss is based on:
    1. Percentage of fields not extracted
    2. Percentage of wrongly extracted field values
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: Threshold for considering strings as matching (0-1)
                                For OCR errors, 0.85 is a good balance
        """
        self.similarity_threshold = similarity_threshold
    
    def process_folders(self, ground_truth_folder: str, json_folder: str) -> Dict:
        """
        Process all invoice pairs and calculate losses.
        
        Args:
            ground_truth_folder: Path to folder containing ground truth JSONs
            json_folder: Path to folder containing predicted JSONs
            
        Returns:
            Dictionary with results including per-invoice losses and average
        """
        gt_path = Path(ground_truth_folder)
        pred_path = Path(json_folder)
        
        # Get all JSON files
        gt_files = {f.stem: f for f in gt_path.glob('*.json')}
        pred_files = {f.stem: f for f in pred_path.glob('*.json')}
        
        results = {
            'per_invoice_results': [],
            'total_invoices': 0,
            'average_loss': 0.0,
            'average_missing_fields_pct': 0.0,
            'average_wrong_values_pct': 0.0
        }
        
        # Find matching files 
        common_files = set(gt_files.keys()) & set(pred_files.keys())
        
        if not common_files:
            print("Warning: No matching files found between folders!")
            return results
        
        total_loss = 0.0
        total_missing_pct = 0.0
        total_wrong_pct = 0.0
        
        for filename in sorted(common_files):
            print(f"\nProcessing: {filename}")
            
            # Load JSONs
            with open(gt_files[filename], 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            
            with open(pred_files[filename], 'r', encoding='utf-8') as f:
                predicted = json.load(f)
            
            # Calculate loss
            loss, missing_pct, wrong_pct, details = self.calculate_loss(
                ground_truth, predicted, filename
            )
            
            results['per_invoice_results'].append({
                'filename': filename,
                'loss': loss,
                'missing_fields_percentage': missing_pct,
                'wrong_values_percentage': wrong_pct,
                'details': details
            })
            
            total_loss += loss
            total_missing_pct += missing_pct
            total_wrong_pct += wrong_pct
            
            print(f"  Missing Fields: {missing_pct:.2f}%")
            print(f"  Wrong Values: {wrong_pct:.2f}%")
            print(f"  Loss: {loss:.4f}")
        
        # Calculate averages
        num_invoices = len(common_files)
        results['total_invoices'] = num_invoices
        results['average_loss'] = total_loss / num_invoices
        results['average_missing_fields_pct'] = total_missing_pct / num_invoices
        results['average_wrong_values_pct'] = total_wrong_pct / num_invoices
        
        return results
    
    def calculate_loss(self, ground_truth: Dict, predicted: Dict, filename: str = "") -> Tuple[float, float, float, Dict]:
        """
        Calculate loss for a single invoice.
        
        Returns:
            Tuple of (loss, missing_fields_pct, wrong_values_pct, details)
        """
        # Flatten both JSONs to get all fields
        gt_fields = self._flatten_dict(ground_truth)
        pred_fields = self._flatten_dict(predicted)
        
        total_gt_fields = len(gt_fields)
        
        if total_gt_fields == 0:
            return 0.0, 0.0, 0.0, {}
        
        # 1. Calculate missing fields percentage
        missing_fields = set(gt_fields.keys()) - set(pred_fields.keys())
        num_missing = len(missing_fields)
        missing_fields_pct = (num_missing / total_gt_fields) * 100
        
        # 2. Calculate wrong values percentage
        common_fields = set(gt_fields.keys()) & set(pred_fields.keys())
        num_extracted = len(common_fields)
        
        wrong_values = []
        if num_extracted > 0:
            for field in common_fields:
                if not self._values_match(gt_fields[field], pred_fields[field]):
                    wrong_values.append(field)
            
            num_wrong = len(wrong_values)
            wrong_values_pct = (num_wrong / num_extracted) * 100
        else:
            wrong_values_pct = 0.0
        
        # 3. Calculate final loss
        loss = ((missing_fields_pct + wrong_values_pct) / 2) / 100
        
        # Prepare details
        details = {
            'total_fields_in_ground_truth': total_gt_fields,
            'fields_extracted': num_extracted,
            'fields_missing': num_missing,
            'fields_with_wrong_values': len(wrong_values),
            'missing_field_names': list(missing_fields),
            'wrong_value_field_names': wrong_values
        }
        
        return loss, missing_fields_pct, wrong_values_pct, details
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """
        Flatten nested dictionary to count all fields.
        Example: {'a': {'b': 1}} becomes {'a.b': 1}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # For lists, we'll create keys for each element
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _values_match(self, gt_value: Any, pred_value: Any) -> bool:
        """
        Check if two values match, with tolerance for OCR errors.
        """
        # Handle None
        if gt_value is None and pred_value is None:
            return True
        if gt_value is None or pred_value is None:
            return False
        
        # Convert to strings for comparison
        gt_str = str(gt_value).strip().lower()
        pred_str = str(pred_value).strip().lower()
        
        # Exact match
        if gt_str == pred_str:
            return True
        
        # Numeric comparison with tolerance
        if self._is_numeric(gt_value) and self._is_numeric(pred_value):
            try:
                gt_num = float(gt_value)
                pred_num = float(pred_value)
                
                # Allow 1% tolerance for numeric values
                if gt_num == 0:
                    return abs(pred_num) < 0.01
                else:
                    relative_error = abs(gt_num - pred_num) / abs(gt_num)
                    return relative_error < 0.01
            except:
                pass
        
        # Fuzzy string matching for OCR tolerance
        similarity = SequenceMatcher(None, gt_str, pred_str).ratio()
        return similarity >= self.similarity_threshold
    
    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def save_results(self, results: Dict, output_file: str):
        """Save results to a JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")


def main():
    ground_truth_folder = r"D:\JCB_use_case\data\ground_truth"
    json_folder = r"D:\JCB_use_case\data\json\doc_AI"
    
    # Initialize calculator
    calculator = InvoiceLossCalculator(similarity_threshold=0.60)
    
    print("="*70)
    print("Invoice Loss Evaluation")
    print("="*70)
    print(f"Ground Truth Folder: {ground_truth_folder}")
    print(f"Predicted JSON Folder: {json_folder}")
    print("="*70)
    
    # Process all invoices
    results = calculator.process_folders(ground_truth_folder, json_folder)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Invoices Processed: {results['total_invoices']}")
    print(f"Average Missing Fields: {results['average_missing_fields_pct']:.2f}%")
    print(f"Average Wrong Values: {results['average_wrong_values_pct']:.2f}%")
    print(f"AVERAGE LOSS: {results['average_loss']:.4f}")
    print(f"AVERAGE ACCURACY: {(1 - results['average_loss']) * 100:.2f}%")
    print("="*70)
    
    # Save detailed results
    output_file = "invoice_loss_results.json"
    calculator.save_results(results, output_file)
    
    # Print per-invoice summary
    print("\nPer-Invoice Results:")
    print("-"*70)
    for result in results['per_invoice_results']:
        print(f"{result['filename']}: Loss={result['loss']:.4f}, "
              f"Missing={result['missing_fields_percentage']:.1f}%, "
              f"Wrong={result['wrong_values_percentage']:.1f}%")


if __name__ == "__main__":
    main()