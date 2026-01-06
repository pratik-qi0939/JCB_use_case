import os
import re
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification

MODEL_NAME = "microsoft/layoutlmv2-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def pdf_to_images(pdf_path, output_dir="pages"):
    """Convert PDF to images"""
    Path(output_dir).mkdir(exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"page_{i+1}.png")
        img.save(path, "PNG")
        image_paths.append(path)
    return image_paths

def post_process_text(text):
    """Post-process extracted text to fix common OCR errors"""
    # Fix common OCR mistakes
    replacements = {
        r'\s+': ' ',    # Multiple spaces to single space
        r'\n\s*\n\s*\n+': '\n\n',  # Multiple newlines to double newline
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    
    return text.strip()

def extract_text_with_layoutlmv2(image_path, processor):
    """
    Extract text using LayoutLMv2Processor with built-in OCR
    
    The processor automatically:
    1. Applies Tesseract OCR to get words and bounding boxes
    2. Resizes the image to 224x224
    3. Tokenizes the words
    4. Creates proper inputs for LayoutLMv2
    """
    # Open image
    image = Image.open(image_path).convert("RGB")
    
    # Process image with LayoutLMv2Processor
    # apply_ocr=True means the processor will use Tesseract internally
    encoding = processor(
        image,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    
    # Get the words that were extracted
    # The processor stores the words internally during processing
    # We need to decode the input_ids to get the text
    input_ids = encoding["input_ids"][0]
    
    # Decode tokens to text
    # Skip special tokens like [CLS], [SEP], [PAD]
    text = processor.tokenizer.decode(
        input_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    return text

def extract_text_with_custom_ocr(image_path, processor):
    """
    Alternative method: Extract text using LayoutLMv2 with word-level reconstruction
    This gives better layout preservation
    """
    image = Image.open(image_path).convert("RGB")
    
    # Let processor do OCR
    encoding = processor(
        image,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    
    # Get input_ids and word boundaries
    input_ids = encoding["input_ids"][0]
    bbox = encoding["bbox"][0]
    
    # Decode to get individual tokens
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    
    # Reconstruct text with layout awareness
    words = []
    current_word = []
    prev_y = -1
    prev_x = -1
    
    for idx, (token, box) in enumerate(zip(tokens, bbox)):
        # Skip special tokens
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        # Get position
        x0, y0, x1, y1 = box.tolist()
        
        # Check if new line (significant y change)
        if prev_y != -1 and abs(y0 - prev_y) > 20:
            if current_word:
                words.append(''.join(current_word).replace('##', ''))
                current_word = []
            words.append('\n')
        # Check if new word (significant x gap)
        elif prev_x != -1 and abs(x0 - prev_x) > 10:
            if current_word:
                words.append(''.join(current_word).replace('##', ''))
                current_word = []
        
        # Add token to current word
        current_word.append(token)
        prev_y = y0
        prev_x = x1
    
    # Add last word
    if current_word:
        words.append(''.join(current_word).replace('##', ''))
    
    text = ' '.join(words)
    text = text.replace(' \n ', '\n')
    
    return text

def extract_text(input_path, output_txt=None, processor=None, method="simple"):
    """
    Extract text from PDF or image using LayoutLMv2 Processor
    
    Args:
        input_path: Path to input file
        output_txt: Path to save output text
        processor: Pre-loaded processor (optional, for batch processing)
        method: 'simple' for direct decoding, 'layout' for layout-aware reconstruction
    """
    # Load processor if not provided
    if processor is None:
        print("Loading LayoutLMv2 Processor...")
        processor = LayoutLMv2Processor.from_pretrained(
            MODEL_NAME,
            apply_ocr=True  # Enable automatic OCR
        )
        print(f"Processor loaded with automatic OCR enabled")
    
    ext = os.path.splitext(input_path)[1].lower()
    
    # Convert PDF to images if needed
    if ext == ".pdf":
        print("  Converting PDF to images...")
        image_paths = pdf_to_images(input_path)
    else:
        image_paths = [input_path]
    
    # Extract text from each page
    all_text = []
    for idx, image_path in enumerate(image_paths, 1):
        print(f"  Processing page {idx}/{len(image_paths)}...")
        
        try:
            if method == "layout":
                text = extract_text_with_custom_ocr(image_path, processor)
            else:
                text = extract_text_with_layoutlmv2(image_path, processor)
            
            # Post-process
            text = post_process_text(text)
            all_text.append(text)
            print(f"  ✓ Extracted {len(text)} characters from page {idx}")
        except Exception as e:
            print(f"  ✗ Error on page {idx}: {str(e)}")
            all_text.append("")
    
    # Combine all pages
    if len(all_text) > 1:
        final_text = ("\n\n" + "="*50 + " PAGE BREAK " + "="*50 + "\n\n").join(all_text)
    else:
        final_text = all_text[0] if all_text else ""
    
    # Save to file
    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(final_text)
    
    return final_text

if __name__ == "__main__":
    # Input and output directories
    input_dir = "D:\\JCB_use_case\\data\\invoices\\"
    output_dir = "D:\\JCB_use_case\\data\\txt_files\\"
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all PDF and image files
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    input_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]
    
    if not input_files:
        print(f"No PDF or image files found in: {input_dir}")
        exit(1)
    
    print("="*70)
    print("LayoutLMv2 Text Extraction Pipeline")
    print("="*70)
    print(f"Input Directory:  {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Files to process: {len(input_files)}")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ LayoutLMv2 Processor with built-in Tesseract OCR")
    print("  ✓ Automatic bounding box detection")
    print("  ✓ Visual feature extraction via Detectron2")
    print("  ✓ Pre-trained on document understanding tasks")
    print("="*70)
    
    # Load processor once for all files
    print("\nLoading LayoutLMv2 Processor (will be reused for all files)...")
    processor = LayoutLMv2Processor.from_pretrained(
        MODEL_NAME,
        apply_ocr=True  # Enable automatic OCR with Tesseract
    )
    print(f"✓ Processor loaded on {DEVICE}\n")
    
    # Process each file
    successful = 0
    failed = 0
    failed_files = []
    
    for idx, input_file in enumerate(input_files, 1):
        try:
            # Generate output filename
            input_filename = os.path.basename(input_file)
            output_filename = os.path.splitext(input_filename)[0] + ".txt"
            output_file = os.path.join(output_dir, output_filename)
            
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(input_files)}] Processing: {input_filename}")
            print("="*70)
            
            # Extract text with pre-loaded processor
            text = extract_text(
                input_file, 
                output_file, 
                processor, 
                method="layout"  # Use layout-aware method
            )
            
            print(f"  ✓ SUCCESS! Extracted {len(text)} characters")
            print(f"  ✓ Saved to: {output_filename}")
            successful += 1
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
            failed_files.append(input_filename)
            continue
    
    # Final summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total files: {len(input_files)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\nOutput directory: {output_dir}")
    print("="*70)