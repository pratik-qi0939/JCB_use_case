import os
import re
import json
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import cv2
import numpy as np
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from collections import defaultdict
from google import genai
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')
load_dotenv()

PROCESSOR = None
MODEL = None
DEVICE = None

# Initialize Gemini client
GEMINI_CLIENT = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Extraction prompt for Gemini
EXTRACTION_PROMPT = """
    You are an information extraction engine.

    You will receive OCR-extracted TEXT from EXACTLY invoice or bill.

    RULES:
    - Use ONLY the information present in the text
    - Do NOT infer, guess, or fabricate any values
    - If a value is missing or unclear, use null
    - Output MUST be valid JSON only
    - No explanations, no markdown, no extra text

    TASK:
    Extract the key invoice details from the given text.
    make a meaningful key so that comparision with ground truth is easy.
    If the text does not contain invoice-related information, prepare the JSON output with given information.
    dont output empty JSON.
    Output must be JSON only.


    Summary:
    print a summary of the invoice  
    Show approval or Rejection based on GST applied on invoice and GSTIN Number availability
    Finally Give reasoning if the Invoice is Rejected if the tax details and GST/ GSTIN details are present or missing.
"""

# initialize LayoutLMv3 model
def initialize_layoutlm(model_name="microsoft/layoutlmv3-base", use_gpu=True):
    """Initialize LayoutLMv3 model and processor"""
    global PROCESSOR, MODEL, DEVICE
    
    print("Initializing LayoutLMv3 model...")
    
    DEVICE = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    
    PROCESSOR = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
    MODEL = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
    
    MODEL.to(DEVICE)
    MODEL.eval()
    print("Model loaded successfully\n")

# converting PDF to images
def pdf_to_images(pdf_path, output_dir="pages"):
    """Convert PDF to high-quality images"""
    Path(output_dir).mkdir(exist_ok=True)
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    for i, img in enumerate(images):
        path = os.path.join(output_dir, f"page_{i+1}.png")
        img.save(path, "PNG")
        image_paths.append(path)
    return image_paths


def preprocess_image(image):
    """Advanced image preprocessing for better OCR"""
    img_array = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )
    
    # Morphological operations
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return Image.fromarray(cleaned)


def extract_ocr_data(image_path, use_preprocessing=True):
    """
    Extract text and bounding boxes using Tesseract
    Returns words, boxes, and original image
    """     
    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    
    # Preprocess if enabled
    if use_preprocessing:
        processed_image = preprocess_image(image)
    else:
        processed_image = image
    
    # Get word-level OCR data
    custom_config = r'--oem 3 --psm 3'
    ocr_data = pytesseract.image_to_data(
        processed_image,
        output_type=pytesseract.Output.DICT,
        config=custom_config
    )
    
    # Extract words and bounding boxes
    words = []
    boxes = []
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0
        
        # Filter low confidence
        if text and conf > 30:
            # Normalize bounding box to 0-1000 scale (LayoutLM requirement)
            left = int(ocr_data['left'][i] * 1000 / width)
            top = int(ocr_data['top'][i] * 1000 / height)
            right = int((ocr_data['left'][i] + ocr_data['width'][i]) * 1000 / width)
            bottom = int((ocr_data['top'][i] + ocr_data['height'][i]) * 1000 / height)
            
            # Ensure bounds are valid
            left = max(0, min(left, 1000))
            top = max(0, min(top, 1000))
            right = max(0, min(right, 1000))
            bottom = max(0, min(bottom, 1000))
            
            if right > left and bottom > top:
                words.append(text)
                boxes.append([left, top, right, bottom])
    
    print(f"Extracted {len(words)} words")
    return words, boxes, image

def process_with_layoutlm(words, boxes, image):
    """
    Use LayoutLMv3 to understand document structure
    Returns structured text
    """
    print(f"Running LayoutLMv3 document understanding..")
    
    if not words:
        print("No words found")
        return ""
    
    # Prepare input for LayoutLMv3
    encoding = PROCESSOR(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    
    # Move to device
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    
    # Get model predictions for token classification
    with torch.no_grad():
        outputs = MODEL(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
    
    # Handle single prediction case
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    # Reconstruct text using spatial layout information
    text = reconstruct_text_from_layout(words, boxes)
    
    return text

def reconstruct_text_from_layout(words, boxes):
    """
    Reconstruct text preserving document layout
    Groups words by vertical position (lines) and sorts horizontally
    """
    if not words or not boxes:
        return ""
    
    # Group words by line based on vertical position
    lines = defaultdict(list)
    
    for word, box in zip(words, boxes):
        # Get vertical center (in normalized 0-1000 scale)
        y_center = (box[1] + box[3]) / 2
        
        # Group into lines (tolerance of 15 units in normalized space)
        line_key = round(y_center / 15) * 15
        
        lines[line_key].append({
            'word': word,
            'x': box[0],  
            'box': box
        })
    
    # Sort lines by vertical position
    sorted_lines = sorted(lines.items())
    
    # Reconstruct text line by line
    output_lines = []
    prev_y = None
    
    for y_pos, line_words in sorted_lines:
        # Sort words in line by horizontal position
        line_words.sort(key=lambda x: x['x'])
        
        # Join words with spaces
        line_text = ' '.join([w['word'] for w in line_words])
        
        # Add paragraph breaks for large vertical gaps
        if prev_y is not None and (y_pos - prev_y) > 50:
            output_lines.append('')
        
        output_lines.append(line_text)
        prev_y = y_pos
    
    return '\n'.join(output_lines)


def post_process_text(text):
    """Clean up extracted text"""
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Remove multiple newlines (keep max 2)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)
    # Remove empty lines at start and end
    text = text.strip()
    
    return text

def extract_text(input_path, output_txt=None, use_preprocessing=True):
    """
    Extract text from PDF or image using LayoutLMv3 + Tesseract
    
    Args:
        input_path: Path to input file
        output_txt: Path to save output text
        use_preprocessing: Whether to preprocess images
    
    Returns:
        Extracted text as string
    """
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
        print(f"\n  Page {idx}/{len(image_paths)}")
        
        try:
            # Step 1: Extract OCR data with Tesseract
            words, boxes, image = extract_ocr_data(image_path, use_preprocessing)
            # Step 2: Use LayoutLMv3 for document understanding
            text = process_with_layoutlm(words, boxes, image)
            # Step 3: Post-process
            text = post_process_text(text)
            all_text.append(text)
            print(f"Extracted {len(text)} characters")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            all_text.append("")
    
    # Combine all pages
    if len(all_text) > 1:
        final_text = ("\n\n" + "="*60 + " PAGE BREAK " + "="*60 + "\n\n").join(all_text)
    else:
        final_text = all_text[0] if all_text else ""
    
    # Save to file if path provided
    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"\nSaved text to: {output_txt}")
    return final_text

def extract_json_from_text(text: str) -> dict:
    """
    Extract structured JSON data from text using Gemini API
    
    Args:
        text: OCR-extracted text
    
    Returns:
        Dictionary with extracted invoice data
    """
    print("    Running Gemini extraction...")
    
    if not text.strip():
        print("Empty text, returning empty dict")
        return {}
    
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.5-flash",
            contents=[EXTRACTION_PROMPT, text]
        )
        
        raw_output = (response.text or "").strip()
        
        if not raw_output:
            return {}
        
        # Try to parse JSON
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or other formatting
            match = re.search(r"\{[\s\S]*\}", raw_output)
            if match:
                return json.loads(match.group(0))
            else:
                print(f"Could not parse JSON from response")
                return {}
                
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return {}

def process_file_to_json(input_file, txt_output_dir, json_output_dir, save_txt=True):
    """
    Process a single file: extract text and convert to JSON
    
    Args:
        input_file: Path to input PDF/image
        txt_output_dir: Directory to save intermediate .txt files
        json_output_dir: Directory to save final .json files
        save_txt: Whether to save intermediate .txt files
    
    Returns:
        Tuple of (text, json_data)
    """
    # Generate output filenames
    input_filename = os.path.basename(input_file)
    base_name = os.path.splitext(input_filename)[0]
    
    txt_file = os.path.join(txt_output_dir, f"{base_name}.txt") if save_txt else None
    json_file = os.path.join(json_output_dir, f"{base_name}.json")
    
    # Step 1: Extract text using LayoutLMv3 + Tesseract
    text = extract_text(input_file, txt_file, use_preprocessing=True)
    
    # Step 2: Extract structured data using Gemini
    print("\nConverting to JSON...")
    json_data = extract_json_from_text(text)
    
    # Step 3: Save JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved JSON to: {json_file}")
    
    return text, json_data

def main():
    """Main processing function"""
    
    # Configuration
    input_dir = "D:\\JCB_use_case\\data\\invoices"
    txt_output_dir = "D:\\JCB_use_case\\data\\txt_files"
    json_output_dir = "D:\\JCB_use_case\\data\\json\\doc_AI"
    model_name = "microsoft/layoutlmv3-base"
    save_intermediate_txt = True  # Set to False if don't want .txt files
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    # Create output directories
    Path(txt_output_dir).mkdir(parents=True, exist_ok=True)
    Path(json_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all PDF and image files
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
    input_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]
    
    if not input_files:
        print(f"No PDF or image files found in: {input_dir}")
        return
    
    # Initialize LayoutLMv3 model
    initialize_layoutlm(model_name=model_name, use_gpu=True)
    
    # Process each file
    successful = 0
    failed = 0
    failed_files = []
    
    for idx, input_file in enumerate(input_files, 1):
        try:
            input_filename = os.path.basename(input_file)
            
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(input_files)}] Processing: {input_filename}")
            print("="*70)
            
            # Process file to JSON
            text, json_data = process_file_to_json(
                input_file,
                txt_output_dir,
                json_output_dir,
                save_txt=save_intermediate_txt
            )
            
            print(f"\n  ✓ SUCCESS!")
            print(f"    Text: {len(text)} characters")
            print(f"    JSON: {len(json_data)} fields extracted")
            successful += 1
            
            # Show JSON preview
            if json_data:
                print(f"\n  JSON Preview:")
                print("  " + "-"*66)
                preview = json.dumps(json_data, indent=2, ensure_ascii=False)
                preview_lines = preview.split('\n')[:15]
                for line in preview_lines:
                    print(f"  {line}")
                if len(preview.split('\n')) > 15:
                    print("  ...")
                print("  " + "-"*66)
            
        except Exception as e:
            print(f"\nERROR: {str(e)}")
    
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"Total files: {len(input_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    
    print(f"\nOutput directories:")
    print(f"  TXT:  {txt_output_dir}")
    print(f"  JSON: {json_output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()