import streamlit as st
import tempfile
import os
import re
import json
from typing import Tuple, Dict, List

from google import genai

from parser import (
    initialize_layoutlm,
    extract_text,
    extract_json_from_text
)


st.set_page_config(
    page_title="Invoice Validation",
    layout="wide"
)

GSTIN_REGEX = r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}\b"


def extract_gstin(text: str) -> str | None:
    """Extract GSTIN from text"""
    match = re.search(GSTIN_REGEX, text)
    return match.group(0) if match else None

def validate_gstin(text: str) -> Tuple[bool, str]:
    """Validate GSTIN presence and format"""
    gstin = extract_gstin(text)
    if not gstin:
        return False, "GSTIN not found in invoice"
    return True, f"Valid GSTIN found: {gstin}"

def find_gst_amounts(data: dict, path: str = "") -> List[Tuple[str, any]]:
    """Recursively find all GST-related fields in the JSON"""
    gst_fields = []
    gst_keywords = ["gst", "cgst", "sgst", "igst", "tax"]
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            key_lower = key.lower()
            
            # Check if this key contains GST-related terms
            if any(keyword in key_lower for keyword in gst_keywords):
                gst_fields.append((current_path, value))
            
            # Recurse into nested structures
            if isinstance(value, (dict, list)):
                gst_fields.extend(find_gst_amounts(value, current_path))
    
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            current_path = f"{path}[{idx}]"
            if isinstance(item, (dict, list)):
                gst_fields.extend(find_gst_amounts(item, current_path))
    
    return gst_fields

def validate_gst_amounts(json_data: dict) -> Tuple[bool, str, List[Tuple[str, any]]]:
    """Validate that GST amounts are present and not null"""
    gst_fields = find_gst_amounts(json_data)
    
    if not gst_fields:
        return False, "No GST fields found in invoice data", []
    
    # Check if at least one GST field has a valid non-null value
    valid_amounts = []
    for field_path, value in gst_fields:
        if value is not None and value != "NULL" and value != "":
            # Try to convert to number if it's a string
            try:
                if isinstance(value, str):
                    # Remove currency symbols and commas
                    cleaned = re.sub(r'[₹$,\s]', '', value)
                    numeric_value = float(cleaned)
                    if numeric_value > 0:
                        valid_amounts.append((field_path, numeric_value))
                elif isinstance(value, (int, float)) and value > 0:
                    valid_amounts.append((field_path, value))
            except (ValueError, TypeError):
                continue
    
    if valid_amounts:
        amounts_str = ", ".join([f"{path}: {val}" for path, val in valid_amounts[:3]])
        return True, f"Valid GST amounts found: {amounts_str}", valid_amounts
    
    return False, "GST fields present but all values are null or invalid", gst_fields

def comprehensive_validation(text: str, json_data: dict) -> Tuple[str, Dict[str, any]]:
    """Perform comprehensive validation and return detailed results"""
    
    validation_results = {
        "gstin_valid": False,
        "gstin_message": "",
        "gstin_value": None,
        "gst_amounts_valid": False,
        "gst_message": "",
        "gst_fields": [],
        "overall_status": "REJECTED"
    }
    
    # Validate GSTIN
    gstin_valid, gstin_msg = validate_gstin(text)
    validation_results["gstin_valid"] = gstin_valid
    validation_results["gstin_message"] = gstin_msg
    if gstin_valid:
        validation_results["gstin_value"] = extract_gstin(text)
    
    # Validate GST amounts
    gst_valid, gst_msg, gst_fields = validate_gst_amounts(json_data)
    validation_results["gst_amounts_valid"] = gst_valid
    validation_results["gst_message"] = gst_msg
    validation_results["gst_fields"] = gst_fields
    
    # Determine overall status
    if gstin_valid and gst_valid:
        validation_results["overall_status"] = "APPROVED"
    else:
        validation_results["overall_status"] = "REJECTED"
    
    return validation_results["overall_status"], validation_results


st.sidebar.title("Upload Invoice")

uploaded_file = st.sidebar.file_uploader(
    "Select PDF/Image file",
    type=["pdf", "png", "jpg", "jpeg"]
)


st.title("Invoice Validation & Approval System")
st.markdown("Automated GST compliance checking")

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Processing invoice..."):
            initialize_layoutlm(use_gpu=True)
            extracted_text = extract_text(file_path)
            extracted_json = extract_json_from_text(extracted_text)

        # Perform comprehensive validation
        result, validation_details = comprehensive_validation(extracted_text, extracted_json)

        # Display Result Banner
        st.markdown("## Validation Result")
        if result == "APPROVED":
            st.success("APPROVED – Invoice is GST compliant")
        else:
            st.error("REJECTED – Invoice failed GST compliance checks")

        # Validation Summary Cards
        col1, col2 = st.columns(2)
        
        with col1:
            if validation_details["gstin_valid"]:
                st.metric(
                    label="GSTIN Status",
                    value="Valid ✓",
                    delta=validation_details["gstin_value"]
                )
            else:
                st.metric(label="GSTIN Status", value="Invalid ✗")
        
        with col2:
            if validation_details["gst_amounts_valid"]:
                st.metric(
                    label="GST Amount Status",
                    value="Valid ✓",
                    delta=f"{len(validation_details['gst_fields'])} field(s) found"
                )
            else:
                st.metric(label="GST Amount Status", value="Invalid ✗")

        st.divider()

        with st.expander("🔍 Detailed Validation Report", expanded=True):
            st.markdown("### GSTIN Validation")
            if validation_details["gstin_valid"]:
                st.success(validation_details["gstin_message"])
            else:
                st.error(validation_details["gstin_message"])
            
            st.markdown("### GST Amount Validation")
            if validation_details["gst_amounts_valid"]:
                st.success(validation_details["gst_message"])
                if validation_details["gst_fields"]:
                    st.markdown("**GST Fields Found:**")
                    for field_path, value in validation_details["gst_fields"]:
                        st.markdown(f"- `{field_path}`: **{value}**")
            else:
                st.error(validation_details["gst_message"])
                if validation_details["gst_fields"]:
                    st.warning("GST-related fields found but values are null or invalid:")
                    for field_path, value in validation_details["gst_fields"][:5]:
                        st.markdown(f"- `{field_path}`: {value}")

        st.divider()

        left_col, right_col = st.columns([1.1, 1])

        with left_col:
            st.subheader("Extracted Invoice Data")

            st.json(extracted_json)

        with right_col:
            st.subheader("AI Compliance Analysis")

            justification_prompt = f"""
You are a financial compliance assistant analyzing an invoice for GST compliance.

**Decision:** {result}

**Validation Details:**
- GSTIN Valid: {validation_details['gstin_valid']}
- GSTIN Message: {validation_details['gstin_message']}
- GST Amounts Valid: {validation_details['gst_amounts_valid']}
- GST Message: {validation_details['gst_message']}

**Compliance Rules:**
1. A valid GSTIN (GST Identification Number) must be present in the format: 22AAAAA0000A1Z5
2. GST must be applied on the invoice with non-null values
3. At least one valid GST amount field (CGST, SGST, IGST, or Total GST) must contain a numeric value

**Extracted Invoice JSON:**
{json.dumps(extracted_json, indent=2)}

**Raw Text Extract (first 500 chars):**
{extracted_text[:500]}

Provide a clear, professional justification explaining:
1. Why the invoice was {result.lower()}
2. What specific issues were found (if rejected)
3. What compliance requirements were met or missed
4. Any recommendations for the invoice issuer

Be concise, factual, and professional. Do not hallucinate or infer information not present in the data.
"""

            with st.spinner("Generating AI analysis..."):
                try:
                    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                    response = client.models.generate_content(
                        model="gemini-2.0-flash-exp",
                        contents=justification_prompt
                    )

                    st.markdown(
                        """
                        <div style="
                            padding: 20px;
                            background: linear-gradient(135deg, #f5f7fb 0%, #e8ecf5 100%);
                            border-left: 5px solid #2c7be5;
                            border-radius: 8px;
                            font-size: 15px;
                            line-height: 1.6;
                        ">
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(response.text)

                    st.markdown("</div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error generating AI analysis: {str(e)}")

        with st.expander("📝 Raw Extracted Text", expanded=False):
            st.text_area(
                "Full extracted text from invoice",
                value=extracted_text,
                height=300,
                disabled=True
            )

else:
    st.info("Please upload an invoice file from the sidebar to begin validation")
    