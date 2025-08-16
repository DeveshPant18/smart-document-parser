import streamlit as st
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pytesseract
import numpy as np
import pandas as pd

# =============================================================================
# CRITICAL FIX: set_page_config() MUST be the first Streamlit command.
# All other Streamlit commands are below this line.
st.set_page_config(layout="wide")
# =============================================================================


# --- Configuration ---
# Set the path to your Tesseract executable if it's not in your system's PATH
# For Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For Linux: You might need to install it via `sudo apt-get install tesseract-ocr`

MODEL_PATH = './models/layoutlm-sroie/best_model'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model and Tokenizer ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the fine-tuned LayoutLM model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        return tokenizer, model
    except Exception as e:
        # This is the first potential Streamlit command AFTER set_page_config
        st.error(f"Error loading model: {e}. Make sure the model is available at '{MODEL_PATH}'")
        return None, None

tokenizer, model = load_model()

# --- Helper Functions ---

def run_ocr(image):
    """
    Runs Tesseract OCR on an image and returns words and normalized bounding boxes.
    """
    # Use pytesseract to get detailed OCR data
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    words = []
    boxes = []
    width, height = image.size

    for i in range(len(ocr_data['text'])):
        # Filter out empty words and non-confident detections
        if int(ocr_data['conf'][i]) > 30 and ocr_data['text'][i].strip() != '':
            word = ocr_data['text'][i]
            x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            
            # Normalize the bounding box to a 0-1000 scale
            normalized_box = [
                int(1000 * x / width),
                int(1000 * y / height),
                int(1000 * (x + w) / width),
                int(1000 * (y + h) / height),
            ]
            
            words.append(word)
            boxes.append(normalized_box)
            
    return words, boxes

def process_image(image):
    """
    Processes an uploaded image to extract entities using the LayoutLM model.
    """
    if not tokenizer or not model:
        return {}

    # 1. Run OCR to get words and boxes
    words, boxes = run_ocr(image)
    if not words:
        st.warning("OCR did not detect any text on the image.")
        return {}

    # 2. Tokenize and prepare inputs for the model
    tokenized_inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Align bounding boxes with tokens
    word_ids = tokenized_inputs.word_ids()
    aligned_boxes = []
    for word_idx in word_ids:
        if word_idx is None:
            aligned_boxes.append([0, 0, 0, 0])
        else:
            aligned_boxes.append(boxes[word_idx])
    
    # Prepare inputs for PyTorch
    input_ids = tokenized_inputs["input_ids"].to(DEVICE)
    attention_mask = tokenized_inputs["attention_mask"].to(DEVICE)
    token_type_ids = tokenized_inputs["token_type_ids"].to(DEVICE)
    bbox = torch.tensor([aligned_boxes]).to(DEVICE)

    # 3. Run Inference
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bbox=bbox
        )

    # 4. Post-process the predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    
    # Group tokens and their predictions
    results = {}
    current_entity = None
    current_words = []

    for token, pred_id in zip(tokens, predictions):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
            
        pred_label = model.config.id2label[pred_id]
        
        if pred_label.startswith("B-"):
            # If we have a current entity, save it
            if current_entity:
                results[current_entity] = " ".join(current_words)
            
            # Start a new entity
            current_entity = pred_label[2:] # Remove "B-"
            current_words = [token.replace("##", "")]
        
        elif pred_label.startswith("I-") and current_entity == pred_label[2:]:
            # Continue the current entity
            current_words.append(token.replace("##", ""))
            
        else:
            # Not part of an entity, so save the current one if it exists
            if current_entity:
                results[current_entity] = " ".join(current_words)
                current_entity = None
                current_words = []

    # Add the last entity if it exists
    if current_entity:
        results[current_entity] = " ".join(current_words)

    return results


# --- Streamlit App UI ---
st.title("🧾 Smart Document Parser")
st.write("Upload a receipt image and the AI will extract the key information.")

uploaded_file = st.file_uploader("Choose a receipt image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Receipt", use_container_width=True)
    
    with col2:
        with st.spinner("🧠 Analyzing the document..."):
            extracted_data = process_image(image)
            
            if extracted_data:
                st.success("✅ Information Extracted Successfully!")
                
                # Display data in a more structured way
                df = pd.DataFrame(extracted_data.items(), columns=['Entity', 'Value'])
                st.table(df)
                
                st.write("### JSON Output")
                st.json(extracted_data)
            else:
                st.error("Could not extract any information. Please try another image.")
