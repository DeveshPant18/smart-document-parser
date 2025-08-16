import os
import json
from PIL import Image
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Update this path to your project's root if necessary
PROJECT_ROOT = '.' 
DATA_DIR = os.path.join(PROJECT_ROOT, 'data/raw/SROIE2019/')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data/processed/')

# --- Helper Functions ---

def get_image_size(image_path):
    """Gets the width and height of an image."""
    with Image.open(image_path) as img:
        return img.size

def normalize_box(box, width, height):
    """Normalizes a bounding box to a 0-1000 scale."""
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]

def parse_ocr_box(box_file_path):
    """Parses a single OCR box annotation file into words and boxes."""
    # FIX: Added errors='ignore' to handle characters that are not valid utf-8
    with open(box_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    words = []
    boxes = []
    for line in lines:
        parts = line.strip().split(',', 8)
        if len(parts) < 9:
            continue

        coords = [int(p) for p in parts[:8]]
        text = parts[8]
        
        min_x = min(coords[0], coords[2], coords[4], coords[6])
        min_y = min(coords[1], coords[3], coords[5], coords[7])
        max_x = max(coords[0], coords[2], coords[4], coords[6])
        max_y = max(coords[1], coords[3], coords[5], coords[7])
        
        # Split text into words if it contains spaces, and approximate boxes
        # This is a simplification; more advanced methods could be used.
        sub_words = text.split()
        if len(sub_words) > 1:
            word_width = (max_x - min_x) / len(sub_words)
            for i, sw in enumerate(sub_words):
                words.append(sw)
                new_box_x_start = min_x + i * word_width
                new_box_x_end = min_x + (i + 1) * word_width
                boxes.append([new_box_x_start, min_y, new_box_x_end, max_y])
        else:
            words.append(text)
            boxes.append([min_x, min_y, max_x, max_y])
            
    return words, boxes

def get_bio_labels(words, entities):
    """
    Assigns BIO labels to each word based on the ground truth entities.
    """
    labels = ['O'] * len(words)
    
    for entity_type, entity_text in entities.items():
        if not isinstance(entity_text, str) or not entity_text:
            continue
            
        entity_words = entity_text.split()
        if not entity_words:
            continue

        # Simple text matching to find the entity in the OCR words
        for i in range(len(words) - len(entity_words) + 1):
            if words[i:i+len(entity_words)] == entity_words:
                # Found a match
                labels[i] = f'B-{entity_type.upper()}'
                for j in range(1, len(entity_words)):
                    labels[i+j] = f'I-{entity_type.upper()}'
                break # Move to next entity once found
                
    return labels

# --- Main Preprocessing Function ---

def create_dataset(split='train'):
    """
    Processes the raw SROIE data and creates a structured dataset file.
    """
    logging.info(f"Starting preprocessing for '{split}' split...")
    
    split_dir = os.path.join(DATA_DIR, split)
    img_dir = os.path.join(split_dir, 'img')
    box_dir = os.path.join(split_dir, 'box')
    entities_dir = os.path.join(split_dir, 'entities')
    
    dataset = []
    
    filenames = [f.replace('.txt', '') for f in os.listdir(entities_dir) if f.endswith('.txt')]
    
    for filename in tqdm(filenames, desc=f"Processing {split} data"):
        image_path = os.path.join(img_dir, f"{filename}.jpg")
        box_path = os.path.join(box_dir, f"{filename}.txt")
        entities_path = os.path.join(entities_dir, f"{filename}.txt")
        
        if not all(os.path.exists(p) for p in [image_path, box_path, entities_path]):
            logging.warning(f"Skipping {filename}, missing one or more files.")
            continue
            
        # 1. Get image dimensions
        width, height = get_image_size(image_path)
        
        # 2. Load entities
        with open(entities_path, 'r', encoding='utf-8', errors='ignore') as f:
            entities = json.load(f)
            
        # 3. Parse OCR data
        words, boxes = parse_ocr_box(box_path)
        if not words:
            logging.warning(f"Skipping {filename}, no words found in OCR.")
            continue
        
        # 4. Normalize boxes
        normalized_boxes = [normalize_box(box, width, height) for box in boxes]
        
        # 5. Get BIO labels
        labels = get_bio_labels(words, entities)
        
        # 6. Assemble the record
        record = {
            'id': filename,
            'words': words,
            'bboxes': normalized_boxes,
            'ner_tags': labels
        }
        dataset.append(record)
        
    return dataset

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process training data
    train_dataset = create_dataset(split='train')
    train_output_path = os.path.join(OUTPUT_DIR, 'train.json')
    with open(train_output_path, 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=4)
    logging.info(f"Training data successfully saved to {train_output_path}")
    
    # Process test data
    test_dataset = create_dataset(split='test')
    test_output_path = os.path.join(OUTPUT_DIR, 'test.json')
    with open(test_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, indent=4)
    logging.info(f"Test data successfully saved to {test_output_path}")

    logging.info("Preprocessing complete!")
