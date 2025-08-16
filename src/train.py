import os
import json
import torch
from datasets import load_dataset, Features, Sequence, ClassLabel, Value, Array2D
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
# FIX: Import accuracy_score directly
from seqeval.metrics import classification_report, accuracy_score

# --- Configuration ---
PROCESSED_DATA_DIR = './data/processed/'
MODEL_OUTPUT_DIR = './models/layoutlm-sroie'
MODEL_CHECKPOINT = "microsoft/layoutlm-base-uncased"

# --- 1. Load Processed Datasets ---
# Load the JSON files we created in the preprocessing step
data_files = {
    "train": os.path.join(PROCESSED_DATA_DIR, "train.json"),
    "test": os.path.join(PROCESSED_DATA_DIR, "test.json"),
}

# Define the features of our dataset to ensure it's loaded correctly
features = Features({
    'id': Value('string'),
    'words': Sequence(Value('string')),
    'bboxes': Sequence(Sequence(Value('int64'))),
    'ner_tags': Sequence(Value('string')),
})

# Load the datasets
raw_datasets = load_dataset("json", data_files=data_files, features=features)

# --- 2. Prepare Labels and Tokenizer ---
# Get the list of unique NER tags from the training data
labels_list = list(set(tag for example in raw_datasets['train'] for tag in example['ner_tags']))
labels_list.sort() # Sort for consistency

# Create mappings between labels and integer IDs
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for i, label in enumerate(labels_list)}
num_labels = len(labels_list)

print("Label Mappings:", label2id)

# Load the tokenizer for the pre-trained model
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

# --- 3. Tokenize and Align Data ---
def tokenize_and_align_labels(examples):
    """
    Tokenizes words and aligns labels and bounding boxes.
    """
    tokenized_inputs = tokenizer(
        examples["words"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=512
    )

    labels = []
    bboxes = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        bbox_inputs = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens like [CLS], [SEP] get a special label and box
                label_ids.append(-100)
                bbox_inputs.append([0, 0, 0, 0])
            elif word_idx != previous_word_idx:
                # First token of a new word
                label_ids.append(label2id[label[word_idx]])
                bbox_inputs.append(examples["bboxes"][i][word_idx])
            else:
                # Subsequent tokens of the same word
                label_ids.append(-100) # Only label the first token of a word
                bbox_inputs.append(examples["bboxes"][i][word_idx])
            previous_word_idx = word_idx
        
        labels.append(label_ids)
        bboxes.append(bbox_inputs)

    tokenized_inputs["labels"] = labels
    tokenized_inputs["bbox"] = bboxes
    return tokenized_inputs

# Apply the tokenization function to our datasets
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# --- 4. Configure Model and Trainer ---
# Load the pre-trained LayoutLM model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    report_to="none",
)

# --- 5. Define Evaluation Metrics ---
def compute_metrics(p):
    """
    Computes precision, recall, F1, and accuracy for the evaluation set.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_predictions, output_dict=True)
    
    # Extract key metrics
    results = {
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        # FIX: Calculate accuracy using the dedicated function for robustness
        "accuracy": accuracy_score(true_labels, true_predictions),
    }
    return results

# --- 6. Train the Model ---
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start the training process
print("Starting model training...")
trainer.train()

# Save the final best model
trainer.save_model(os.path.join(MODEL_OUTPUT_DIR, "best_model"))
print("Training complete. Best model saved.")
