# 🧾 Smart Document Parser: Receipt OCR & Information Extraction

This project is an end-to-end pipeline for automatically parsing scanned receipts. It uses Optical Character Recognition (OCR) to extract text and a fine-tuned LayoutLM model to understand the document's layout and extract key entities such as the company name, date, address, and total amount. The entire project is wrapped in a user-friendly web application built with Streamlit.



---

## ✨ Features

- **OCR Integration:** Uses Tesseract to perform OCR on uploaded receipt images.
- **Deep Learning Model:** Leverages a fine-tuned `microsoft/layoutlm-base-uncased` model for state-of-the-art information extraction.
- **High Accuracy:** Achieves an **F1-score of ~89%** and **token-level accuracy of ~97%** on the SROIE test dataset.
- **Interactive UI:** A simple and intuitive web interface built with Streamlit for uploading receipts and viewing results.
- **End-to-End Pipeline:** Covers the full ML lifecycle from data preprocessing and model training to inference and application deployment.

---

## 📂 Project Structure

```bash
receipt-parser/
├── data/
│   ├── raw/                # Original SROIE dataset
│   ├── processed/          # Processed data for model training
│   └── output/             # (Optional) Saved extraction results
│
├── models/
│   └── layoutlm-sroie/     # Saved fine-tuned model weights
│
├── notebooks/              # Jupyter notebooks for exploration and baselining
│
├── src/
│   ├── preprocess.py       # Script to process raw data
│   ├── train.py            # Script to fine-tune the LayoutLM model
│   └── app.py              # The Streamlit web application
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup and Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd receipt-parser
```

### 2. Install Tesseract OCR

The project uses Tesseract for OCR.

- **Windows:** Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Note the installation path.
- **macOS:**
  ```bash
  brew install tesseract
  ```
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get install tesseract-ocr
  ```

> 📌 In `src/app.py`, set the Tesseract path if needed:
```python
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 3. Create a Virtual Environment and Install Dependencies

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the **SROIE 2019** dataset (e.g., from Kaggle) and place it under:

```
data/raw/SROIE2019/
```

---

## 🚀 Usage

Once setup is complete, follow these steps:

### 1. Preprocess the Data

Convert raw OCR and annotations into LayoutLM-compatible format:

```bash
python src/preprocess.py
```

Creates:
```
data/processed/train.json
data/processed/test.json
```

### 2. Train the Model

Fine-tune the LayoutLM model on the processed data:

```bash
python src/train.py
```

The best-performing model will be saved to:

```
models/layoutlm-sroie/best_model/
```

> ⚠️ This step requires a GPU for efficient training.

### 3. Launch the Streamlit Web App

Start the app to upload and parse receipt images:

```bash
streamlit run src/app.py
```

Your default browser will open with the app interface.

---

## 📈 Model Performance

The LayoutLM model was fine-tuned for 5 epochs. Final performance on the SROIE test set:

| Metric     | Score   |
|------------|---------|
| **F1-Score**   | 0.8888  |
| **Precision**  | 0.8547  |
| **Recall**     | 0.9296  |
| **Accuracy**   | 0.9687  |

This shows strong capability in extracting structured information from unseen receipts.

---

## 🔮 Future Improvements

- **Docker Container:** Package the app using Docker for reproducible deployment.
- **Cloud Hosting:** Deploy the app to platforms like Streamlit Cloud, Hugging Face Spaces, or AWS.
- **Alternative OCR:** Evaluate OCR quality using alternatives like [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).
- **Line-Item Extraction:** Extend entity recognition to extract:
  - Product names
  - Quantities
  - Prices
  - Subtotals
- **Multilingual Support:** Extend the model for multilingual receipts and international formats.

---

## 🧠 Credits

- OCR Engine: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- Document Model: [LayoutLM (Microsoft)](https://github.com/microsoft/unilm/tree/master/layoutlm)
- UI Framework: [Streamlit](https://streamlit.io/)
- Dataset: [SROIE 2019](https://rrc.cvc.uab.es/?ch=13)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
