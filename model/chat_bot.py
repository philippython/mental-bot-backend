import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# ========== Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model/trained_model"

# ========== Load Label Encoder ==========
label_enc = LabelEncoder()
label_enc.classes_ = np.array(['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Suicidal'])

# ========== Lazy Loading ==========
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at '{model_path}'. Please train the model first.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

# ========== Prediction Function ==========
def classify_statement(statement: str) -> str:
    load_model()  # Load only when called
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return label_enc.inverse_transform([predicted_label])[0]
