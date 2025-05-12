import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ========== Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model/trained_model"

# ========== Load Model & Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

# ========== Load Label Encoder ==========
label_enc = LabelEncoder()
label_enc.classes_ = np.array(['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Suicidal'])

# ========== Prediction Function ==========
def classify_statement(statement):
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return label_enc.inverse_transform([predicted_label])[0]
