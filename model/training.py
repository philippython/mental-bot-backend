import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import os


model_path = "model/trained_model"
os.environ["WANDB_DISABLED"] = "true"

# Check for CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA Device: {torch.cuda.get_device_name(0)}")

# ========== Load Datasets ==========
print("Loading datasets...")
try:
    df_text = pd.read_csv("data/dementia.csv")
    df_country = pd.read_csv("data/global_mental_health_stats.csv")
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error loading datasets: {e}")

# ========== Clean and Encode ==========
print("Cleaning and encoding data...")
try:
    df_text.dropna(subset=['statement', 'status'], inplace=True)
    df_text = df_text[df_text['status'].isin(['Anxiety', 'Depression', 'Bipolar', 'Suicidal', 'Normal'])]
    print(f"Data after cleaning: {df_text.shape[0]} rows")

    label_enc = LabelEncoder()
    df_text['label'] = label_enc.fit_transform(df_text['status'])
    num_labels = len(label_enc.classes_)
    print(f"Labels encoded. Number of unique labels: {num_labels}")
except Exception as e:
    print(f"Error during cleaning and encoding: {e}")

# ========== Train/Test Split ==========
print("Splitting data into train and test sets...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        df_text['statement'].tolist(),
        df_text['label'].tolist(),
        test_size=0.2,
        stratify=df_text['label'],
        random_state=42
    )
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
except Exception as e:
    print(f"Error during train/test split: {e}")

# ========== Define PyTorch Dataset ==========
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# ========== Load Tokenizer and Model ==========
print("Loading tokenizer and model...")
try:
    model_name = "distilbert-base-uncased"
    trained_model_path = "./trained_model"

    if os.path.exists(trained_model_path):
        print(f"Loading pre-trained model and tokenizer from {trained_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(trained_model_path, num_labels=num_labels)
        print("Pre-trained model and tokenizer loaded successfully.")
    else:
        print(f"No pre-trained model found at {trained_model_path}. Initializing new model.")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        print("Tokenizer and model loaded successfully.")

except Exception as e:
    print(f"Error loading tokenizer and model: {e}")

# ========== Create Datasets ==========
print("Creating datasets for training and testing...")
try:
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)
    print(f"Training dataset size: {len(train_dataset)}, Testing dataset size: {len(test_dataset)}")
except Exception as e:
    print(f"Error creating datasets: {e}")

# ========== Training Arguments ==========
print("Setting up training arguments...")
try:
    training_args = TrainingArguments(
        output_dir="model/results",
        logging_dir="./logs",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        learning_rate=2e-5
    )
    print("Training arguments set up successfully.")
except Exception as e:
    print(f"Error setting up training arguments: {e}")

# ========== Trainer ==========
print("Setting up trainer...")
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    print("Trainer setup successful.")
except Exception as e:
    print(f"Error setting up trainer: {e}")

# ========== Training ==========
print("Training model...")
try:
    if Path(model_path).exists():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        trainer.train()
        print("Training completed successfully.")

        # ========== Save Trained Model and Tokenizer ==========
        output_model_path = "./trained_model"
        trainer.save_model(output_model_path)
        tokenizer.save_pretrained(output_model_path)
        print(f"Trained model and tokenizer saved to {output_model_path}")
    else:
        print("Pre-trained model found. Skipping training.")

except Exception as e:
    print(f"Error during training: {e}")
