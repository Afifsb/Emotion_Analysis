import os
import sys

# DEBUG: Print immediately to see if the script even starts
print("--- Script Process Started ---")

try:
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    print("--- Libraries Imported Successfully ---")
except ImportError as e:
    print(f"--- Error: Missing library: {e} ---")
    sys.exit(1)

def train():
    print("Step 2: Starting Model Training (Fine-tuning DistilBERT)...")

    # 1. Check if data exists
    data_path = 'data/processed_data.csv'
    if not os.path.exists(data_path):
        print(f"--- Error: {data_path} not found! Did you run preprocessing.py? ---")
        return
    
    print(f"--- Loading data from {data_path}... ---")
    df = pd.read_csv(data_path)
    print(f"--- Found {len(df)} rows. ---")
    
    # 2. Convert to HuggingFace Dataset format
    dataset = Dataset.from_pandas(df[['cleaned_text', 'label_encoded']].head(2000))
    dataset = dataset.rename_column("label_encoded", "label")
    dataset = dataset.rename_column("cleaned_text", "text")

    # 3. Load Tokenizer and Model
    model_name = "distilbert-base-uncased"
    print(f"--- Downloading/Loading Tokenizer: {model_name}... ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    print("--- Tokenizing data... ---")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 4. Model Setup
    print("--- Loading Pre-trained Model... ---")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

    # 5. Training Settings
    training_args = TrainingArguments(
        output_dir="./models/results",
        num_train_epochs=1,
        max_steps=100,
        per_device_train_batch_size=4, # Reduced for better local performance
        weight_decay=0.01,
        logging_dir='./logs',
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # 6. Run Training
    print("--- Starting Trainer.train() (This will take time)... ---")
    trainer.train()

    # 7. Save the model
    save_path = "models/emotion_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"--- Step 2 Complete: Model saved in {save_path} ---")

if __name__ == "__main__":
    train()