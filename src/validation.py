import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def validate():
    print("Step 3: Starting Model Validation...")
    
    # 1. Load the model we just trained
    model_path = "models/emotion_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # 2. Load the data and take a small test sample (the last 100 rows)
    df = pd.read_csv("data/processed_data.csv").tail(100)
    texts = df['cleaned_text'].tolist()
    labels = df['label_encoded'].tolist()
    
    # 3. Predictions
    print("Testing model on 100 unseen sentences...")
    preds = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            preds.append(prediction)
            
    # 4. Calculate Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    print("\n" + "="*30)
    print(f"ACCURACY:  {acc:.4f}")
    print(f"F1-SCORE:  {f1:.4f}")
    print("="*30)
    
    # 5. Confusion Matrix
    emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotion_labels, yticklabels=emotion_labels, cmap='Purples')
    plt.title('Confusion Matrix - Emotion Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save results
    plt.savefig('models/confusion_matrix.png')
    print("\nConfusion Matrix saved to models/confusion_matrix.png")

if __name__ == "__main__":
    validate()