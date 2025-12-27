import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from logger_utils import log_prediction

def run_test_suite():
    print("Step 4: Running Model Test Suite...")
    
    # 1. Load Model
    model_path = "models/emotion_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    # 2. Define Test Cases (Sarcasm, Slang, Abbreviations, Double Negations)
    test_data = [
        {"desc": "Simple Positive", "text": "I love this product!"},
        {"desc": "Abbreviation/Slang", "text": "r u gud? see u nxt week"},
        {"desc": "Double Negation", "text": "I am not unhappy with this."},
        {"desc": "Sarcasm Test", "text": "Oh great, another bug in the system. Exactly what I needed."},
        {"desc": "Complex Feedback", "text": "The delivery was late, but the food was delicious."}
    ]

    results = []

    # 3. Execute Testing
    for case in test_data:
        inputs = tokenizer(case['text'], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        label = emotions[pred_idx.item()]
        score = conf.item()

        # Log the result (Requirement: Logging)
        log_prediction(case['text'], label, score)

        results.append({
            "Description": case['desc'],
            "Input": case['text'],
            "Predicted Label": label,
            "Confidence Score": round(score, 4)
        })

    # 4. Save to CSV for the Final Report
    df_results = pd.DataFrame(results)
    df_results.to_csv("logs/test_results_report.csv", index=False)
    print("\nTest Suite Complete. Results saved to 'logs/test_results_report.csv'")
    print(df_results)

if __name__ == "__main__":
    run_test_suite()