import os
import pandas as pd
from datasets import load_dataset

def fetch_and_save_data():
    print("Step 1.1: Sourcing Data...")
    
    # We use 'dair-ai/emotion' - a dataset of 16,000+ social media-style comments.
    # Note: We are using a subset of 2,000 rows to optimize training speed for local CPU 
    # while maintaining a balanced representation of emotions: joy, sadness, anger, fear, love, surprise.
    dataset = load_dataset("dair-ai/emotion")
    
    # Convert to DataFrames
    train_df = pd.DataFrame(dataset['train'])
    
    # Map the numeric labels to actual emotion names
    label_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    train_df['emotion'] = train_df['label'].map(label_map)
    
    # Create data directory if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Save raw data
    train_df.to_csv("data/raw_data.csv", index=False)
    print(f"Data Sourced! Saved {len(train_df)} rows to data/raw_data.csv")

if __name__ == "__main__":
    fetch_and_save_data()