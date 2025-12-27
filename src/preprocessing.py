import os  # <--- Added this line
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

# Download necessary NLTK data
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Slang dictionary for normalization (Specific to project brief)
SLANG_MAP = {
    "nxt": "next",
    "2mrw": "tomorrow",
    "r u gud": "are you good",
    "u": "you",
    "r": "are",
    "gud": "good",
    "txt": "text",
    "sry": "sorry"
}

def text_cleaner(text):
    text = str(text).lower()
    for slang, formal in SLANG_MAP.items():
        text = re.sub(r'\b' + slang + r'\b', formal, text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def run_preprocessing():
    print("Step 1.2: Preprocessing and Normalization...")
    
    if not os.path.exists('data/raw_data.csv'):
        print("Error: raw_data.csv not found. Run data_loader.py first.")
        return

    df = pd.read_csv('data/raw_data.csv')
    df['cleaned_text'] = df['text'].apply(text_cleaner)
    
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['emotion'])
    
    # Save the processed data
    df.to_csv("data/processed_data.csv", index=False)
    
    print("Preprocessing Complete!")
    print(f"Sample mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print("Cleaned data saved to data/processed_data.csv")

if __name__ == "__main__":
    run_preprocessing()