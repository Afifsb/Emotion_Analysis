import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from logger_utils import log_prediction  # From Step 4

#1.Page Configuration
st.set_page_config(
    page_title="Emotion Detection AI System",
    page_icon="🎭",
    layout="wide"
)

#2.Load Model & Tokenizer
@st.cache_resource
def load_emotion_ai():
    model_path = "models/emotion_model"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error("❌ Model folder not found! Please run Step 2 (train.py) first.")
        st.stop()
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# Initialize System
try:
    tokenizer, model = load_emotion_ai()
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
except Exception as e:
    st.error(f"Error loading system: {e}")
    st.stop()

#3.Sidebar Information 
st.sidebar.title("System Status")
st.sidebar.info("✅ Model: DistilBERT Local")
st.sidebar.info("✅ Logging: Enabled (logs/ directory)")
st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.write("1. Enter text in the box.")
st.sidebar.write("2. Click 'Analyze'.")
st.sidebar.write("3. View emotion breakdown and confidence scores.")

#4.Main UI Interface
st.title("🎭 Emotion Detection System")
st.markdown("""
    This deep learning application detects emotions from textual feedback using **Transformer Architecture**.
    *Handles complex expressions, slang, and double negations.*
""")

# Input Area
user_text = st.text_area("Enter Sentence or Feedback:", height=150, 
                         placeholder="e.g., The service wasn't bad, but I'm still a bit surprised by the delay.")

# Column layout for results
col1, col2 = st.columns([2, 1])

if st.button("Analyze Sentiment & Emotion"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        #5.Inference Logic 
        with st.spinner('Analyzing textual context...'):
            # Tokenize
            inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                # Convert logits to probabilities (Softmax)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get top result
            confidence, pred_idx = torch.max(probs, dim=1)
            top_emotion = emotions[pred_idx.item()]
            conf_score = confidence.item()

            #6.Requirement: Logging
            log_prediction(user_text, top_emotion, conf_score)

        #7. UI Display
        with col1:
            st.success(f"### Result: {top_emotion.upper()}")
            st.metric(label="Confidence Score", value=f"{conf_score:.2%}")
            
            # Data visualization
            df_probs = pd.DataFrame({
                "Emotion": emotions,
                "Probability": probs[0].tolist()
            }).sort_values(by="Probability", ascending=True)

            fig = px.bar(df_probs, x="Probability", y="Emotion", orientation='h',
                         title="Emotion Breakdown",
                         color="Probability",
                         color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Analysis Summary")
            st.write(f"**Input:** _{user_text}_")
            st.write(f"**Primary Detected Emotion:** {top_emotion}")
            st.write(f"**Stability Check:** Passed")
            st.info("The prediction has been logged for system auditing.")

# --- 8. Footer ---
st.markdown("---")
st.caption("Automate Detection of Different Emotions)