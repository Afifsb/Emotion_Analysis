# 🎭 Emotion Detection System using Transformers

## Overview
This project is a Deep Learning based system developed to detect emotions (Joy, Sadness, Anger, Fear, Love, Surprise) from textual comments and feedback.

## Tech Stack
- **Model:** DistilBERT (Transformers)
- **Frontend:** Streamlit
- **Backend:** Python 3.x
- **CI/CD:** GitHub Actions

## How to Run Locally
1. Clone the repository: `git clone <your-repo-link>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run src/app.py`

## Project Structure
- `src/`: Source code for training and frontend.
- `models/`: Saved model weights (locally).
- `logs/`: Performance logs and test results.