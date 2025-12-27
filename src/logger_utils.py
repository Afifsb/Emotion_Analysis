import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
log_filename = f"logs/model_usage_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_prediction(input_text, predicted_emotion, confidence):
    """Records every prediction made by the model for auditing."""
    logging.info(f"Input: '{input_text}' | Predicted: {predicted_emotion} | Confidence: {confidence:.4f}")