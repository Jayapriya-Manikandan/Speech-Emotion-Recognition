
import os
import joblib
import numpy as np

from src.feature_extraction import extract_features


MODEL_PATH = "models/speech_emotion_model.pkl"


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Please train the model first."
        )

    return joblib.load(MODEL_PATH)


def predict_emotion(audio_path):

    if not os.path.exists(audio_path):
        raise FileNotFoundError("Audio file not found.")

    # Load trained model
    model = load_model()

    # Extract audio features
    features = extract_features(audio_path)

    if features is None:
        raise ValueError("Unable to extract features from the audio file.")

    # Prepare features for prediction
    features = np.array(features).reshape(1, -1)

    # Predict emotion
    prediction = model.predict(features)[0]

    # Get model confidence scores
    probabilities = model.predict_proba(features)[0]

    confidence = float(np.max(probabilities)) * 100

    # Get probability for each emotion
    emotion_scores = {
        emotion: float(probability) * 100
        for emotion, probability in zip(model.classes_, probabilities)
    }

    return {
        "emotion": prediction,
        "confidence": confidence,
        "emotion_scores": emotion_scores
    }
