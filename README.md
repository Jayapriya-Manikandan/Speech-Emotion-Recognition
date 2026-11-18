
# Speech Emotion Recognition 🎙️

## About the Project

This project is about recognizing human emotions through speech using Python and Machine Learning.

I developed this project to explore how a person's voice can be analyzed to identify different emotional expressions. The system takes an audio recording as input, extracts important audio features, and predicts the emotion using a trained machine learning model.

## What This Project Does

- Allows users to upload speech audio files.
- Analyzes the uploaded audio.
- Predicts the emotion expressed in the speech.
- Displays the detected emotion and model confidence.
- Shows a visual comparison of scores for different emotions.

## Emotions Recognized

The system can recognize eight emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Dataset Used

I used the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset for this project.

A total of 1,440 speech recordings were processed, and 188 audio features were extracted from each recording.

## Technologies Used

- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Machine Learning Models

I experimented with four machine learning algorithms:

- Logistic Regression
- Support Vector Machine
- Random Forest
- Gradient Boosting

After comparing their performance, Gradient Boosting was selected as the final model with 59.03% test accuracy.

## My Learning Experience

Through this project, I gained practical experience in audio processing, feature extraction, machine learning model training, model evaluation, and building interactive applications using Streamlit.

## Running the Project

Install the required libraries:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

---

**Speech Emotion Recognition | Machine Learning Project**
