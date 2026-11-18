
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import tempfile

from src.predict_emotion import predict_emotion


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #161b22;
}

.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #ffffff;
}

.subtitle {
    color: #aab4c3;
    font-size: 16px;
}

.metric-card {
    background-color: #1c2333;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #30394a;
    text-align: center;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #60a5fa;
}

.metric-label {
    font-size: 14px;
    color: #aab4c3;
}

.section-title {
    font-size: 23px;
    font-weight: 600;
    color: #60a5fa;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🎙️ SER System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Emotion Detection",
        "Emotion Analysis",
        "About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Speech Emotion Recognition using Machine Learning"
)


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":

    st.markdown(
        '<div class="main-title">Speech Emotion Recognition</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'An intelligent system that identifies human emotions '
        'from speech audio using machine learning.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Welcome to the SER System")

    st.write(
        """
        Speech Emotion Recognition (SER) is an AI-based technology
        that analyzes speech signals and identifies the emotional
        state expressed through a person's voice.

        This application uses audio feature extraction and
        machine learning techniques to recognize emotions
        from uploaded speech recordings.
        """
    )

    st.markdown("### Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎧 Audio Processing")
        st.write(
            "Upload and listen to speech recordings "
            "before performing emotion detection."
        )

        st.markdown("#### 🤖 Machine Learning")
        st.write(
            "Uses a trained Gradient Boosting model "
            "to classify speech emotions."
        )

    with col2:
        st.markdown("#### 📊 Emotion Analysis")
        st.write(
            "View prediction confidence and emotion "
            "scores through interactive visualizations."
        )

        st.markdown("#### 🎯 Eight Emotion Classes")
        st.write(
            "Recognizes Neutral, Calm, Happy, Sad, Angry, "
            "Fearful, Disgust and Surprised."
        )

    st.markdown("---")

    st.info(
        "Navigate to Emotion Detection to upload an audio "
        "recording and analyze its emotional expression."
    )


# --------------------------------------------------
# EMOTION DETECTION PAGE
# --------------------------------------------------

elif page == "Emotion Detection":

    st.title("🎧 Emotion Detection")

    st.write(
        "Upload a speech recording in WAV, MP3, or other "
        "supported audio formats to detect its emotion."
    )

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "ogg", "flac", "m4a"]
    )

    if uploaded_file is not None:

        st.success("Audio uploaded successfully!")

        st.audio(uploaded_file)

        if st.button(
            "Analyze Emotion",
            type="primary",
            use_container_width=True
        ):

            temp_path = None

            try:
                with st.spinner(
                    "Analyzing audio and detecting emotion..."
                ):

                    suffix = os.path.splitext(
                        uploaded_file.name
                    )[1]

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:

                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    result = predict_emotion(temp_path)

                st.markdown("---")

                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">
                                {result['emotion']}
                            </div>
                            <div class="metric-label">
                                Detected Emotion
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-value">
                                {result['confidence']:.2f}%
                            </div>
                            <div class="metric-label">
                                Model Confidence
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.caption(
                    "Model confidence is the classifier's highest "
                    "predicted score, not a guarantee of correctness."
                )

                st.markdown("### Emotion Scores")

                scores_df = pd.DataFrame({
                    "Emotion": list(
                        result["emotion_scores"].keys()
                    ),
                    "Score": list(
                        result["emotion_scores"].values()
                    )
                })

                scores_df = scores_df.sort_values(
                    by="Score",
                    ascending=False
                )

                fig = px.bar(
                    scores_df,
                    x="Emotion",
                    y="Score",
                    color="Score",
                    color_continuous_scale="blues",
                    title="Emotion Prediction Scores"
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    font_color="white",
                    yaxis_title="Score (%)",
                    xaxis_title="Emotion",
                    coloraxis_showscale=False
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Unable to analyze the audio: {str(e)}"
                )

            finally:

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# --------------------------------------------------
# EMOTION ANALYSIS PAGE
# --------------------------------------------------

elif page == "Emotion Analysis":

    st.title("📊 Emotion Analysis")

    st.write(
        "Explore the eight emotion categories supported "
        "by the Speech Emotion Recognition model."
    )

    emotion_descriptions = {
        "Neutral": "A balanced and emotionally neutral expression.",
        "Calm": "A relaxed and peaceful vocal expression.",
        "Happy": "A positive and cheerful vocal expression.",
        "Sad": "A low-energy and sorrowful vocal expression.",
        "Angry": "A strong and intense vocal expression.",
        "Fearful": "A nervous or frightened vocal expression.",
        "Disgust": "An expression associated with dislike.",
        "Surprised": "An expression of sudden unexpected emotion."
    }

    for emotion, description in emotion_descriptions.items():

        with st.expander(emotion):
            st.write(description)

    st.markdown("---")

    st.subheader("Supported Emotion Categories")

    emotion_df = pd.DataFrame({
        "Emotion": list(emotion_descriptions.keys()),
        "Category Count": [1] * 8
    })

    fig = px.pie(
        emotion_df,
        names="Emotion",
        values="Category Count",
        hole=0.45,
        title="Eight Recognized Emotion Categories"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        font_color="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About":

    st.title("ℹ️ About the Project")

    st.write(
        """
        **Speech Emotion Recognition using Machine Learning**

        This project identifies emotional expressions in speech
        recordings by extracting audio features and applying
        machine learning classification techniques.
        """
    )

    st.markdown("### Dataset")

    st.write(
        """
        The project uses the Ryerson Audio-Visual Database of
        Emotional Speech and Song (RAVDESS) dataset.

        The dataset contains speech recordings representing
        eight different emotional categories.
        """
    )

    st.markdown("### Audio Features")

    st.write(
        """
        - Mel-Frequency Cepstral Coefficients (MFCC)
        - Chroma Features
        - Mel Spectrogram
        - Spectral Contrast
        - Zero Crossing Rate
        """
    )

    st.markdown("### Machine Learning Algorithms")

    st.write(
        """
        - Logistic Regression
        - Support Vector Machine (SVM)
        - Random Forest
        - Gradient Boosting
        """
    )

    st.markdown("### Selected Model")

    st.success("Gradient Boosting")

    st.write(
        """
        Gradient Boosting was selected based on its performance
        during model evaluation, achieving approximately 59.03%
        accuracy on the held-out test set.
        """
    )

    st.markdown("---")

    st.caption(
        "Speech Emotion Recognition | Machine Learning Project"
    )
