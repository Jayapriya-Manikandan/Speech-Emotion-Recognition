
import numpy as np
import librosa


def extract_features(file_path):
    """
    Extract audio features from a speech recording.
    """

    try:
        # Load audio file
        audio, sample_rate = librosa.load(
            file_path,
            sr=22050,
            duration=3,
            offset=0.5
        )

        # MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )
        mfcc_features = np.mean(mfcc.T, axis=0)

        # Chroma features
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(
            S=stft,
            sr=sample_rate
        )
        chroma_features = np.mean(chroma.T, axis=0)

        # Mel Spectrogram features
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate
        )
        mel_features = np.mean(mel.T, axis=0)

        # Spectral Contrast features
        contrast = librosa.feature.spectral_contrast(
            S=stft,
            sr=sample_rate
        )
        contrast_features = np.mean(contrast.T, axis=0)

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio
        )
        zcr_features = np.mean(zcr.T, axis=0)

        # Combine all features
        features = np.hstack([
            mfcc_features,
            chroma_features,
            mel_features,
            contrast_features,
            zcr_features
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
