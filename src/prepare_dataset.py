
import os
import pandas as pd
from src.feature_extraction import extract_features


# Emotion labels in the RAVDESS dataset
EMOTIONS = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}


def prepare_dataset(data_path="data"):

    features_list = []
    labels_list = []

    print("\nStarting Speech Emotion Recognition Dataset Preparation")
    print("=" * 60)

    # Loop through all actor folders
    for actor in sorted(os.listdir(data_path)):

        actor_path = os.path.join(data_path, actor)

        if not os.path.isdir(actor_path):
            continue

        print(f"\nProcessing {actor}...")

        for file_name in os.listdir(actor_path):

            if not file_name.lower().endswith(".wav"):
                continue

            try:
                # Extract emotion code from filename
                parts = file_name.split("-")

                if len(parts) != 7:
                    continue

                emotion_code = parts[2]

                if emotion_code not in EMOTIONS:
                    continue

                emotion = EMOTIONS[emotion_code]

                file_path = os.path.join(actor_path, file_name)

                # Extract audio features
                features = extract_features(file_path)

                if features is not None:
                    features_list.append(features)
                    labels_list.append(emotion)

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        print(f"Completed {actor}")

    # Create DataFrame
    if not features_list:
        raise ValueError(
            "No audio features extracted. Check the dataset path."
        )

    feature_columns = [
        f"feature_{i}"
        for i in range(len(features_list[0]))
    ]

    df = pd.DataFrame(
        features_list,
        columns=feature_columns
    )

    df["emotion"] = labels_list

    # Create processed data folder
    os.makedirs("data/processed", exist_ok=True)

    # Save dataset
    output_path = "data/processed/speech_emotion_dataset.csv"

    df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)

    print(f"Total audio files processed: {len(df)}")
    print(f"Total features: {len(feature_columns)}")

    print("\nEmotion Distribution:")
    print(df["emotion"].value_counts())

    print(f"\nDataset saved at: {output_path}")

    return df


if __name__ == "__main__":
    prepare_dataset()
