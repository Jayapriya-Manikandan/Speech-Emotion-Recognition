
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# Dataset and model paths
DATA_PATH = "data/processed/speech_emotion_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "speech_emotion_model.pkl")


def train_models():

    print("\nSPEECH EMOTION RECOGNITION - MODEL TRAINING")
    print("=" * 60)

    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["emotion"])
    y = df["emotion"]

    print(f"Total samples: {len(df)}")
    print(f"Total features: {X.shape[1]}")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Machine learning algorithms
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                probability=True,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    results = []
    best_model = None
    best_model_name = None
    best_f1 = -1

    # Train and evaluate each model
    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name

    # Display model comparison
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Detailed evaluation of the best model
    best_predictions = best_model.predict(X_test)

    print("\nBEST MODEL:", best_model_name)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        best_predictions,
        zero_division=0
    ))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(
        y_test,
        best_predictions,
        labels=best_model.classes_
    ))

    # Save the best trained model
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best model: {best_model_name}")
    print(f"Best macro F1-score: {best_f1:.4f}")
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train_models()
