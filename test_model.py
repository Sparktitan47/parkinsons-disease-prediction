import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/parkinsons.data")

print(f"Dataset loaded successfully.")
print(f"Number of records: {len(df)}")


# ==========================================
# 2. PREPARE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["name", "status"])
y = df["status"]


# ==========================================
# 3. LOAD TRAINED MODELS
# ==========================================

print("Loading trained models...")

gnb = joblib.load("models/gnb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
rnn = tf.keras.models.load_model("models/rnn_model.h5")

print("Models loaded successfully.")


# ==========================================
# 4. SCALE DATA
# ==========================================

X_scaled = scaler.transform(X)


# ==========================================
# 5. GNB PREDICTIONS
# ==========================================

print("Generating GNB predictions...")

gnb_probabilities = gnb.predict_proba(X_scaled)[:, 1]

gnb_predictions = (
    gnb_probabilities >= 0.5
).astype(int)


# ==========================================
# 6. RNN PREDICTIONS
# ==========================================

print("Generating RNN predictions...")

rnn_input = X_scaled.reshape(
    X_scaled.shape[0],
    1,
    X_scaled.shape[1]
)

rnn_probabilities = rnn.predict(
    rnn_input,
    verbose=0
).flatten()

rnn_predictions = (
    rnn_probabilities >= 0.5
).astype(int)


# ==========================================
# 7. WEIGHTED ENSEMBLE
# ==========================================

gnb_weight = 0.4
rnn_weight = 0.6

ensemble_probabilities = (
    gnb_weight * gnb_probabilities
    +
    rnn_weight * rnn_probabilities
)

ensemble_predictions = (
    ensemble_probabilities >= 0.5
).astype(int)


# ==========================================
# 8. CREATE RESULTS TABLE
# ==========================================

results = pd.DataFrame({

    "name": df["name"],

    "actual_status": y,

    "gnb_probability": gnb_probabilities,

    "gnb_prediction": gnb_predictions,

    "rnn_probability": rnn_probabilities,

    "rnn_prediction": rnn_predictions,

    "ensemble_probability": ensemble_probabilities,

    "ensemble_prediction": ensemble_predictions
})


# ==========================================
# 9. DISPLAY FIRST 10 RECORDS
# ==========================================

print("\n")
print("=" * 80)
print("FIRST 10 PREDICTIONS")
print("=" * 80)

display_results = results.head(10).copy()

display_results["gnb_probability"] = (
    display_results["gnb_probability"] * 100
).round(6)

display_results["rnn_probability"] = (
    display_results["rnn_probability"] * 100
).round(6)

display_results["ensemble_probability"] = (
    display_results["ensemble_probability"] * 100
).round(6)

print(display_results.to_string(index=False))


# ==========================================
# 10. MODEL PERFORMANCE
# ==========================================

print("\n")
print("=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)


def show_metrics(model_name, actual, predicted):

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        zero_division=0
    )

    cm = confusion_matrix(
        actual,
        predicted
    )

    print(f"\n{model_name}")
    print("-" * 40)

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)


show_metrics(
    "Gaussian Naive Bayes",
    y,
    gnb_predictions
)

show_metrics(
    "Recurrent Neural Network",
    y,
    rnn_predictions
)

show_metrics(
    "Weighted Ensemble",
    y,
    ensemble_predictions
)


# ==========================================
# 11. PROBABILITY ANALYSIS
# ==========================================

print("\n")
print("=" * 80)
print("PROBABILITY ANALYSIS")
print("=" * 80)


print(
    f"\nGNB probability range: "
    f"{gnb_probabilities.min():.8f} "
    f"to "
    f"{gnb_probabilities.max():.8f}"
)

print(
    f"RNN probability range: "
    f"{rnn_probabilities.min():.8f} "
    f"to "
    f"{rnn_probabilities.max():.8f}"
)

print(
    f"Ensemble probability range: "
    f"{ensemble_probabilities.min():.8f} "
    f"to "
    f"{ensemble_probabilities.max():.8f}"
)


# ==========================================
# 12. EXTREME PROBABILITY COUNT
# ==========================================

print("\n")
print("=" * 80)
print("EXTREME PROBABILITY CHECK")
print("=" * 80)


gnb_extreme = np.sum(
    (gnb_probabilities >= 0.99)
    |
    (gnb_probabilities <= 0.01)
)

rnn_extreme = np.sum(
    (rnn_probabilities >= 0.99)
    |
    (rnn_probabilities <= 0.01)
)

ensemble_extreme = np.sum(
    (ensemble_probabilities >= 0.99)
    |
    (ensemble_probabilities <= 0.01)
)


print(
    f"GNB extreme predictions: "
    f"{gnb_extreme}/{len(df)}"
)

print(
    f"RNN extreme predictions: "
    f"{rnn_extreme}/{len(df)}"
)

print(
    f"Ensemble extreme predictions: "
    f"{ensemble_extreme}/{len(df)}"
)


# ==========================================
# 13. SAVE RESULTS
# ==========================================

results.to_csv(
    "prediction_results.csv",
    index=False
)

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    "\nComplete results saved to:"
)

print(
    "prediction_results.csv"
)