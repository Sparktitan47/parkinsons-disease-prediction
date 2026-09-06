import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# SETTINGS
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

GNB_WEIGHT_OPTIONS = np.arange(0.0, 1.01, 0.05)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv("data/parkinsons.data")

df = df.drop(columns=["name"])

X = df.drop(columns=["status"])
y = df["status"]

print(f"Dataset shape: {df.shape}")
print(f"Number of features: {X.shape[1]}")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: 80% temporary data, 20% final unseen test data
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

# Then: divide the remaining 80% into
# 80% training and 20% validation
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    stratify=y_temp,
    random_state=SEED
)

print("\nData split:")
print(f"Training:   {len(X_train_raw)} samples")
print(f"Validation: {len(X_val_raw)} samples")
print(f"Test:       {len(X_test)} samples")


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

# IMPORTANT:
# Fit the scaler ONLY on training data
X_train = scaler.fit_transform(X_train_raw)

# Transform validation and test using the same scaler
X_val = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# GAUSSIAN NAIVE BAYES
# ============================================================

print("\nTraining Gaussian Naive Bayes...")

gnb = GaussianNB()

gnb.fit(X_train, y_train)

gnb_val_probability = gnb.predict_proba(X_val)[:, 1]
gnb_test_probability = gnb.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# RECURRENT NEURAL NETWORK
# ============================================================

print("Training Recurrent Neural Network...")

X_train_rnn = X_train.reshape(
    X_train.shape[0],
    1,
    X_train.shape[1]
)

X_val_rnn = X_val.reshape(
    X_val.shape[0],
    1,
    X_val.shape[1]
)

X_test_rnn = X_test_scaled.reshape(
    X_test_scaled.shape[0],
    1,
    X_test_scaled.shape[1]
)

rnn = tf.keras.Sequential([
    tf.keras.Input(
        shape=(1, X_train.shape[1])
    ),

    tf.keras.layers.SimpleRNN(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

rnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

rnn.fit(
    X_train_rnn,
    y_train,
    epochs=50,
    batch_size=16,
    verbose=0
)

rnn_val_probability = rnn.predict(
    X_val_rnn,
    verbose=0
).flatten()

rnn_test_probability = rnn.predict(
    X_test_rnn,
    verbose=0
).flatten()


# ============================================================
# VALIDATION-BASED ENSEMBLE WEIGHT SEARCH
# ============================================================

print("\nSearching for best ensemble weight...")

best_weight = None
best_accuracy = -1
best_f1 = -1

for gnb_weight in GNB_WEIGHT_OPTIONS:

    rnn_weight = 1.0 - gnb_weight

    ensemble_probability = (
        gnb_weight * gnb_val_probability
        + rnn_weight * rnn_val_probability
    )

    ensemble_prediction = (
        ensemble_probability >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        ensemble_prediction
    )

    f1 = f1_score(
        y_val,
        ensemble_prediction,
        zero_division=0
    )

    print(
        f"GNB: {gnb_weight * 100:5.1f}% | "
        f"RNN: {rnn_weight * 100:5.1f}% | "
        f"Accuracy: {accuracy * 100:6.2f}% | "
        f"F1: {f1 * 100:6.2f}%"
    )

    # Select based on validation accuracy.
    # If accuracy is tied, use F1 as the tie-breaker.
    if (
        accuracy > best_accuracy
        or (
            accuracy == best_accuracy
            and f1 > best_f1
        )
    ):
        best_accuracy = accuracy
        best_f1 = f1
        best_weight = gnb_weight


best_gnb_weight = best_weight
best_rnn_weight = 1.0 - best_weight


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)

print("\nSelected weights:")
print(f"GNB: {best_gnb_weight * 100:.0f}%")
print(f"RNN: {best_rnn_weight * 100:.0f}%")


# Apply selected weights to the untouched test set
ensemble_probability = (
    best_gnb_weight * gnb_test_probability
    + best_rnn_weight * rnn_test_probability
)

ensemble_probability = np.clip(
    ensemble_probability,
    0.0,
    1.0
)

ensemble_prediction = (
    ensemble_probability >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    ensemble_prediction
)

precision = precision_score(
    y_test,
    ensemble_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    ensemble_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    ensemble_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    ensemble_probability
)

cm = confusion_matrix(
    y_test,
    ensemble_prediction
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nTest Set Performance:")

print(f"Accuracy:     {accuracy * 100:.2f}%")
print(f"Precision:    {precision * 100:.2f}%")
print(f"Recall:       {recall * 100:.2f}%")
print(f"Specificity:  {specificity * 100:.2f}%")
print(f"F1 Score:     {f1 * 100:.2f}%")
print(f"ROC-AUC:      {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nInterpretation:")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")


# ============================================================
# SAVE MODELS
# ============================================================

os.makedirs("models", exist_ok=True)

import joblib

joblib.dump(
    gnb,
    "models/gnb_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

rnn.save("models/rnn_model.keras")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MODELS SAVED SUCCESSFULLY")
print("=" * 70)

print("\nSaved files:")

print("models/gnb_model.pkl")
print("models/scaler.pkl")
print("models/rnn_model.h5")

print("\nTraining pipeline completed.")