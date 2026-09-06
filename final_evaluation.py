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


SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/parkinsons.data")

df = df.drop(columns=["name"])

X = df.drop(columns=["status"])
y = df["status"]


# ============================================================
# SAME TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    stratify=y_temp,
    random_state=SEED
)


# ============================================================
# SCALE
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# TRAIN GNB
# ============================================================

gnb = GaussianNB()

gnb.fit(X_train, y_train)

gnb_test_probability = gnb.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# TRAIN RNN
# ============================================================

X_train_rnn = X_train.reshape(
    X_train.shape[0],
    1,
    X_train.shape[1]
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


# ============================================================
# RNN TEST PREDICTIONS
# ============================================================

rnn_test_probability = rnn.predict(
    X_test_rnn,
    verbose=0
).flatten()

rnn_test_prediction = (
    rnn_test_probability >= 0.5
).astype(int)


# ============================================================
# FINAL SELECTED ENSEMBLE
# ============================================================

# Validation search selected:
# GNB = 0%
# RNN = 100%

gnb_weight = 0.0
rnn_weight = 1.0

ensemble_probability = (
    gnb_weight * gnb_test_probability
    + rnn_weight * rnn_test_probability
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
# EVALUATION
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

cm = confusion_matrix(
    y_test,
    ensemble_prediction
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

roc_auc = roc_auc_score(
    y_test,
    ensemble_probability
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL UNSEEN TEST SET EVALUATION")
print("=" * 70)

print(f"Test samples: {len(y_test)}")

print("\nSelected model weighting:")
print("GNB: 0%")
print("RNN: 100%")

print("\nPerformance:")
print(f"Accuracy:     {accuracy * 100:.2f}%")
print(f"Precision:    {precision * 100:.2f}%")
print(f"Recall:       {recall * 100:.2f}%")
print(f"Specificity:  {specificity * 100:.2f}%")
print(f"F1 Score:     {f1 * 100:.2f}%")
print(f"ROC-AUC:      {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(f"""
True Negatives:  {tn}
False Positives: {fp}
False Negatives: {fn}
True Positives:  {tp}
""")
