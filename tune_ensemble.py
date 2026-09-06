import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, recall_score


# ============================================================
# SETTINGS
# ============================================================

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
# CREATE TRAIN / VALIDATION / TEST SETS
# ============================================================

# First: separate the final test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

# Then: split the remaining 80% into training and validation
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    stratify=y_temp,
    random_state=SEED
)


print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Training samples:   ", len(X_train_raw))
print("Validation samples: ", len(X_val_raw))
print("Test samples:       ", len(X_test))


# ============================================================
# SCALE DATA
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# TRAIN GNB
# ============================================================

print("\n" + "=" * 70)
print("TRAINING GNB")
print("=" * 70)

gnb = GaussianNB()

gnb.fit(X_train, y_train)


gnb_val_probability = gnb.predict_proba(X_val)[:, 1]


# ============================================================
# TRAIN RNN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RNN")
print("=" * 70)

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
    verbose=1
)


rnn_val_probability = rnn.predict(
    X_val_rnn,
    verbose=0
).flatten()


# ============================================================
# TEST DIFFERENT ENSEMBLE WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("SEARCHING FOR BEST ENSEMBLE WEIGHT")
print("=" * 70)

results = []


for gnb_weight in np.arange(0.0, 1.01, 0.05):

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

    recall = recall_score(
        y_val,
        ensemble_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        ensemble_prediction,
        zero_division=0
    )

    results.append({
        "GNB Weight": gnb_weight,
        "RNN Weight": rnn_weight,
        "Accuracy": accuracy,
        "Recall": recall,
        "F1 Score": f1
    })


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Accuracy", "F1 Score"],
    ascending=False
)


print("\n")
print(results_df.to_string(index=False))


# ============================================================
# BEST WEIGHT
# ============================================================

best = results_df.iloc[0]

best_gnb_weight = best["GNB Weight"]
best_rnn_weight = best["RNN Weight"]


print("\n" + "=" * 70)
print("BEST ENSEMBLE WEIGHT")
print("=" * 70)

print(
    f"GNB Weight: {best_gnb_weight * 100:.0f}%"
)

print(
    f"RNN Weight: {best_rnn_weight * 100:.0f}%"
)

print(
    f"Validation Accuracy: {best['Accuracy'] * 100:.2f}%"
)

print(
    f"Validation Recall: {best['Recall'] * 100:.2f}%"
)

print(
    f"Validation F1 Score: {best['F1 Score'] * 100:.2f}%"
)


print("\nIMPORTANT:")
print(
    "The final test set has NOT been used to select the weight."
)
print(
    "The selected weight should now be evaluated once on the test set."
)