import streamlit as st
import numpy as np
import joblib
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

gnb = joblib.load(
    "models/gnb_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

rnn = tf.keras.models.load_model(
    "models/rnn_model.keras"
)


# ============================================================
# EXAMPLE DATA
# ============================================================

healthy_example = [
    197.076,
    206.896,
    192.055,
    0.00289,
    0.00001,
    0.00166,
    0.00168,
    0.00498,
    0.01098,
    0.097,
    0.00563,
    0.00680,
    0.00802,
    0.01689,
    0.00339,
    26.775,
    0.422229,
    0.741367,
    -7.3483,
    0.177551,
    1.743867,
    0.085569
]


parkinsons_example = [
    119.992,
    157.302,
    74.997,
    0.00784,
    0.00007,
    0.00370,
    0.00554,
    0.01109,
    0.04374,
    0.426,
    0.02182,
    0.03130,
    0.02971,
    0.06545,
    0.02211,
    21.033,
    0.414783,
    0.815285,
    -4.813031,
    0.266482,
    2.301442,
    0.284654
]


# ============================================================
# SESSION STATE
# ============================================================

feature_names = [
    "fo",
    "fhi",
    "flo",
    "jitter_percent",
    "jitter_abs",
    "rap",
    "ppq",
    "jitter_ddp",
    "shimmer",
    "shimmer_db",
    "apq3",
    "apq5",
    "apq",
    "dda",
    "nhr",
    "hnr",
    "rpde",
    "dfa",
    "spread1",
    "spread2",
    "d2",
    "ppe"
]


# Initialize all input values
for feature in feature_names:

    if feature not in st.session_state:
        st.session_state[feature] = 0.0


# ============================================================
# PAGE TITLE
# ============================================================

st.title(
    "🧠 Parkinson's Disease Prediction System"
)

st.write(
    "Enter the biomedical voice measurements below "
    "to generate a machine-learning prediction."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.info(
    "The final selected model is the Recurrent Neural "
    "Network (RNN), based on validation performance."
)


st.subheader(
    "📊 Final Model Performance"
)

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "Accuracy",
        "97.44%"
    )

with metric2:

    st.metric(
        "Precision",
        "96.67%"
    )

with metric3:

    st.metric(
        "Recall",
        "100.00%"
    )


metric4, metric5, metric6 = st.columns(3)

with metric4:

    st.metric(
        "Specificity",
        "90.00%"
    )

with metric5:

    st.metric(
        "F1 Score",
        "98.31%"
    )

with metric6:

    st.metric(
        "ROC-AUC",
        "99.66%"
    )


# ============================================================
# QUICK EXAMPLES
# ============================================================

st.divider()

st.subheader(
    "🧪 Quick Examples"
)

st.write(
    "Use the buttons below to automatically load sample "
    "measurements from the Parkinson's dataset."
)


example_col1, example_col2, example_col3 = st.columns(3)


with example_col1:

    load_healthy = st.button(
        "✅ Load Healthy Example",
        use_container_width=True
    )


with example_col2:

    load_parkinson = st.button(
        "⚠️ Load Parkinson's Example",
        use_container_width=True
    )


with example_col3:

    clear_inputs = st.button(
        "🧹 Clear Inputs",
        use_container_width=True
    )


# Load healthy example
if load_healthy:

    for feature, value in zip(
        feature_names,
        healthy_example
    ):

        st.session_state[feature] = value

    st.success(
        "Healthy example loaded successfully."
    )


# Load Parkinson's example
if load_parkinson:

    for feature, value in zip(
        feature_names,
        parkinsons_example
    ):

        st.session_state[feature] = value

    st.success(
        "Parkinson's example loaded successfully."
    )


# Clear all inputs
if clear_inputs:

    for feature in feature_names:

        st.session_state[feature] = 0.0

    st.success(
        "All input fields have been cleared."
    )


# ============================================================
# VOICE MEASUREMENTS
# ============================================================

st.divider()

st.subheader(
    "🎙️ Voice Measurements"
)

st.write(
    "Enter the 22 biomedical voice measurements used "
    "by the trained models."
)


col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    fo = st.number_input(
        "Fo (Hz)",
        format="%.6f",
        key="fo"
    )

    fhi = st.number_input(
        "Fhi (Hz)",
        format="%.6f",
        key="fhi"
    )

    flo = st.number_input(
        "Flo (Hz)",
        format="%.6f",
        key="flo"
    )

    jitter_percent = st.number_input(
        "Jitter (%)",
        format="%.6f",
        key="jitter_percent"
    )

    jitter_abs = st.number_input(
        "Jitter (Abs)",
        format="%.6f",
        key="jitter_abs"
    )

    rap = st.number_input(
        "RAP",
        format="%.6f",
        key="rap"
    )

    ppq = st.number_input(
        "PPQ",
        format="%.6f",
        key="ppq"
    )

    jitter_ddp = st.number_input(
        "Jitter DDP",
        format="%.6f",
        key="jitter_ddp"
    )

    shimmer = st.number_input(
        "Shimmer",
        format="%.6f",
        key="shimmer"
    )

    shimmer_db = st.number_input(
        "Shimmer (dB)",
        format="%.6f",
        key="shimmer_db"
    )

    apq3 = st.number_input(
        "APQ3",
        format="%.6f",
        key="apq3"
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    apq5 = st.number_input(
        "APQ5",
        format="%.6f",
        key="apq5"
    )

    apq = st.number_input(
        "APQ",
        format="%.6f",
        key="apq"
    )

    dda = st.number_input(
        "Shimmer DDA",
        format="%.6f",
        key="dda"
    )

    nhr = st.number_input(
        "NHR",
        format="%.6f",
        key="nhr"
    )

    hnr = st.number_input(
        "HNR",
        format="%.6f",
        key="hnr"
    )

    rpde = st.number_input(
        "RPDE",
        format="%.6f",
        key="rpde"
    )

    dfa = st.number_input(
        "DFA",
        format="%.6f",
        key="dfa"
    )

    spread1 = st.number_input(
        "Spread1",
        format="%.6f",
        key="spread1"
    )

    spread2 = st.number_input(
        "Spread2",
        format="%.6f",
        key="spread2"
    )

    d2 = st.number_input(
        "D2",
        format="%.6f",
        key="d2"
    )

    ppe = st.number_input(
        "PPE",
        format="%.6f",
        key="ppe"
    )


# ============================================================
# CREATE INPUT ARRAY
# ============================================================

input_data = np.array([
    fo,
    fhi,
    flo,
    jitter_percent,
    jitter_abs,
    rap,
    ppq,
    jitter_ddp,
    shimmer,
    shimmer_db,
    apq3,
    apq5,
    apq,
    dda,
    nhr,
    hnr,
    rpde,
    dfa,
    spread1,
    spread2,
    d2,
    ppe
]).reshape(1, -1)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Parkinson's Risk",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # SCALE INPUT
    # --------------------------------------------------------

    scaled_input = scaler.transform(
        input_data
    )


    # --------------------------------------------------------
    # GNB PREDICTION
    # --------------------------------------------------------

    gnb_probability = gnb.predict_proba(
        scaled_input
    )[0][1]


    # --------------------------------------------------------
    # RNN PREDICTION
    # --------------------------------------------------------

    rnn_input = scaled_input.reshape(
        1,
        1,
        scaled_input.shape[1]
    )

    rnn_probability = rnn.predict(
        rnn_input,
        verbose=0
    )[0][0]


    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    # Validation-selected weighting
    gnb_weight = 0.0
    rnn_weight = 1.0


    final_probability = (
        gnb_weight * gnb_probability
        + rnn_weight * rnn_probability
    )


    # Keep probability between 0 and 1
    final_probability = np.clip(
        final_probability,
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if final_probability >= 0.5:

        prediction = "Parkinson's Disease"

    else:

        prediction = "Healthy"


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader(
        "📊 Prediction Results"
    )


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "GNB Predicted Probability",
            f"{gnb_probability * 100:.2f}%"
        )


    with result_col2:

        st.metric(
            "RNN Predicted Probability",
            f"{rnn_probability * 100:.2f}%"
        )


    with result_col3:

        st.metric(
            "Final Predicted Probability",
            f"{final_probability * 100:.2f}%"
        )


    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Final Prediction"
    )


    if prediction == "Parkinson's Disease":

        st.error(
            "⚠️ The model predicts a higher likelihood "
            "of Parkinson's Disease."
        )

    else:

        st.success(
            "✅ The model predicts a higher likelihood "
            "of a healthy result."
        )


    # ========================================================
    # PREDICTION INTERPRETATION
    # ========================================================

    st.subheader(
        "📌 Prediction Interpretation"
    )


    if final_probability < 0.2:

        st.success(
            "🟢 Low predicted likelihood of "
            "Parkinson's Disease."
        )

    elif final_probability < 0.5:

        st.info(
            "🟡 Moderate predicted likelihood of "
            "Parkinson's Disease."
        )

    elif final_probability < 0.8:

        st.warning(
            "🟠 High predicted likelihood of "
            "Parkinson's Disease."
        )

    else:

        st.error(
            "🔴 Very high predicted likelihood of "
            "Parkinson's Disease."
        )


    # ========================================================
    # MODEL SELECTION
    # ========================================================

    st.subheader(
        "⚖️ Model Selection"
    )


    selection_col1, selection_col2 = st.columns(2)


    with selection_col1:

        st.info(
            "Gaussian Naive Bayes\n\n"
            "**Weight: 0%**"
        )


    with selection_col2:

        st.success(
            "Recurrent Neural Network\n\n"
            "**Weight: 100%**"
        )


    # ========================================================
    # RAW PROBABILITIES
    # ========================================================

    with st.expander(
        "🔎 View Raw Model Probabilities"
    ):

        st.write(
            "GNB raw probability:",
            gnb_probability
        )

        st.write(
            "RNN raw probability:",
            rnn_probability
        )

        st.write(
            "Final model probability:",
            final_probability
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.divider()

    st.warning(
        "⚠️ This system is intended for educational "
        "and research purposes only. It is not a medical "
        "diagnostic tool and should not replace evaluation "
        "by a qualified healthcare professional."
    )