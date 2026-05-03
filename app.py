import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model_train.pkl")

st.title("🌸 Welcome to Flower Prediction App (Logistic Regression)")

# Input sliders
a = st.slider("Sepal length (cm)", 4.3, 7.9, 5.1)
b = st.slider("Sepal width (cm)", 2.0, 4.4, 3.0)
c = st.slider("Petal length (cm)", 1.0, 6.9, 1.5)
d = st.slider("Petal width (cm)", 0.1, 2.5, 0.3)

btn = st.button("Predict")

if btn:
    # Prepare input
    input_data = np.array([[a, b, c, d]])
    
    # Predict probability
    proba = model.predict_proba(input_data)[0]
    pred_index = np.argmax(proba)
    pred_label = model.classes_[pred_index]

    # Map numerical label to name
    label_map = {
        0: "Iris Setosa",
        1: "Iris Versicolour",
        2: "Iris Virginica"
    }
    pred_label_name = label_map[pred_label]

    # Output results
    st.subheader(f"🔍 Predicted Class: **{pred_label_name}**")

    st.write("### 🧪 Class Probabilities")
    for label, p in zip(model.classes_, proba):
        st.write(f"- **{label_map[label]}**: {p:.2%}")

    # Display image based on prediction
    if pred_label == 0:
        st.image("setosa.jpg")
    elif pred_label == 1:
        st.image("versicolor.jpg")
    else:
        st.image("virginica.jpg")
