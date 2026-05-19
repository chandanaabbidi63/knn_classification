# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import seaborn as sns

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="KNN Classification - Iris Dataset",
    page_icon="🌸",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    color: #4B0082;
    text-align: center;
}

.stButton>button {
    background-color: #6A5ACD;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #483D8B;
    color: white;
}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🌸 KNN Classification Using Iris Dataset")

st.markdown("""
This application demonstrates **K-Nearest Neighbors (KNN) Classification**
using the famous **Iris Dataset**.
""")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target
df["species"] = df["target"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

k_value = st.sidebar.slider(
    "Select K Value",
    min_value=1,
    max_value=15,
    value=5
)

test_size = st.sidebar.slider(
    "Test Size (%)",
    min_value=10,
    max_value=40,
    value=20
)

# ---------------------------------------------------
# DISPLAY DATASET
# ---------------------------------------------------
st.subheader("📂 Iris Dataset")

st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------------
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
model = KNeighborsClassifier(n_neighbors=k_value)

model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div class='metric-box'>
        <h3>✅ Accuracy</h3>
        <h2>{accuracy:.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class='metric-box'>
        <h3>📌 K Value</h3>
        <h2>{k_value}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------
st.subheader("🧮 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

# ---------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------
st.subheader("📄 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df, use_container_width=True)

# ---------------------------------------------------
# USER INPUT FOR PREDICTION
# ---------------------------------------------------
st.subheader("🔍 Predict Iris Flower")

st.markdown("Enter flower measurements below:")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2
    )

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------
if st.button("Predict Flower Type"):

    user_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Scale user input
    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)[0]
    prediction_proba = model.predict_proba(user_data_scaled)

    flower_name = iris.target_names[prediction]

    st.success(f"🌸 Predicted Flower Type: **{flower_name.upper()}**")

    # Prediction Probabilities
    st.subheader("📈 Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Flower Type": iris.target_names,
        "Probability": prediction_proba[0]
    })

    st.dataframe(prob_df, use_container_width=True)

    # Bar Chart
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    ax2.bar(
        prob_df["Flower Type"],
        prob_df["Probability"]
    )

    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Flower Type")
    ax2.set_title("Prediction Probability")

    st.pyplot(fig2)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Made with ❤️ using Streamlit & Scikit-Learn</center>",
    unsafe_allow_html=True
)