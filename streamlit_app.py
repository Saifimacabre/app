import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay
)
import optuna

# --- Cache CSV loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("Fertilizer Prediction.csv")
    return df

# --- Cache model training ---
@st.cache_resource
def train_model(df):
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('Fertilizer Name', axis=1)
    y = df['Fertilizer Name']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        }
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    best_params = study.best_params

    clf = RandomForestClassifier(**best_params, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return clf, label_encoders, X_test, y_test, y_pred, accuracy

# --- Load data and train model ---
df = load_data()
clf, label_encoders, X_test, y_test, y_pred, accuracy = train_model(df)

# --- Streamlit UI ---
st.title("🌿 Fertilizer Prediction Web App")

st.sidebar.header("Input Parameters")
temp = st.sidebar.number_input("Temperature", 0.0, 100.0)
humidity = st.sidebar.number_input("Humidity", 0.0, 100.0)
moisture = st.sidebar.number_input("Moisture", 0.0, 100.0)
soil = st.sidebar.selectbox("Soil Type", label_encoders['Soil Type'].classes_)
crop = st.sidebar.selectbox("Crop Type", label_encoders['Crop Type'].classes_)
nitrogen = st.sidebar.number_input("Nitrogen", 0.0, 100.0)
potassium = st.sidebar.number_input("Potassium", 0.0, 100.0)
phosphorous = st.sidebar.number_input("Phosphorous", 0.0, 100.0)

if st.sidebar.button("🔍 Predict"):
    soil_enc = label_encoders['Soil Type'].transform([soil])[0]
    crop_enc = label_encoders['Crop Type'].transform([crop])[0]
    input_data = np.array([[temp, humidity, moisture, soil_enc, crop_enc,
                            nitrogen, potassium, phosphorous]])
    pred_code = clf.predict(input_data)[0]
    fert_name = label_encoders['Fertilizer Name'].inverse_transform([pred_code])[0]
    st.success(f"✅ Predicted Fertilizer: **{fert_name}**")

st.markdown(f"### Model Accuracy: `{accuracy:.2%}`")

st.markdown("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig1, ax1 = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=label_encoders['Fertilizer Name'].classes_)
disp.plot(ax=ax1, cmap="Greens")
st.pyplot(fig1)

st.markdown("### Precision-Recall Curves")
fig2, ax2 = plt.subplots(figsize=(10, 6))
for i in np.unique(y_test):
    precision, recall, _ = precision_recall_curve((y_test == i).astype(int), clf.predict_proba(X_test)[:, i])
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot(ax=ax2, name=label_encoders['Fertilizer Name'].inverse_transform([i])[0])
plt.title("Precision-Recall Curve")
st.pyplot(fig2)

if st.checkbox("Show Predictions Table"):
    results_df = X_test.copy()
    results_df['Actual'] = label_encoders['Fertilizer Name'].inverse_transform(y_test)
    results_df['Predicted'] = label_encoders['Fertilizer Name'].inverse_transform(y_pred)
    st.dataframe(results_df)
