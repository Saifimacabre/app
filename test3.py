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
# === Data Loading ===
df = pd.read_csv("F:\\ML pros\\Fertilizer Prediction.csv")
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Optuna Hyperparameter Tuning ===
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

# === Train Final Model ===
clf = RandomForestClassifier(**best_params, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# === Export Predictions ===
results_df = X_test.copy()
results_df['Actual'] = label_encoders['Fertilizer Name'].inverse_transform(y_test)
results_df['Predicted'] = label_encoders['Fertilizer Name'].inverse_transform(y_pred)
results_df.to_csv("fertilizer_predictions.csv", index=False)

# === Save Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
fig1, ax1 = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=label_encoders['Fertilizer Name'].classes_)
disp.plot(ax=ax1, cmap='Greens')
plt.title("Confusion Matrix")
fig1.savefig("confusion_matrix.png")

# === Save Precision-Recall Curves ===
fig2, ax2 = plt.subplots(figsize=(10, 7))
for i in np.unique(y):
    precision, recall, _ = precision_recall_curve((y_test == i).astype(int), clf.predict_proba(X_test)[:, i])
    display = PrecisionRecallDisplay(precision=precision, recall=recall)
    display.plot(ax=ax2, name=label_encoders['Fertilizer Name'].inverse_transform([i])[0])
plt.title("Precision-Recall Curves")
plt.legend(loc='lower left')
fig2.savefig("precision_recall.png")

# === Tkinter GUI ===
def run_gui():
    def predict():
        try:
            input_data = [float(entry.get()) for entry in entries[:3]] +                          [label_encoders['Soil Type'].transform([entries[3].get()])[0]] +                          [label_encoders['Crop Type'].transform([entries[4].get()])[0]] +                          [float(entry.get()) for entry in entries[5:]]
            pred_code = clf.predict([input_data])[0]
            result = label_encoders['Fertilizer Name'].inverse_transform([pred_code])[0]
            messagebox.showinfo("Prediction", f"Predicted Fertilizer: {result}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Fertilizer Prediction (Tkinter)")
    labels = ["Temperature", "Humidity", "Moisture", "Soil Type", "Crop Type",
              "Nitrogen", "Potassium", "Phosphorous"]
    entries = []
    for i, label in enumerate(labels):
        tk.Label(root, text=label).grid(row=i, column=0)
        entry = tk.Entry(root)
        entry.grid(row=i, column=1)
        entries.append(entry)
    tk.Button(root, text="Predict", command=predict).grid(row=len(labels), columnspan=2)
    root.mainloop()

# === Streamlit Web App ===
def run_streamlit():
    st.title("Fertilizer Prediction Web App")
    temp = st.number_input("Temperature", 0.0, 100.0)
    humidity = st.number_input("Humidity", 0.0, 100.0)
    moisture = st.number_input("Moisture", 0.0, 100.0)
    soil = st.selectbox("Soil Type", label_encoders['Soil Type'].classes_)
    crop = st.selectbox("Crop Type", label_encoders['Crop Type'].classes_)
    nitrogen = st.number_input("Nitrogen", 0.0, 100.0)
    potassium = st.number_input("Potassium", 0.0, 100.0)
    phosphorous = st.number_input("Phosphorous", 0.0, 100.0)

    if st.button("Predict"):
        soil_enc = label_encoders['Soil Type'].transform([soil])[0]
        crop_enc = label_encoders['Crop Type'].transform([crop])[0]
        input_data = np.array([[temp, humidity, moisture, soil_enc, crop_enc,
                                nitrogen, potassium, phosphorous]])
        pred_code = clf.predict(input_data)[0]
        fert_name = label_encoders['Fertilizer Name'].inverse_transform([pred_code])[0]
        st.success(f"Predicted Fertilizer: {fert_name}")

# Uncomment one of the following to run desired interface:
# run_gui()
run_streamlit()
