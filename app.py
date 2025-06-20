import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- Load & Preprocess Data ---
df = pd.read_csv('/content/Kelulusan Train.csv')

if 'NAMA' in df.columns:
    df.drop(columns=['NAMA'], inplace=True)

label_cols = ['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH', 'STATUS KELULUSAN']
encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# --- Split Data dan Train Model ---
X = df.drop(columns=['STATUS KELULUSAN'])
y = df['STATUS KELULUSAN']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# --- Streamlit UI ---
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi apakah dia akan **TEPAT** atau **TERLAMBAT** lulus.")

# --- Input Form ---
jk = st.selectbox("Jenis Kelamin", encoders['JENIS KELAMIN'].classes_)
status_mahasiswa = st.selectbox("Status Mahasiswa", encoders['STATUS MAHASISWA'].classes_)
status_nikah = st.selectbox("Status Nikah", encoders['STATUS NIKAH'].classes_)
umur = st.number_input("Umur", min_value=17, max_value=60, value=25)

ips_values = []
for i in range(1, 9):
    ips = st.number_input(f"IPS {i}", min_value=0.0, max_value=4.0, value=3.0)
    ips_values.append(ips)
ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0)

# --- Prediksi ---
if st.button("Prediksi Kelulusan"):
    input_data = pd.DataFrame([[
        encoders['JENIS KELAMIN'].transform([jk])[0],
        encoders['STATUS MAHASISWA'].transform([status_mahasiswa])[0],
        umur,
        encoders['STATUS NIKAH'].transform([status_nikah])[0],
        *ips_values,
        ipk
    ]], columns=X.columns)

    prediction = model.predict(input_data)[0]
    result = encoders['STATUS KELULUSAN'].inverse_transform([prediction])[0]

    st.success(f"✅ Prediksi Kelulusan: **{result.upper()}**")
    st.info(f"🎯 Akurasi model: {accuracy * 100:.2f}%")
