import streamlit as st
import pandas as pd
import joblib

# Load model dan encoders
model = joblib.load('model_kelulusan.pkl')
encoders = joblib.load('encoders.pkl')

st.title("Prediksi Kelulusan Mahasiswa")

with st.form("form_prediksi"):
    # Input kategorikal
    jenis_kelamin = st.selectbox("Jenis Kelamin", encoders['JENIS KELAMIN'].classes_)
    status_mahasiswa = st.selectbox("Status Mahasiswa", encoders['STATUS MAHASISWA'].classes_)
    status_nikah = st.selectbox("Status Nikah", encoders['STATUS NIKAH'].classes_)

    # Input IPS 1-8
    ips1 = st.number_input("IPS 1", min_value=0.0, max_value=4.0, step=0.01)
    ips2 = st.number_input("IPS 2", min_value=0.0, max_value=4.0, step=0.01)
    ips3 = st.number_input("IPS 3", min_value=0.0, max_value=4.0, step=0.01)
    ips4 = st.number_input("IPS 4", min_value=0.0, max_value=4.0, step=0.01)
    ips5 = st.number_input("IPS 5", min_value=0.0, max_value=4.0, step=0.01)
    ips6 = st.number_input("IPS 6", min_value=0.0, max_value=4.0, step=0.01)
    ips7 = st.number_input("IPS 7", min_value=0.0, max_value=4.0, step=0.01)
    ips8 = st.number_input("IPS 8", min_value=0.0, max_value=4.0, step=0.01)

    # Input IPK
    ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01)

    submit = st.form_submit_button("Prediksi")

if submit:
    # Encode input kategorikal
    input_data = {
    'JENIS_KELAMIN': encoders['JENIS KELAMIN'].transform([jenis_kelamin])[0],
    'STATUS_MAHASISWA': encoders['STATUS MAHASISWA'].transform([status_mahasiswa])[0],
    'STATUS_NIKAH': encoders['STATUS NIKAH'].transform([status_nikah])[0],
    'IPS_1': ips1,
    'IPS_2': ips2,
    'IPS_3': ips3,
    'IPS_4': ips4,
    'IPS_5': ips5,
    'IPS_6': ips6,
    'IPS_7': ips7,
    'IPS_8': ips8,
    'IPK': ipk
}

    df_input = pd.DataFrame([input_data])
    
    # Pastikan kolom sesuai urutan saat training
    expected_cols = model.feature_names_in_  # properti dari sklearn >= 1.0
    df_input = df_input[expected_cols]
    
    pred = model.predict(df_input)[0]
    label_prediksi = encoders['STATUS KELULUSAN'].inverse_transform([pred])[0]

    st.success(f"Hasil Prediksi: Mahasiswa diperkirakan akan **{label_prediksi.upper()}**")

print("Input cols:", df_input.columns.tolist())
print("Expected cols:", model.feature_names_in_.tolist())
