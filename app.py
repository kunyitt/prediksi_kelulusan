import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan encoders
model = joblib.load('model_kelulusan.pkl')
encoders = joblib.load('encoders.pkl')

st.title("Prediksi Kelulusan Mahasiswa")

with st.form("form_prediksi"):
    # Input kategorikal
    jenis_kelamin = st.selectbox("Jenis Kelamin", encoders['JENIS KELAMIN'].classes_)
    status_mahasiswa = st.selectbox("Status Mahasiswa", encoders['STATUS MAHASISWA'].classes_)
    status_nikah = st.selectbox("Status Nikah", encoders['STATUS NIKAH'].classes_)

    # Input umur
    umur = st.number_input("Umur", min_value=15, max_value=100)

    # Input IPS 1 - 8
    ips1 = st.number_input("IPS 1", min_value=0.0, max_value=4.0, step=0.01)
    ips2 = st.number_input("IPS 2", min_value=0.0, max_value=4.0, step=0.01)
    ips3 = st.number_input("IPS 3", min_value=0.0, max_value=4.0, step=0.01)
    ips4 = st.number_input("IPS 4", min_value=0.0, max_value=4.0, step=0.01)
    ips5 = st.number_input("IPS 5", min_value=0.0, max_value=4.0, step=0.01)
    ips6 = st.number_input("IPS 6", min_value=0.0, max_value=4.0, step=0.01)
    ips7 = st.number_input("IPS 7", min_value=0.0, max_value=4.0, step=0.01)
    ips8 = st.number_input("IPS 8", min_value=0.0, max_value=4.0, step=0.01)
    
    # Hitung IPK dari IPS 1-8
    ipk = round(np.mean([ips1, ips2, ips3, ips4, ips5, ips6, ips7, ips8]), 2)
    st.info(f"IPK secara otomatis dihitung: {ipk}")

    submit = st.form_submit_button("Prediksi")

if submit:
    # Encode input kategorikal
    input_data = {
        'JENIS KELAMIN': encoders['JENIS KELAMIN'].transform([jenis_kelamin])[0],
        'STATUS MAHASISWA': encoders['STATUS MAHASISWA'].transform([status_mahasiswa])[0],
        'STATUS NIKAH': encoders['STATUS NIKAH'].transform([status_nikah])[0],
        'UMUR': umur,
        'IPS 1': ips1,
        'IPS 2': ips2,
        'IPS 3': ips3,
        'IPS 4': ips4,
        'IPS 5': ips5,
        'IPS 6': ips6,
        'IPS 7': ips7,
        'IPS 8': ips8,
        'IPK ': ipk
    }

    # Pastikan urutan kolom sesuai saat training
    df_input = pd.DataFrame([input_data])
    pred = model.predict(df_input)[0]
    hasil = encoders['STATUS KELULUSAN'].inverse_transform([pred])[0]

    st.success(f"Hasil Prediksi: Mahasiswa diperkirakan akan **{hasil.upper()}**")
