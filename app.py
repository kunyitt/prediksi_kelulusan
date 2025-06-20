import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model dan LabelEncoder ---
model = joblib.load("prediksi_kelulusan.pkl")
encoders = joblib.load("encoders.pkl")

# --- Judul aplikasi ---
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi apakah dia akan **TEPAT** atau **TERLAMBAT** lulus.")

# --- Form input ---
with st.form("form_kelulusan"):
    st.subheader("Masukkan Data Mahasiswa")

    jk = st.selectbox("Jenis Kelamin", encoders['JENIS KELAMIN'].classes_)
    status_mhs = st.selectbox("Status Mahasiswa", encoders['STATUS MAHASISWA'].classes_)
    status_nikah = st.selectbox("Status Nikah", encoders['STATUS NIKAH'].classes_)
    umur = st.number_input("Umur", min_value=17, max_value=60, value=22)

    ips_values = []
    for i in range(1, 9):
        ips = st.number_input(f"IPS {i}", min_value=0.0, max_value=4.0, value=3.0)
        ips_values.append(ips)

    ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0)

    # ✅ Tombol submit HARUS di dalam form
    submit = st.form_submit_button("Prediksi")

# --- Proses prediksi saat submit ditekan ---
if submit:
    try:
        # Encode input
        input_data = pd.DataFrame([[
            encoders['JENIS KELAMIN'].transform([jk])[0],
            encoders['STATUS MAHASISWA'].transform([status_mhs])[0],
            umur,
            encoders['STATUS NIKAH'].transform([status_nikah])[0],
            *ips_values,
            ipk
        ]], columns=model.feature_names_in_)

        # Prediksi
        prediction = model.predict(input_data)[0]
        hasil = encoders['STATUS KELULUSAN'].inverse_transform([prediction])[0]

        st.subheader("Hasil Prediksi")
        if hasil.upper() == "TERLAMBAT":
            st.error(f"Mahasiswa diprediksi LULUS **{hasil.upper()}**")
        else:
            st.success(f"Mahasiswa diprediksi LULUS **{hasil.upper()}**")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses prediksi.")
        st.exception(e)
