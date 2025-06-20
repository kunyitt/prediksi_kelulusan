import streamlit as st
import pandas as pd
import joblib

# Load model & encoder
model = joblib.load("prediksi_kelulusan.pkl")
encoders = joblib.load("encoders.pkl")

# Kolom input yang digunakan model
input_columns = [
    'JENIS KELAMIN', 'STATUS MAHASISWA', 'UMUR', 'STATUS NIKAH',
    'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4',
    'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8', 'IPK'
]

# Judul Aplikasi
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi apakah akan **Tepat** atau **Terlambat** lulus.")

# Form input pengguna
with st.form("form_kelulusan"):
    jk = st.selectbox("Jenis Kelamin", encoders['JENIS KELAMIN'].classes_)
    status_mhs = st.selectbox("Status Mahasiswa", encoders['STATUS MAHASISWA'].classes_)
    status_nikah = st.selectbox("Status Nikah", encoders['STATUS NIKAH'].classes_)
    umur = st.number_input("Umur", 17, 60, value=22)

    ips_values = [st.number_input(f"IPS {i}", 0.0, 4.0, 3.0) for i in range(1, 9)]
    ipk = st.number_input("IPK", 0.0, 4.0, 3.0)

    submit = st.form_submit_button("Prediksi")

# Ketika tombol submit ditekan
if submit:
    try:
        # Susun input dalam DataFrame
        input_df = pd.DataFrame([[
            encoders['JENIS KELAMIN'].transform([jk])[0],
            encoders['STATUS MAHASISWA'].transform([status_mhs])[0],
            umur,
            encoders['STATUS NIKAH'].transform([status_nikah])[0],
            *ips_values,
            ipk
        ]], columns=input_columns)

        # Prediksi
        pred = model.predict(input_df)[0]
        hasil = encoders['STATUS KELULUSAN'].inverse_transform([pred])[0]

        # Tampilkan hasil
        if hasil.upper() == "TERLAMBAT":
            st.error(f"❌ Mahasiswa diprediksi LULUS {hasil.upper()}")
        else:
            st.success(f"✅ Mahasiswa diprediksi LULUS {hasil.upper()}")

    except Exception as e:
        st.error("Terjadi error saat memproses prediksi.")
        st.exception(e)
