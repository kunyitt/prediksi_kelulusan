import streamlit as st
import pandas as pd
import joblib

# Load model dan encoder
model = joblib.load('model_kelulusan.pkl')
encoders = joblib.load('encoders.pkl')

st.title("Prediksi Kelulusan Mahasiswa")

# Form input user
with st.form("form_kelulusan"):
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-Laki", "Perempuan"])
    status_mahasiswa = st.selectbox("Status Mahasiswa", ["Aktif", "Cuti", "DO", "Lulus"])
    status_nikah = st.selectbox("Status Nikah", ["Belum Menikah", "Menikah"])
    ips = st.number_input("IP Semester", min_value=0.0, max_value=4.0, step=0.01)
    ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, step=0.01)
    sks = st.number_input("Jumlah SKS", min_value=0)

    submit = st.form_submit_button("Prediksi")

if submit:
    # Encode input user
    encoded_input = {
        'JENIS KELAMIN': encoders['JENIS KELAMIN'].transform([jenis_kelamin])[0],
        'STATUS MAHASISWA': encoders['STATUS MAHASISWA'].transform([status_mahasiswa])[0],
        'STATUS NIKAH': encoders['STATUS NIKAH'].transform([status_nikah])[0],
        'IPS': ips,
        'IPK': ipk,
        'SKS': sks
    }

    df_input = pd.DataFrame([encoded_input])
    pred = model.predict(df_input)[0]
    label = encoders['STATUS KELULUSAN'].inverse_transform([pred])[0]

    st.success(f"Prediksi: Mahasiswa diperkirakan akan {label}")
