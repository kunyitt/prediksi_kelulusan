import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Judul Aplikasi
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Aplikasi ini memprediksi status kelulusan mahasiswa berdasarkan data input.")

# Load Model dan Encoder
@st.cache_resource  # Cache untuk mempercepat loading
def load_model():
    model = joblib.load('prediksi_kelulusan.joblib')
    encoders = joblib.load('encoders.joblib')
    return model, encoders

model, encoders = load_model()

# Input Data
st.sidebar.header("Input Data Mahasiswa")

# Fungsi untuk input user
def user_input():
    jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    status_mahasiswa = st.sidebar.selectbox("Status Mahasiswa", ["Aktif", "Non-Aktif"])
    status_nikah = st.sidebar.selectbox("Status Nikah", ["Belum Menikah", "Menikah"])
    ipk = st.sidebar.slider("IPK", 0.0, 4.0, 3.0)
    lama_studi = st.sidebar.number_input("Lama Studi (Tahun)", min_value=1, max_value=10, value=4)
    
    data = {
        'JENIS KELAMIN': jenis_kelamin,
        'STATUS MAHASISWA': status_mahasiswa,
        'STATUS NIKAH': status_nikah,
        'IPK': ipk,
        'LAMA STUDI': lama_studi
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# Tampilkan Input
st.subheader("Data Input")
st.write(input_df)

# Preprocessing
def preprocess(data):
    # Salin data
    df = data.copy()
    
    # Label Encoding
    df['JENIS KELAMIN'] = encoders['JENIS KELAMIN'].transform(df['JENIS KELAMIN'])
    df['STATUS MAHASISWA'] = encoders['STATUS MAHASISWA'].transform(df['STATUS MAHASISWA'])
    df['STATUS NIKAH'] = encoders['STATUS NIKAH'].transform(df['STATUS NIKAH'])
    
    return df

# Prediksi
if st.button("Prediksi Kelulusan"):
    # Preprocess input
    processed_df = preprocess(input_df)
    
    # Prediksi
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)
    
    # Mapping hasil prediksi
    status_map = {0: "Tidak Lulus", 1: "Lulus"}  # Sesuaikan dengan mapping LabelEncoder
    
    st.subheader("Hasil Prediksi")
    st.write(f"Status Kelulusan: **{status_map[prediction[0]]}**")
    
    st.subheader("Probabilitas Prediksi")
    st.write(f"Probabilitas Tidak Lulus: {prediction_proba[0][0]:.2f}")
    st.write(f"Probabilitas Lulus: {prediction_proba[0][1]:.2f}")
    
    # Visualisasi
    st.bar_chart({
        "Tidak Lulus": prediction_proba[0][0],
        "Lulus": prediction_proba[0][1]
    })

# Catatan
st.sidebar.markdown("---")
st.sidebar.info(
    "Pastikan file `prediksi_kelulusan.joblib` dan `encoders.joblib` "
    "berada di folder yang sama dengan script ini."
)
