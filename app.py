import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Konfigurasi tampilan
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

# Judul Aplikasi
st.markdown("# Prediksi Kelulusan Mahasiswa")
st.markdown("Masukkan data mahasiswa untuk memprediksi apakah akan **Tepat** atau **Terlambat** lulus.")

# Load Model dan Encoder
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('prediksi_kelulusan.joblib')
        encoders = joblib.load('encoders.joblib')
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, encoders = load_artifacts()

if model is None or encoders is None:
    st.stop()

# Input Data
with st.form("input_form"):
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Jenis Kelamin")
        jenis_kelamin = st.radio(
            "Pilih jenis kelamin",
            ["LAKI-LAKI", "PEREMPUAN"],
            index=0
        )
        
        st.markdown("### Status Mahasiswa")
        status_mahasiswa = st.radio(
            "Pilih status",
            ["BEKERJA", "TIDAK BEKERJA"],
            index=0
        )
        
        st.markdown("### Status Nikah")
        status_nikah = st.radio(
            "Pilih status",
            ["BELUM MENIKAH", "MENIKAH"],
            index=0
        )
        
    with col2:
        st.markdown("### Umur")
        umur = st.number_input("Masukkan umur", min_value=17, max_value=50, value=22)
        
        st.markdown("### IP Semester")
        ips_values = []
        cols = st.columns(8)
        for i in range(8):
            with cols[i]:
                ips = st.number_input(
                    f"IPS.{i+1}",
                    min_value=0.0,
                    max_value=4.0,
                    value=3.0,
                    step=0.01,
                    key=f"ips_{i}"
                )
                ips_values.append(ips)
        
        st.markdown("### IPK")
        ipk = st.number_input("Masukkan IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    
    st.markdown("---")
    submit_button = st.form_submit_button("PREDIKSI")

if submit_button:
    try:
        jk_encoded = encoders['JENIS KELAMIN'].transform([jenis_kelamin])[0]
        status_m_encoded = encoders['STATUS MAHASISWA'].transform([status_mahasiswa])[0]
        status_nikah_encoded = encoders['STATUS NIKAH'].transform([status_nikah])[0]

        input_df = pd.DataFrame([[
            jk_encoded,
            status_m_encoded,
            umur,
            status_nikah_encoded,
            *ips_values,
            ipk
        ]], columns=[
            'JENIS KELAMIN', 'STATUS MAHASISWA', 'UMUR', 'STATUS NIKAH',
            'IPS.1', 'IPS.2', 'IPS.3', 'IPS.4', 'IPS.5', 'IPS.6', 'IPS.7', 'IPS.8',
            'IPK'
        ])

        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.markdown("## Hasil Prediksi")
        if prediction[0] == 1:
            st.success(f"### Prediksi: TEPAT LULUS (Probabilitas: {prediction_proba[0][1]*100:.2f}%)")
        else:
            st.error(f"### Prediksi: TERLAMBAT LULUS (Probabilitas: {prediction_proba[0][0]*100:.2f}%)")

    except Exception as e:
        st.error(f"Terjadi error dalam pemrosesan: {str(e)}")
        st.info("Pastikan semua input sesuai dengan format data training.")

