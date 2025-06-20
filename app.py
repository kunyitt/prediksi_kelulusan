import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Konfigurasi tampilan
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

# Judul Aplikasi
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status kelulusan")

# 1. Load Model dan Encoder dengan mapping yang benar
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('prediksi_kelulusan.joblib')
        encoders = joblib.load('encoders.joblib')
        
        # Reverse mapping untuk LabelEncoder
        encoder_mappings = {}
        for col in encoders:
            if hasattr(encoders[col], 'classes_'):
                encoder_mappings[col] = {
                    'classes': encoders[col].classes_.tolist(),
                    'inverse_mapping': {i: label for i, label in enumerate(encoders[col].classes_)}
                }
        
        return model, encoders, encoder_mappings
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None, None

model, encoders, encoder_mappings = load_artifacts()

if model is None:
    st.stop()

# 2. Input Data dengan format yang sesuai
with st.form("input_form"):
    st.header("Data Mahasiswa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Jenis Kelamin (sesuaikan dengan mapping encoder)
        st.subheader("Jenis Kelamin")
        jk_mapping = {
            "Laki-laki (0)": 0,
            "Perempuan (1)": 1
        }
        jenis_kelamin = st.radio(
            "Pilih:",
            list(jk_mapping.keys()),
            index=0
        )
        jk_value = jk_mapping[jenis_kelamin]
        
        # Status Mahasiswa
        st.subheader("Status Mahasiswa")
        status_mapping = {
            "Aktif (0)": 0,
            "Non-Aktif (1)": 1
        }
        status_mhs = st.radio(
            "Pilih:",
            list(status_mapping.keys()),
            index=0
        )
        status_value = status_mapping[status_mhs]
        
    with col2:
        # Status Nikah
        st.subheader("Status Nikah")
        nikah_mapping = {
            "Belum Menikah (0)": 0,
            "Menikah (1)": 1
        }
        status_nikah = st.radio(
            "Pilih:",
            list(nikah_mapping.keys()),
            index=0
        )
        nikah_value = nikah_mapping[status_nikah]
        
        # Input numerik
        st.subheader("IPK")
        ipk = st.slider("Pilih IPK", 0.0, 4.0, 3.0, 0.01)
        
    submitted = st.form_submit_button("Prediksi")

# 3. Proses Prediksi
if submitted:
    try:
        # Membuat DataFrame input
        input_data = {
            'JENIS KELAMIN': [jk_value],
            'STATUS MAHASISWA': [status_value],
            'STATUS NIKAH': [nikah_value],
            'IPK': [ipk]
            # Tambahkan kolom lain sesuai kebutuhan
        }
        
        df_input = pd.DataFrame(input_data)
        
        # Prediksi
        prediction = model.predict(df_input)
        proba = model.predict_proba(df_input)
        
        # Tampilkan hasil
        st.success("### Hasil Prediksi")
        
        # Mapping hasil prediksi
        result_mapping = {
            0: ("Tidak Lulus", "danger"),
            1: ("Lulus", "success")
        }
        
        pred_label, pred_color = result_mapping[prediction[0]]
        
        st.metric(
            label="Status Kelulusan",
            value=pred_label,
            delta=f"Probabilitas: {proba[0][prediction[0]]*100:.2f}%"
        )
        
        # Tampilkan probabilitas
        st.write("Detail Probabilitas:")
        proba_df = pd.DataFrame({
            'Kelas': ['Tidak Lulus', 'Lulus'],
            'Probabilitas': [proba[0][0], proba[0][1]]
        })
        st.bar_chart(proba_df.set_index('Kelas'))
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.info("Pastikan semua input sudah sesuai")

# Informasi tambahan
st.sidebar.info(
    "Pastikan input sesuai dengan format data training:\n"
    f"Jenis Kelamin: {encoder_mappings.get('JENIS KELAMIN', {}).get('classes', 'N/A')}\n"
    f"Status Mahasiswa: {encoder_mappings.get('STATUS MAHASISWA', {}).get('classes', 'N/A')}"
)
