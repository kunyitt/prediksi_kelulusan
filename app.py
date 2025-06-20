import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Konfigurasi tampilan
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

# Judul Aplikasi
st.markdown("# Prediksi Kelulusan Mahasiswa")
st.markdown("Masukkan data mahasiswa untuk memprediksi apakah akan **Tepat** atau **Terlambat** lulus.")

# Garis pemisah
st.markdown("---")

# Load Model dan Encoder
@st.cache_resource
def load_model():
    model = joblib.load('prediksi_kelulusan.joblib')
    encoders = joblib.load('encoders.joblib')
    return model, encoders

model, encoders = load_model()

# Input Data dalam bentuk expander
with st.expander("Input Data Mahasiswa", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Jenis Kelamin")
        jenis_kelamin = st.radio(
            "Pilih jenis kelamin",
            ["LAKI-LAKI", "PEREMPUAN"],
            index=0,
            label_visibility="collapsed"
        )
        st.checkbox("✔", value=True, key="jk_check")
        
        st.markdown("### Status Mahasiswa")
        status_mahasiswa = st.radio(
            "Pilih status",
            ["BEKERJA", "TIDAK BEKERJA"],
            index=0,
            label_visibility="collapsed"
        )
        st.checkbox("✔", value=True, key="status_check")
        
        st.markdown("### Status Nikah")
        status_nikah = st.radio(
            "Pilih status",
            ["BELUM MENIKAH", "MENIKAH"],
            index=0,
            label_visibility="collapsed"
        )
        st.checkbox("✔", value=True, key="nikah_check")
        
    with col2:
        st.markdown("### Umur")
        umur = st.number_input(
            "Masukkan umur",
            min_value=17,
            max_value=50,
            value=22,
            label_visibility="collapsed"
        )
        st.markdown("↕️")
        
        st.markdown("### IP Semester")
        cols = st.columns(8)
        ips_values = []
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
                st.markdown("↕️")
        
        st.markdown("### IPK")
        ipk = st.number_input(
            "Masukkan IPK",
            min_value=0.0,
            max_value=4.0,
            value=3.0,
            step=0.01,
            label_visibility="collapsed"
        )
        st.markdown("↕️")

# Garis pemisah
st.markdown("---")

# Tombol Prediksi
if st.button("**PREDIKSI**", type="primary", use_container_width=True):
    # Preprocessing data
    input_data = {
        'JENIS KELAMIN': jenis_kelamin,
        'STATUS MAHASISWA': status_mahasiswa,
        'STATUS NIKAH': status_nikah,
        'UMUR': umur,
        'IPS.1': ips_values[0],
        'IPS.2': ips_values[1],
        'IPS.3': ips_values[2],
        'IPS.4': ips_values[3],
        'IPS.5': ips_values[4],
        'IPS.6': ips_values[5],
        'IPS.7': ips_values[6],
        'IPS.8': ips_values[7],
        'IPK': ipk
    }
    
    # Convert ke DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Label Encoding
    for col in ['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH']:
        input_df[col] = encoders[col].transform(input_df[col])
    
    # Prediksi
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Tampilkan hasil
    st.markdown("## Hasil Prediksi")
    
    if prediction[0] == 1:
        st.success(f"### Prediksi: TEPAT LULUS (Probabilitas: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.error(f"### Prediksi: TERLAMBAT LULUS (Probabilitas: {prediction_proba[0][0]*100:.2f}%)")

# Catatan kaki
st.markdown("---")
st.caption("Aplikasi Prediksi Kelulusan Mahasiswa © 2023")
