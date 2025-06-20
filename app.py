import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi tampilan
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

# Judul Aplikasi
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status kelulusan")

# 1. Load Model dengan pengecekan fitur
@st.cache_resource
def load_model():
    try:
        model = joblib.load('prediksi_kelulusan.joblib')
        
        # Dapatkan nama fitur yang digunakan saat training
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        else:
            # Jika model tidak menyimpan nama fitur, gunakan default
            feature_names = [
                'JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH',
                'IPK', 'IPS_1', 'IPS_2', 'IPS_3', 'IPS_4',
                'IPS_5', 'IPS_6', 'IPS_7', 'IPS_8'
            ]
        
        return model, feature_names
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None

model, feature_names = load_model()

if model is None:
    st.stop()

# 2. Input Data - Pastikan sesuai dengan feature_names
with st.form("input_form"):
    st.header("Data Mahasiswa")
    
    # Mapping nilai untuk fitur kategorikal
    st.subheader("Data Kategorikal")
    col1, col2 = st.columns(2)
    
    with col1:
        jenis_kelamin = st.radio(
            "Jenis Kelamin",
            [0, 1],
            format_func=lambda x: "Laki-laki" if x == 0 else "Perempuan"
        )
        
        status_mahasiswa = st.radio(
            "Status Mahasiswa",
            [0, 1],
            format_func=lambda x: "Aktif" if x == 0 else "Non-Aktif"
        )
    
    with col2:
        status_nikah = st.radio(
            "Status Nikah",
            [0, 1],
            format_func=lambda x: "Belum Menikah" if x == 0 else "Menikah"
        )
        
        ipk = st.slider("IPK", 0.0, 4.0, 3.0, 0.01)
    
    # Input IPS per semester
    st.subheader("IP Semester")
    ips_cols = st.columns(8)
    ips_values = []
    for i in range(8):
        with ips_cols[i]:
            ips = st.number_input(
                f"Sem {i+1}",
                min_value=0.0,
                max_value=4.0,
                value=3.0,
                step=0.01,
                key=f"ips_{i}"
            )
            ips_values.append(ips)
    
    submitted = st.form_submit_button("Prediksi")

# 3. Proses Prediksi dengan validasi fitur
if submitted:
    try:
        # Membuat DataFrame dengan urutan fitur yang benar
        input_data = {
            'JENIS KELAMIN': [jenis_kelamin],
            'STATUS MAHASISWA': [status_mahasiswa],
            'STATUS NIKAH': [status_nikah],
            'IPK': [ipk]
        }
        
        # Tambahkan IPS ke input data
        for i in range(8):
            input_data[f'IPS_{i+1}'] = [ips_values[i]]
        
        # Buat DataFrame dengan urutan fitur yang sesuai training
        df_input = pd.DataFrame(input_data)[feature_names]
        
        # Prediksi
        prediction = model.predict(df_input)
        proba = model.predict_proba(df_input)
        
        # Tampilkan hasil
        st.success("### Hasil Prediksi")
        
        # Mapping hasil prediksi
        result_mapping = {
            0: ("Tidak Lulus", "🔴"),
            1: ("Lulus", "🟢")
        }
        
        pred_label, pred_icon = result_mapping[prediction[0]]
        
        st.metric(
            label="Status Kelulusan",
            value=f"{pred_icon} {pred_label}",
            delta=f"Probabilitas: {proba[0][prediction[0]]*100:.2f}%"
        )
        
        # Tampilkan detail probabilitas
        st.write("**Detail Probabilitas:**")
        proba_data = {
            'Kelas': ['Tidak Lulus', 'Lulus'],
            'Probabilitas': [proba[0][0], proba[0][1]]
        }
        st.bar_chart(pd.DataFrame(proba_data).set_index('Kelas'))
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.error("Pastikan semua input sesuai dengan data training")
        st.write(f"Fitur yang dibutuhkan: {feature_names}")

# Informasi Fitur
st.sidebar.info(
    "**Panduan Input:**\n"
    "- Pastikan semua fitur diisi\n"
    "- Gunakan format yang sesuai\n"
    f"\n**Fitur yang dibutuhkan:**\n{feature_names}"
)
