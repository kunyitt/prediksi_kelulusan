import streamlit as st
import pandas as pd
import joblib

# Konfigurasi tampilan
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

# Judul Aplikasi
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status kelulusan")

# 1. Load Model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('prediksi_kelulusan.joblib')
        # Fitur yang dibutuhkan model (sesuaikan dengan yang ada di model Anda)
        required_features = [
            'JENIS KELAMIN', 
            'STATUS MAHASISWA', 
            'UMUR',
            'STATUS NIKAH',
            'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4',
            'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8',
            'IPK'
        ]
        return model, required_features
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None

model, required_features = load_model()

if model is None:
    st.stop()

# 2. Input Data
with st.form("input_form"):
    st.header("Data Mahasiswa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Identitas")
        jenis_kelamin = st.radio(
            "Jenis Kelamin",
            options=[0, 1],
            format_func=lambda x: "Laki-laki" if x == 0 else "Perempuan"
        )
        
        umur = st.number_input("Umur", min_value=17, max_value=50, value=22)
        
        status_nikah = st.radio(
            "Status Nikah",
            options=[0, 1],
            format_func=lambda x: "Belum Menikah" if x == 0 else "Menikah"
        )
    
    with col2:
        st.subheader("Status Akademik")
        # Penyesuaian untuk status mahasiswa (bekerja/tidak bekerja)
        status_mahasiswa = st.radio(
            "Status Mahasiswa",
            options=[0, 1],
            format_func=lambda x: "Bekerja" if x == 0 else "Tidak Bekerja",
            help="Pilih status pekerjaan mahasiswa"
        )
        
        ipk = st.slider(
            "IPK",
            min_value=0.0,
            max_value=4.0,
            value=3.0,
            step=0.01
        )
    
    st.subheader("IP Semester")
    ips_cols = st.columns(8)
    ips_values = []
    for i in range(1, 9):
        with ips_cols[i-1]:
            ips = st.number_input(
                f"IPS {i}",
                min_value=0.0,
                max_value=4.0,
                value=3.0,
                step=0.01,
                key=f"ips_{i}"
            )
            ips_values.append(ips)
    
    submitted = st.form_submit_button("Prediksi")

# 3. Proses Prediksi
if submitted:
    try:
        # Membuat dictionary input data
        input_data = {
            'JENIS KELAMIN': [jenis_kelamin],
            'STATUS MAHASISWA': [status_mahasiswa],
            'UMUR': [umur],
            'STATUS NIKAH': [status_nikah],
            'IPK': [ipk]
        }
        
        # Menambahkan IPS ke input data
        for i in range(1, 9):
            input_data[f'IPS {i}'] = [ips_values[i-1]]
        
        # Membuat DataFrame dengan urutan kolom yang benar
        df_input = pd.DataFrame(input_data)[required_features]
        
        # Melakukan prediksi
        prediction = model.predict(df_input)
        prediction_proba = model.predict_proba(df_input)
        
        # Menampilkan hasil
        st.success("### Hasil Prediksi")
        
        # Mapping hasil prediksi
        result_labels = {
            0: ("Terlambat Lulus", "🔴"),
            1: ("Tepat Lulus", "🟢")
        }
        
        label, icon = result_labels[prediction[0]]
        probability = prediction_proba[0][prediction[0]] * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Status Kelulusan", value=f"{icon} {label}")
        with col2:
            st.metric(label="Probabilitas", value=f"{probability:.2f}%")
        
        # Visualisasi probabilitas
        st.write("**Detail Probabilitas:**")
        prob_df = pd.DataFrame({
            'Status': ['Terlambat Lulus', 'Tepat Lulus'],
            'Probabilitas': [prediction_proba[0][0], prediction_proba[0][1]]
        })
        st.bar_chart(prob_df.set_index('Status'))
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.error("""
        Pastikan:
        1. Semua input telah diisi
        2. Format data sesuai
        3. Nama kolom tepat
        """)

# Informasi tambahan
st.sidebar.info("""
**Panduan Pengisian:**
- **Status Mahasiswa:**
  - 0 = Bekerja
  - 1 = Tidak Bekerja
- **Format IPS:** IPS 1 sampai IPS 8
- **IPK:** Gunakan titik sebagai desimal
""")
