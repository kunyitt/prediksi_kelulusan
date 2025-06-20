import streamlit as st
import pandas as pd
import joblib

# Konfigurasi tampilan
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

# Judul Aplikasi
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status kelulusan")

# 1. Load Model dengan validasi fitur
@st.cache_resource
def load_model():
    try:
        model = joblib.load('prediksi_kelulusan.joblib')
        
        # Dapatkan nama fitur dari model
        if hasattr(model, 'feature_names_in_'):
            required_features = model.feature_names_in_.tolist()
        else:
            # Jika model tidak menyimpan nama fitur, gunakan default
            required_features = [
                'JENIS KELAMIN', 
                'STATUS MAHASISWA',  # Perhatikan penulisan 'MAHASISWA' vs 'MAHASISWA'
                'UMUR',
                'STATUS NIKAH',
                'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4',
                'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8',
                'IPK'  # Pastikan tidak ada spasi di akhir
            ]
        
        return model, required_features
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None

model, required_features = load_model()

if model is None:
    st.stop()

# Debug: Tampilkan fitur yang dibutuhkan
st.sidebar.info(f"Fitur yang dibutuhkan model:\n{required_features}")

# 2. Input Data - Pastikan nama kolom sama persis
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
        status_mahasiswa = st.radio(
            "Status Mahasiswa",  # Pastikan nama ini sesuai dengan fitur training
            options=[0, 1],
            format_func=lambda x: "Bekerja" if x == 0 else "Tidak Bekerja"
        )
        
        ipk = st.slider(
            "IPK",  # Pastikan nama ini sama persis dengan training (termasuk huruf besar/kecil)
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
                f"IPS {i}",  # Format harus sama dengan training ('IPS 1' bukan 'IPS1')
                min_value=0.0,
                max_value=4.0,
                value=3.0,
                step=0.01,
                key=f"ips_{i}"
            )
            ips_values.append(ips)
    
    submitted = st.form_submit_button("Prediksi")

# 3. Proses Prediksi dengan validasi ketat
if submitted:
    try:
        # Membuat dictionary dengan nama kolom yang sama persis
        input_data = {
            'JENIS KELAMIN': [jenis_kelamin],
            'STATUS MAHASISWA': [status_mahasiswa],  # Pastikan penulisan sama
            'UMUR': [umur],
            'STATUS NIKAH': [status_nikah],
            'IPK': [ipk]  # Tidak ada spasi di akhir
        }
        
        # Tambahkan IPS dengan format yang benar
        for i in range(1, 9):
            input_data[f'IPS {i}'] = [ips_values[i-1]]  # Format: 'IPS 1' bukan 'IPS1'
        
        # Debug: Tampilkan data input
        st.sidebar.write("Data Input:", input_data.keys())
        
        # Buat DataFrame dengan urutan kolom yang tepat
        df_input = pd.DataFrame(input_data)
        
        # Pastikan semua fitur ada
        missing_features = [f for f in required_features if f not in df_input.columns]
        if missing_features:
            raise ValueError(f"Fitur yang kurang: {missing_features}")
        
        # Urutkan kolom sesuai dengan training
        df_input = df_input[required_features]
        
        # Prediksi
        prediction = model.predict(df_input)
        prediction_proba = model.predict_proba(df_input)
        
        # Tampilkan hasil
        st.success("### Hasil Prediksi")
        
        result_mapping = {
            0: ("Terlambat Lulus", "🔴"),
            1: ("Tepat Lulus", "🟢")
        }
        
        label, icon = result_mapping[prediction[0]]
        probability = prediction_proba[0][prediction[0]] * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Status Kelulusan", value=f"{icon} {label}")
        with col2:
            st.metric(label="Probabilitas", value=f"{probability:.2f}%")
        
        # Visualisasi probabilitas
        st.bar_chart(pd.DataFrame({
            'Status': ['Terlambat Lulus', 'Tepat Lulus'],
            'Probabilitas': [prediction_proba[0][0], prediction_proba[0][1]]
        }).set_index('Status'))
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.error("""
        SOLUSI:
        1. Periksa penulisan nama kolom (huruf besar/kecil dan spasi)
        2. Pastikan semua fitur diisi
        3. Cek format data (angka/string)
        """)
        st.write(f"Fitur yang dibutuhkan model: {required_features}")
        st.write(f"Fitur yang Anda berikan: {list(input_data.keys())}")

# Panduan penggunaan
st.sidebar.info("""
**Panduan Penting:**
1. Nama kolom HARUS sama persis dengan saat training
2. Perhatikan:
   - Huruf besar/kecil
   - Spasi (misal 'IPS 1' bukan 'IPS1')
   - Karakter khusus
3. Semua fitur harus diisi
""")
