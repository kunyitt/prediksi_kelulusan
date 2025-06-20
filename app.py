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
        
        # Dapatkan nama fitur dari model (termasuk yang ada spasi)
        if hasattr(model, 'feature_names_in_'):
            required_features = [f.strip() for f in model.feature_names_in_.tolist()]  # Bersihkan spasi
        else:
            required_features = [
                'JENIS KELAMIN', 'STATUS MAHASISWA', 'UMUR', 'STATUS NIKAH',
                'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4',
                'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8',
                'IPK '  # Tanpa spasi
            ]
        
        return model, required_features
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None

model, required_features = load_model()

if model is None:
    st.stop()

# 2. Input Data - Penyesuaian untuk 'IPK ' dengan spasi
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
            "Status Mahasiswa",
            options=[0, 1],
            format_func=lambda x: "Bekerja" if x == 0 else "Tidak Bekerja"
        )
        
        # Penyesuaian khusus untuk 'IPK ' dengan spasi
        ipk = st.slider(
            "IPK",  # Tampilan untuk user
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

# 3. Proses Prediksi dengan penanganan spasi di 'IPK'
if submitted:
    try:
        # Membuat dictionary dengan nama kolom yang sesuai model (termasuk spasi di 'IPK ')
        input_data = {
            'JENIS KELAMIN': [jenis_kelamin],
            'STATUS MAHASISWA': [status_mahasiswa],
            'UMUR': [umur],
            'STATUS NIKAH': [status_nikah],
            'IPK ': [ipk]  # <- Perhatikan spasi di akhir!
        }
        
        # Tambahkan IPS
        for i in range(1, 9):
            input_data[f'IPS {i}'] = [ips_values[i-1]]
        
        # Buat DataFrame dan pastikan urutan kolom sesuai
        df_input = pd.DataFrame(input_data)[required_features]
        
        # Prediksi
        prediction = model.predict(df_input)
        prediction_proba = model.predict_proba(df_input)
        
        # Tampilkan hasil
        st.success("### Hasil Prediksi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Status Kelulusan",
                "Tepat Lulus" if prediction[0] == 1 else "Terlambat Lulus"
            )
        with col2:
            st.metric(
                "Probabilitas",
                f"{prediction_proba[0][prediction[0]]*100:.2f}%"
            )
        
        # Visualisasi
        st.bar_chart(pd.DataFrame({
            'Status': ['Terlambat Lulus', 'Tepat Lulus'],
            'Probabilitas': [prediction_proba[0][0], prediction_proba[0][1]]
        }).set_index('Status'))
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.error("""
        Pastikan:
        1. Format nama kolom sama persis dengan training
        2. Khusus untuk IPK, gunakan format dengan spasi di akhir
        """)

# Panduan di Sidebar
st.sidebar.info("""
**Catatan Penting:**
1. Sistem mengharuskan penulisan 'IPK ' dengan spasi di akhir
2. Format IPS: 'IPS 1' sampai 'IPS 8' (dengan spasi)
3. Semua nilai harus diisi
""")
