import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Konfigurasi tampilan
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

# Judul Aplikasi
st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status kelulusan")

# Load model dan fitur
@st.cache_resource
def load_model():
    try:
        model = joblib.load('prediksi_kelulusan.joblib')
        
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        else:
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

# Form input
with st.form("input_form"):
    st.header("Data Mahasiswa")
    
    st.subheader("Data Kategorikal")
    col1, col2 = st.columns(2)
    
    with col1:
        jenis_kelamin = st.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Laki-laki" if x == 0 else "Perempuan")
        status_mahasiswa = st.radio("Status Mahasiswa", [0, 1], format_func=lambda x: "Aktif" if x == 0 else "Non-Aktif")
    
    with col2:
        status_nikah = st.radio("Status Nikah", [0, 1], format_func=lambda x: "Belum Menikah" if x == 0 else "Menikah")
        ipk = st.slider("IPK", 0.0, 4.0, 3.0, 0.01)

    st.subheader("IP Semester")
    ips_cols = st.columns(8)
    ips_values = []
    for i in range(8):
        with ips_cols[i]:
            ips = st.number_input(f"Sem {i+1}", min_value=0.0, max_value=4.0, value=3.0, step=0.01, key=f"ips_{i}")
            ips_values.append(ips)

    submitted = st.form_submit_button("Prediksi")

# Proses prediksi
if submitted:
    try:
        input_data = {
            'JENIS KELAMIN': [jenis_kelamin],
            'STATUS MAHASISWA': [status_mahasiswa],
            'STATUS NIKAH': [status_nikah],
            'IPK': [ipk]
        }

        # Tambahkan kolom IPS dengan nama menyesuaikan training
        for i in range(8):
            col_with_underscore = f'IPS_{i+1}'
            col_with_space = f'IPS {i+1}'
            if col_with_underscore in feature_names:
                input_data[col_with_underscore] = [ips_values[i]]
            elif col_with_space in feature_names:
                input_data[col_with_space] = [ips_values[i]]

        df_input = pd.DataFrame(input_data)[feature_names]

        prediction = model.predict(df_input)
        proba = model.predict_proba(df_input)

        st.success("### Hasil Prediksi")
        result_mapping = {
            0: ("Terlambat Lulus", "🔴"),
            1: ("Tepat Lulus", "🟢")
        }

        pred_label, pred_icon = result_mapping[prediction[0]]
        st.metric(
            label="Status Kelulusan",
            value=f"{pred_icon} {pred_label}",
            delta=f"Probabilitas: {proba[0][prediction[0]]*100:.2f}%"
        )

        st.write("**Detail Probabilitas:**")
        st.bar_chart(pd.DataFrame({
            'Kelas': ['Terlambat Lulus', 'Tepat Lulus'],
            'Probabilitas': [proba[0][0], proba[0][1]]
        }).set_index('Kelas'))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
        st.warning("Periksa kembali urutan dan nama kolom input sesuai data training.")

# Sidebar info
st.sidebar.info(
    "**Petunjuk Pengisian:**\n"
    "- Jenis Kelamin: 0=Laki-laki, 1=Perempuan\n"
    "- Status Mahasiswa: 0=Aktif, 1=Non-Aktif\n"
    "- Status Nikah: 0=Belum Menikah, 1=Menikah\n"
    f"\n**Fitur Model:**\n{feature_names}"
)
