import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered")

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Masukkan data mahasiswa untuk memprediksi status kelulusan.")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('prediksi_kelulusan.joblib')
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        else:
            feature_names = [
                'JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH',
                'UMUR', 'IPK ',  # <- perhatikan spasi jika ada
                'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4',
                'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8'
            ]
        return model, feature_names
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        return None, None

model, feature_names = load_model()
if model is None:
    st.stop()

# --- Form Input ---
with st.form("form_input"):
    st.header("Data Mahasiswa")

    col1, col2 = st.columns(2)
    with col1:
        jenis_kelamin = st.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Laki-laki" if x == 0 else "Perempuan")
        status_mahasiswa = st.radio("Status Mahasiswa", [0, 1], format_func=lambda x: "Aktif" if x == 0 else "Non-Aktif")
    with col2:
        status_nikah = st.radio("Status Nikah", [0, 1], format_func=lambda x: "Belum Menikah" if x == 0 else "Menikah")
        umur = st.number_input("Umur", min_value=17, max_value=60, value=22)

    st.subheader("IP Semester")
ips_values = []
ips_cols = st.columns(8)
for i in range(8):
    with ips_cols[i]:
        ips = st.number_input(f"IPS {i+1}", 0.0, 4.0, 3.0, 0.01, key=f"ips_{i}")
        ips_values.append(ips)

# ✅ Hitung IPK otomatis dari rata-rata IPS 1-8
ipk = round(np.mean(ips_values), 2)
st.info(f"🎓 IPK dihitung otomatis: {ipk}")


    submit = st.form_submit_button("Prediksi")

# --- Prediksi ---
if submit:
    try:
        input_data = {
            'JENIS KELAMIN': [jenis_kelamin],
            'STATUS MAHASISWA': [status_mahasiswa],
            'STATUS NIKAH': [status_nikah],
            'UMUR': [umur]
        }

        # Sesuaikan nama IPK dengan model
        if 'IPK ' in feature_names:
            input_data['IPK '] = [ipk]  # <- pakai spasi
        else:
            input_data['IPK'] = [ipk]

        # Tambahkan IPS
        for i in range(8):
            key_underscore = f'IPS_{i+1}'
            key_space = f'IPS {i+1}'
            if key_space in feature_names:
                input_data[key_space] = [ips_values[i]]
            elif key_underscore in feature_names:
                input_data[key_underscore] = [ips_values[i]]

        df_input = pd.DataFrame(input_data)[feature_names]

        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)

        label_map = {0: "Terlambat Lulus", 1: "Tepat Lulus"}
        icon_map = {0: "🔴", 1: "🟢"}

        st.success(f"### Prediksi: {icon_map[pred]} {label_map[pred]}")
        st.metric("Probabilitas", f"{proba[0][pred]*100:.2f}%")

        # Grafik probabilitas
        df_proba = pd.DataFrame({
            "Status": ["Terlambat Lulus", "Tepat Lulus"],
            "Probabilitas": [proba[0][0], proba[0][1]]
        }).set_index("Status")
        st.bar_chart(df_proba)

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi:\n{str(e)}")
        st.info(f"Fitur yang diperlukan:\n{feature_names}")
