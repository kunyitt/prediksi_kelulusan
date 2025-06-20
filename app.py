import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", layout="centered", icon="🎓")

st.title("🎓 Prediksi Kelulusan Mahasiswa")
st.write("Aplikasi ini memprediksi status kelulusan mahasiswa berdasarkan data yang Anda masukkan.")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """
    Loads the pre-trained machine learning model and its expected feature names.
    Caches the model to avoid reloading on every rerun.
    """
    try:
        model = joblib.load('prediksi_kelulusan.joblib')
        # Attempt to get feature names from the model itself (preferred)
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        else:
            # Fallback feature names if not available from the model.
            # ENSURE THESE EXACTLY MATCH YOUR MODEL'S TRAINING FEATURES!
            feature_names = [
                'JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH',
                'UMUR', 'IPK ', # Ensure this matches your model (with or without space)
                'IPS 1', 'IPS 2', 'IPS 3', 'IPS 4',
                'IPS 5', 'IPS 6', 'IPS 7', 'IPS 8'
            ]
        st.success("Model berhasil dimuat!")
        return model, feature_names
    except FileNotFoundError:
        st.error("Gagal memuat model: File 'prediksi_kelulusan.joblib' tidak ditemukan. Pastikan model berada di direktori yang sama.")
        return None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.info("Pastikan file model tidak rusak dan kompatibel dengan versi Joblib Anda.")
        return None, None

model, feature_names = load_model()

# Stop the app if model loading failed
if model is None:
    st.stop()

# --- Form Input ---
with st.form("form_input", clear_on_submit=False): # Set clear_on_submit to False if you want inputs to persist after submission
    st.header("Data Mahasiswa")

    col1, col2 = st.columns(2)
    with col1:
        jenis_kelamin = st.radio(
            "Jenis Kelamin",
            options=[0, 1],
            format_func=lambda x: "Laki-laki" if x == 0 else "Perempuan",
            help="0: Laki-laki, 1: Perempuan"
        )
        status_mahasiswa = st.radio(
            "Status Mahasiswa",
            options=[0, 1],
            format_func=lambda x: "Bekerja" if x == 0 else "Tidak Bekerja",
            help="0: Bekerja, 1: Tidak Bekerja"
        )
    with col2:
        status_nikah = st.radio(
            "Status Nikah",
            options=[0, 1],
            format_func=lambda x: "Belum Menikah" if x == 0 else "Menikah",
            help="0: Belum Menikah, 1: Menikah"
        )
        umur = st.number_input(
            "Umur (Tahun)",
            min_value=17,
            max_value=60,
            value=22,
            help="Umur mahasiswa saat ini."
        )

    st.subheader("Indeks Prestasi Semester (IPS)")
    st.info("Masukkan nilai IPS untuk setiap semester (skala 0.0 - 4.0).")

    ips_values = []
    # Use 4 columns for better layout of 8 IPS inputs
    ips_cols = st.columns(4)
    for i in range(8):
        with ips_cols[i % 4]: # Distribute across 4 columns
            ips = st.number_input(
                f"IPS {i+1}",
                min_value=0.0,
                max_value=4.0,
                value=3.0,
                step=0.01,
                key=f"ips_{i}",
                help=f"Nilai IPS Semester {i+1}."
            )
            ips_values.append(ips)

    # Calculate IPK automatically from IPS values
    ipk = round(np.mean(ips_values), 2)
    st.markdown(f"**🎓 IPK Otomatis Dihitung:** <span style='font-size:24px; color:green;'>{ipk}</span>", unsafe_allow_html=True)


    submitted = st.form_submit_button("Prediksi Kelulusan")

# --- Prediction Logic ---
if submitted:
    try:
        input_data = {
            'JENIS KELAMIN': [jenis_kelamin],
            'STATUS MAHASISWA': [status_mahasiswa],
            'STATUS NIKAH': [status_nikah],
            'UMUR': [umur]
        }

        # Dynamically add IPK based on feature_names
        if 'IPK ' in feature_names:
            input_data['IPK '] = [ipk]
        elif 'IPK' in feature_names: # Fallback for 'IPK' without space
            input_data['IPK'] = [ipk]
        else:
            st.error("Nama fitur IPK (IPK atau IPK ) tidak ditemukan dalam model. Periksa penamaan fitur model Anda.")
            st.stop()


        # Dynamically add IPS values based on feature_names (handling 'IPS X' or 'IPS_X')
        for i in range(8):
            key_space = f'IPS {i+1}'
            key_underscore = f'IPS_{i+1}'
            if key_space in feature_names:
                input_data[key_space] = [ips_values[i]]
            elif key_underscore in feature_names:
                input_data[key_underscore] = [ips_values[i]]
            else:
                st.error(f"Nama fitur IPS {i+1} (IPS {i+1} atau IPS_{i+1}) tidak ditemukan dalam model. Periksa penamaan fitur model Anda.")
                st.stop()


        # Create DataFrame, ensuring column order matches model's expectations
        df_input = pd.DataFrame(input_data)
        # Reindex to ensure columns are in the same order as model's training features
        df_input = df_input[feature_names]

        # Make prediction
        pred = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)

        label_map = {0: "Terlambat Lulus", 1: "Tepat Lulus"}
        icon_map = {0: "🔴", 1: "🟢"}

        st.markdown("---")
        st.subheader("Hasil Prediksi")
        st.markdown(f"### Status Kelulusan: {icon_map[pred]} **{label_map[pred]}**")
        st.metric("Probabilitas", f"{proba[0][pred]*100:.2f}%")

        # Probability chart
        st.markdown("---")
        st.subheader("Distribusi Probabilitas")
        df_proba = pd.DataFrame({
            "Status": ["Terlambat Lulus", "Tepat Lulus"],
            "Probabilitas": [proba[0][0], proba[0][1]]
        }).set_index("Status")
        st.bar_chart(df_proba)

    except KeyError as ke:
        st.error(f"❌ Kesalahan data: Salah satu fitur yang diperlukan oleh model tidak ditemukan atau namanya tidak cocok: `{ke}`.")
        st.info(f"Pastikan nama fitur yang dimasukkan sesuai dengan yang diharapkan oleh model Anda. Fitur yang diharapkan: `{feature_names}`")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memprediksi:\n{str(e)}")
        st.info("Silakan periksa kembali input Anda atau hubungi administrator jika masalah berlanjut.")
