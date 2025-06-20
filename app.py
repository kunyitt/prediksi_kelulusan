import streamlit as st
import pandas as pd
import pickle

# --- FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_model():
    try:
        with open('trained_model.pkl', 'rb') as file:
            saved_data = pickle.load(file)
        return saved_data
    except FileNotFoundError:
        st.error("File model 'trained_model.pkl' tidak ditemukan. Pastikan file sudah dibuat dan ada di repositori.")
        return None

# Muat model, encoders, dan nama fitur
saved_data = load_model()

# Hanya lanjutkan jika model berhasil dimuat
if saved_data:
    model = saved_data['model']
    encoders = saved_data['encoders']
    feature_names = saved_data['feature_names']

    # --- UI APLIKASI ---
    st.title('Aplikasi Prediksi Kelulusan Mahasiswa')
    st.write("Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus TEPAT WAKTU atau TERLAMBAT.")
    
    st.sidebar.header('Input Data Mahasiswa')

    # Dapatkan label asli untuk dropdown dari encoders
    jk_labels = list(encoders['JENIS KELAMIN'].classes_)
    sm_labels = list(encoders['STATUS MAHASISWA'].classes_)
    sn_labels = list(encoders['STATUS NIKAH'].classes_)

    # Input dari pengguna
    jenis_kelamin = st.sidebar.selectbox('Jenis Kelamin', jk_labels)
    status_mahasiswa = st.sidebar.selectbox('Status Mahasiswa', sm_labels)
    status_nikah = st.sidebar.selectbox('Status Nikah', sn_labels)
    
    ips1 = st.sidebar.slider('IPS Semester 1', 0.0, 4.0, 3.0)
    ips2 = st.sidebar.slider('IPS Semester 2', 0.0, 4.0, 3.1)
    ips3 = st.sidebar.slider('IPS Semester 3', 0.0, 4.0, 3.2)
    ips4 = st.sidebar.slider('IPS Semester 4', 0.0, 4.0, 3.3)
    ips5 = st.sidebar.slider('IPS Semester 5', 0.0, 4.0, 3.4)
    ips6 = st.sidebar.slider('IPS Semester 6', 0.0, 4.0, 3.5)
    ips7 = st.sidebar.slider('IPS Semester 7', 0.0, 4.0, 3.6)
    ips8 = st.sidebar.slider('IPS Semester 8', 0.0, 4.0, 3.7)
    ipk = st.sidebar.slider('IPK', 0.0, 4.0, 3.5)

    # Tombol Prediksi
    if st.sidebar.button('Prediksi Kelulusan'):
        # 1. Buat DataFrame dari input
        input_data = {
            'JENIS KELAMIN': [jenis_kelamin], 'STATUS MAHASISWA': [status_mahasiswa],
            'STATUS NIKAH': [status_nikah], 'IPS1': [ips1], 'IPS2': [ips2],
            'IPS3': [ips3], 'IPS4': [ips4], 'IPS5': [ips5], 'IPS6': [ips6],
            'IPS7': [ips7], 'IPS8': [ips8], 'IPK': [ipk]
        }
        input_df = pd.DataFrame(input_data)

        # 2. Encode input kategorikal menggunakan encoder yang disimpan
        for col in ['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH']:
            input_df[col] = encoders[col].transform(input_df[col])
        
        # 3. Pastikan urutan kolom benar
        input_df = input_df[feature_names]

        # 4. Lakukan prediksi
        prediction = model.predict(input_df.values)
        prediction_proba = model.predict_proba(input_df.values)

        # 5. Ubah hasil prediksi ke label asli
        pred_label = encoders['STATUS KELULUSAN'].inverse_transform(prediction)[0]
        
        # Tampilkan hasil
        st.subheader('Hasil Prediksi:')
        if pred_label == 'TEPAT WAKTU':
            st.success(f'Mahasiswa diprediksi akan lulus **{pred_label}**')
        else:
            st.warning(f'Mahasiswa diprediksi akan lulus **{pred_label}**')
        
        st.write('**Tingkat Keyakinan Model (Probabilitas):**')
        proba_df = pd.DataFrame(prediction_proba, columns=encoders['STATUS KELULUSAN'].classes_)
        st.write(proba_df)
