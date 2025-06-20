import streamlit as st
import pandas as pd
import pickle

# --- FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_model():
    """
    Memuat model, encoders, dan nama fitur dari file .pkl yang sudah disimpan.
    """
    try:
        with open('trained_model.pkl', 'rb') as file:
            saved_data = pickle.load(file)
        return saved_data
    except FileNotFoundError:
        st.error("File model 'trained_model.pkl' tidak ditemukan. Pastikan file sudah dibuat (dengan menjalankan create_model.py) dan ada di repositori GitHub Anda.")
        return None

# Muat semua data yang dibutuhkan dari file .pkl
saved_data = load_model()

# Hanya tampilkan aplikasi jika model berhasil dimuat
if saved_data:
    model = saved_data['model']
    encoders = saved_data['encoders']
    feature_names = saved_data['feature_names']

    # --- UI APLIKASI ---
    st.title('Aplikasi Prediksi Kelulusan Mahasiswa')
    st.write("Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus TEPAT WAKTU atau TERLAMBAT.")
    
    st.sidebar.header('Input Data Mahasiswa')

    # Dapatkan label asli untuk dropdown dari encoders
    # Ini memastikan label di UI cocok dengan data training
    jk_labels = list(encoders['JENIS KELAMIN'].classes_)
    sm_labels = list(encoders['STATUS MAHASISWA'].classes_)
    sn_labels = list(encoders['STATUS NIKAH'].classes_)

    # Input dari pengguna di sidebar
    jenis_kelamin = st.sidebar.selectbox('Jenis Kelamin', jk_labels)
    status_mahasiswa = st.sidebar.selectbox('Status Mahasiswa', sm_labels)
    status_nikah = st.sidebar.selectbox('Status Nikah', sn_labels)
    
    ips1 = st.sidebar.slider('IPS Semester 1', 0.0, 4.0, 3.00, 0.01)
    ips2 = st.sidebar.slider('IPS Semester 2', 0.0, 4.0, 3.10, 0.01)
    ips3 = st.sidebar.slider('IPS Semester 3', 0.0, 4.0, 3.20, 0.01)
    ips4 = st.sidebar.slider('IPS Semester 4', 0.0, 4.0, 3.30, 0.01)
    ips5 = st.sidebar.slider('IPS Semester 5', 0.0, 4.0, 3.40, 0.01)
    ips6 = st.sidebar.slider('IPS Semester 6', 0.0, 4.0, 3.50, 0.01)
    ips7 = st.sidebar.slider('IPS Semester 7', 0.0, 4.0, 3.60, 0.01)
    ips8 = st.sidebar.slider('IPS Semester 8', 0.0, 4.0, 3.70, 0.01)
    ipk = st.sidebar.slider('IPK', 0.0, 4.0, 3.50, 0.01)

    # Tombol Prediksi
    if st.sidebar.button('Prediksi Kelulusan'):
        
        # ====================================================================
        # BAGIAN YANG DIPERBAIKI ADA DI SINI
        # ====================================================================

        # 1. Kumpulkan semua nilai input ke dalam sebuah list
        # Urutannya HARUS SAMA PERSIS dengan urutan kolom saat training
        input_values = [
            jenis_kelamin, status_mahasiswa, status_nikah,
            ips1, ips2, ips3, ips4, ips5, ips6, ips7, ips8, ipk
        ]

        # 2. Buat DataFrame langsung menggunakan feature_names dari file .pkl
        # Ini memastikan nama dan urutan kolom 100% benar
        input_df = pd.DataFrame([input_values], columns=feature_names)

        # 3. Lakukan encoding pada kolom kategorikal di DataFrame yang baru dibuat
        for col in ['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH']:
            # Gunakan encoder yang sesuai untuk mengubah label string menjadi angka
            input_df[col] = encoders[col].transform(input_df[col])
        
        # Baris `input_df = input_df[feature_names]` yang menyebabkan error
        # sekarang sudah tidak dibutuhkan lagi dan bisa dihapus.

        # 4. Lakukan prediksi menggunakan data yang sudah siap
        prediction = model.predict(input_df.values)
        prediction_proba = model.predict_proba(input_df.values)

        # 5. Ubah hasil prediksi (angka) kembali ke label asli (string)
        pred_label = encoders['STATUS KELULUSAN'].inverse_transform(prediction)[0]
        
        # Tampilkan hasil prediksi
        st.subheader('Hasil Prediksi:')
        if pred_label == 'TEPAT WAKTU':
            st.success(f'Mahasiswa diprediksi akan lulus **{pred_label}**')
        else:
            st.warning(f'Mahasiswa diprediksi akan lulus **{pred_label}**')
        
        st.write('**Tingkat Keyakinan Model (Probabilitas):**')
        proba_df = pd.DataFrame(prediction_proba, columns=encoders['STATUS KELULUSAN'].classes_)
        st.write(proba_df)
