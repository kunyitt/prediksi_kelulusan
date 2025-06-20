import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt # Tetap diperlukan untuk visualisasi
import seaborn as sns          # Tetap diperlukan untuk visualisasi

# --- FUNGSI UNTUK MEMUAT MODEL YANG SUDAH DILATIH ---
@st.cache_resource
def load_model_and_data():
    """
    Fungsi ini akan memuat model, encoders, dan data asli
    dari file .pkl dan .csv. Hasilnya akan disimpan di cache.
    """
    try:
        # Muat dari file .pkl
        with open('trained_model.pkl', 'rb') as file:
            saved_data = pickle.load(file)

        # Muat data asli untuk visualisasi
        df_original = pd.read_csv('Kelulusan Train.csv')
        if 'NAMA' in df_original.columns:
            df_original.drop(columns=["NAMA"], inplace=True)
        
        # Label encode data asli untuk heatmap (menggunakan encoder yang disimpan)
        df_encoded = df_original.copy()
        for col, le in saved_data['encoders'].items():
            if col in df_encoded.columns:
                df_encoded[col] = le.transform(df_encoded[col])

        return saved_data['model'], saved_data['encoders'], saved_data['feature_names'], df_original, df_encoded
        
    except FileNotFoundError:
        st.error("File 'trained_model.pkl' atau 'Kelulusan Train.csv' tidak ditemukan. Pastikan file-file tersebut ada di repositori GitHub Anda.")
        return None, None, None, None, None

# Memanggil fungsi untuk memuat model
model, encoders, feature_names, df_raw, df_encoded = load_model_and_data()

# --- UI APLIKASI ---
st.title('Aplikasi Prediksi Status Kelulusan Mahasiswa')
st.write("Aplikasi ini memprediksi status kelulusan berdasarkan data input menggunakan model yang sudah dilatih.")

# Hanya tampilkan aplikasi jika model berhasil dimuat
if model is not None:
    st.sidebar.header('Input Data Mahasiswa')

    def user_input_features():
        # Dapatkan pilihan unik dari data mentah
        jenis_kelamin = st.sidebar.selectbox('Jenis Kelamin', df_raw['JENIS KELAMIN'].unique())
        status_mahasiswa = st.sidebar.selectbox('Status Mahasiswa', df_raw['STATUS MAHASISWA'].unique())
        status_nikah = st.sidebar.selectbox('Status Nikah', df_raw['STATUS NIKAH'].unique())
        
        ips1 = st.sidebar.slider('IPS Semester 1', 0.0, 4.0, 3.0)
        ips2 = st.sidebar.slider('IPS Semester 2', 0.0, 4.0, 3.1)
        ips3 = st.sidebar.slider('IPS Semester 3', 0.0, 4.0, 3.2)
        ips4 = st.sidebar.slider('IPS Semester 4', 0.0, 4.0, 3.3)
        ips5 = st.sidebar.slider('IPS Semester 5', 0.0, 4.0, 3.4)
        ips6 = st.sidebar.slider('IPS Semester 6', 0.0, 4.0, 3.5)
        ips7 = st.sidebar.slider('IPS Semester 7', 0.0, 4.0, 3.6)
        ips8 = st.sidebar.slider('IPS Semester 8', 0.0, 4.0, 3.7)
        ipk = st.sidebar.slider('IPK', 0.0, 4.0, 3.5)

        data = {'JENIS KELAMIN': jenis_kelamin, 'STATUS MAHASISWA': status_mahasiswa,
                'STATUS NIKAH': status_nikah, 'IPS1': ips1, 'IPS2': ips2, 'IPS3': ips3,
                'IPS4': ips4, 'IPS5': ips5, 'IPS6': ips6, 'IPS7': ips7, 'IPS8': ips8, 'IPK': ipk}
        
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

    st.subheader('Prediksi Status Kelulusan')
    if st.sidebar.button('Prediksi'):
        predict_df = input_df.copy()
        for col in ['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH']:
            predict_df[col] = encoders[col].transform(predict_df[col])
        predict_df = predict_df[feature_names]

        prediction = model.predict(predict_df.values)
        prediction_proba = model.predict_proba(predict_df.values)
        pred_label = encoders['STATUS KELULUSAN'].inverse_transform(prediction)[0]
        
        if pred_label == 'TEPAT WAKTU':
            st.success(f'**Hasil Prediksi: {pred_label}**')
        else:
            st.warning(f'**Hasil Prediksi: {pred_label}**')
        
        st.write('**Probabilitas:**')
        proba_df = pd.DataFrame(prediction_proba, columns=encoders['STATUS KELULUSAN'].classes_)
        st.write(proba_df)
    
    # --- VISUALISASI DATA ---
    st.subheader('Analisis Data Latih')
    with st.expander("Lihat Heatmap Korelasi Fitur"):
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

    with st.expander("Lihat Tingkat Kepentingan Fitur"):
        importances = model.feature_importances_
        forest_importances = pd.Series(importances, index=feature_names)
        fig, ax = plt.subplots(figsize=(12, 7))
        forest_importances.sort_values().plot(kind='barh', color='skyblue', ax=ax)
        ax.set_title("Pentingnya Fitur menurut Random Forest")
        ax.set_xlabel("Skor Pentingnya Fitur")
        plt.tight_layout()
        st.pyplot(fig)
