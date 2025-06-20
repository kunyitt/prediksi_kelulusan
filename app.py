import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, encoder, dan urutan kolom fitur
model = joblib.load('model_kelulusan.pkl')
encoders = joblib.load('encoders.pkl')
fitur_model = joblib.load('fitur_model.pkl')  # urutan kolom saat training

st.title("🎓 Prediksi Kelulusan Mahasiswa")

with st.form("form_prediksi"):
    # Input fitur kategorikal
    jenis_kelamin = st.selectbox("Jenis Kelamin", encoders['JENIS KELAMIN'].classes_)
    status_mahasiswa = st.selectbox("Status Mahasiswa", encoders['STATUS MAHASISWA'].classes_)
    status_nikah = st.selectbox("Status Nikah", encoders['STATUS NIKAH'].classes_)

    # Input umur
    umur = st.number_input("Umur", min_value=15, max_value=100)

    # Input IPS 1-8
    ips = []
    for i in range(1, 9):
        ips_val = st.number_input(f"IPS {i}", min_value=0.0, max_value=4.0, step=0.01)
        ips.append(ips_val)

    # IPK otomatis dari rata-rata IPS
    ipk = round(np.mean(ips), 2)
    st.info(f"IPK otomatis dihitung: {ipk}")

    submit = st.form_submit_button("Prediksi")

if submit:
    # Encode kategorikal
    input_data = {
        'JENIS KELAMIN': encoders['JENIS KELAMIN'].transform([jenis_kelamin])[0],
        'STATUS MAHASISWA': encoders['STATUS MAHASISWA'].transform([status_mahasiswa])[0],
        'UMUR': umur,
        'STATUS NIKAH': encoders['STATUS NIKAH'].transform([status_nikah])[0],
        'IPS 1': ips[0],
        'IPS 2': ips[1],
        'IPS 3': ips[2],
        'IPS 4': ips[3],
        'IPS 5': ips[4],
        'IPS 6': ips[5],
        'IPS 7': ips[6],
        'IPS 8': ips[7],
        'IPK': ipk
    }

    # Buat DataFrame dari input
    df_input = pd.DataFrame([input_data])
    
    # Bersihkan spasi pada nama kolom input & fitur_model
    df_input.columns = df_input.columns.str.strip()
    fitur_model = [col.strip() for col in fitur_model]
    
    # Pastikan urutan dan nama kolom cocok dengan saat training
        try:
        df_input = df_input[fitur_model]

        # Gunakan .to_numpy() agar aman dari error _check_feature_names
        input_array = df_input.to_numpy()

        # Prediksi label dan probabilitas
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0]

        # Ambil label hasil prediksi
        hasil = encoders['STATUS KELULUSAN'].inverse_transform([pred])[0]

        # Tampilkan hasil dan probabilitas
        st.success(f"Hasil Prediksi: Mahasiswa diperkirakan akan **{hasil.upper()}**")
        st.write("📊 Probabilitas:")
        for i, label in enumerate(encoders['STATUS KELULUSAN'].classes_):
            st.write(f"- {label}: {round(prob[i]*100, 2)}%")

    except KeyError as e:
        st.error(f"Terjadi error karena kolom tidak cocok: {e}")
        st.write("Kolom input saat ini:", df_input.columns.tolist())
        st.write("Kolom yang dibutuhkan model:", fitur_model)


