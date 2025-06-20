import streamlit as st
import pandas as pd
import numpy as np
import pickle # For saving and loading models
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # Needed for dummy model training

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="�",
    layout="wide", # Using wide layout
    initial_sidebar_state="auto"
)

# Function to load and train the model (or load an existing one)
@st.cache_resource # Cache resource to avoid retraining every time the app runs
def load_and_train_model():
    # --- IMPORTANT: REPLACE THIS SECTION WITH YOUR ACTUAL MODEL TRAINING CODE ---
    # If you have a saved model and encoders (e.g., .pkl files),
    # you can load them here. Example:
    # try:
    #     with open('model.pkl', 'rb') as f:
    #         model = pickle.load(f)
    #     with open('le_gender.pkl', 'rb') as f:
    #         le_gender = pickle.load(f)
    #     with open('le_status_mhs.pkl', 'rb') as f:
    #         le_status_mhs = pickle.load(f)
    #     with open('le_status_nikah.pkl', 'rb') as f:
    #         le_status_nikah = pickle.load(f)
    #     # Define the exact feature order your model expects
    #     # IPK will be calculated, so ensure your model was trained with IPK as a feature
    #     model_features = ['Usia', 'IPK', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8',
    #                       'Gender_encoded', 'Status_Mahasiswa_encoded', 'Status_Nikah_encoded']
    #     st.success("Model dan encoder berhasil dimuat dari file.")
    #     return model, le_gender, le_status_mhs, le_status_nikah, model_features
    # except FileNotFoundError:
    #     st.warning("File model atau encoder tidak ditemukan. Melatih ulang model dummy...")

    # Extended dummy data for training example
    # In a real application, you would use data from your 'Kelulusan Train.csv' file.
    # Ensure this dummy data covers all categories that can appear in the radio buttons
    data = {
        'Gender': ['Pria', 'Wanita'] * 10, # 20 entries to ensure both categories are present
        'Status_Mahasiswa': ['Bekerja', 'Tidak Bekerja'] * 10, # 20 entries
        'Status_Nikah': ['Belum Menikah', 'Menikah'] * 10, # 20 entries
        'Usia': [20, 21, 22, 20, 23, 21, 24, 20, 25, 22, 19, 20, 21, 22, 23, 20, 24, 21, 25, 22],
        'IPS1': [3.5, 3.8, 2.5, 3.9, 3.0, 3.7, 3.1, 3.8, 2.0, 3.6, 3.7, 3.5, 3.8, 2.6, 3.9, 3.0, 3.2, 3.7, 2.1, 3.6],
        'IPS2': [3.6, 3.7, 2.6, 3.8, 3.1, 3.6, 3.2, 3.7, 2.1, 3.5, 3.8, 3.6, 3.7, 2.7, 3.8, 3.1, 3.3, 3.6, 2.2, 3.5],
        'IPS3': [3.4, 3.9, 2.4, 4.0, 2.9, 3.8, 3.0, 3.9, 1.9, 3.7, 3.6, 3.4, 3.9, 2.5, 4.0, 2.9, 3.1, 3.8, 2.0, 3.7],
        'IPS4': [3.5, 3.8, 2.5, 3.9, 3.0, 3.7, 3.1, 3.8, 2.0, 3.6, 3.7, 3.5, 3.8, 2.6, 3.9, 3.0, 3.2, 3.7, 2.1, 3.6],
        'IPS5': [3.6, 3.7, 2.6, 3.8, 3.1, 3.6, 3.2, 3.7, 2.1, 3.5, 3.8, 3.6, 3.7, 2.7, 3.8, 3.1, 3.3, 3.6, 2.2, 3.5],
        'IPS6': [3.4, 3.9, 2.4, 4.0, 2.9, 3.8, 3.0, 3.9, 1.9, 3.7, 3.6, 3.4, 3.9, 2.5, 4.0, 2.9, 3.1, 3.8, 2.0, 3.7],
        'IPS7': [3.5, 3.8, 2.5, 3.9, 3.0, 3.7, 3.1, 3.8, 2.0, 3.6, 3.7, 3.5, 3.8, 2.6, 3.9, 3.0, 3.2, 3.7, 2.1, 3.6],
        'IPS8': [3.6, 3.7, 2.6, 3.8, 3.1, 3.6, 3.2, 3.7, 2.1, 3.5, 3.8, 3.6, 3.7, 2.7, 3.8, 3.1, 3.3, 3.6, 2.2, 3.5],
        'Kelulusan': [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1] # 1 for Lulus, 0 for Tidak Lulus
    }
    df_dummy = pd.DataFrame(data)

    # Calculate IPK as average of IPS1-IPS8 for dummy data
    ips_cols = [f'IPS{i}' for i in range(1, 9)]
    df_dummy['IPK'] = df_dummy[ips_cols].mean(axis=1)

    # Initialize LabelEncoders
    le_gender = LabelEncoder()
    le_status_mahasiswa = LabelEncoder()
    le_status_nikah = LabelEncoder()

    # Explicitly fit LabelEncoders with ALL possible categories to prevent unseen label errors
    # This is crucial for robustness
    le_gender.fit(['Pria', 'Wanita'])
    le_status_mahasiswa.fit(['Bekerja', 'Tidak Bekerja'])
    le_status_nikah.fit(['Belum Menikah', 'Menikah'])

    # Transform categorical columns in dummy data
    df_dummy['Gender_encoded'] = le_gender.transform(df_dummy['Gender'])
    df_dummy['Status_Mahasiswa_encoded'] = le_status_mahasiswa.transform(df_dummy['Status_Mahasiswa'])
    df_dummy['Status_Nikah_encoded'] = le_status_nikah.transform(df_dummy['Status_Nikah'])

    # Define features (X) and target (y)
    # Ensure column order matches what your model expects during training
    model_features = [
        'Usia', 'IPK', # IPK is now calculated
        'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'IPS7', 'IPS8',
        'Gender_encoded', 'Status_Mahasiswa_encoded', 'Status_Nikah_encoded'
    ]
    X_dummy = df_dummy[model_features]
    y_dummy = df_dummy['Kelulusan']

    # Train RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_dummy, y_dummy)

    # Return model, encoders, and feature list for consistency
    return model, le_gender, le_status_mahasiswa, le_status_nikah, model_features

# Load model and encoders (will be trained once or loaded if files exist)
model, le_gender, le_status_mahasiswa, le_status_nikah, model_features = load_and_train_model()

# --- Streamlit Application Title ---
st.title('Aplikasi Prediksi Kelulusan Mahasiswa')
st.write('Isi form di bawah ini untuk memprediksi probabilitas kelulusan mahasiswa berdasarkan berbagai kriteria.')

# --- Input Form Section ---
st.header('Data Diri Mahasiswa')

col1, col2 = st.columns(2)
with col1:
    # Changed to st.radio for Jenis Kelamin
    gender = st.radio('Jenis Kelamin', ['Pria', 'Wanita'], index=0) # index=0 sets 'Pria' as default
    # Changed to st.radio for Status Mahasiswa
    status_mahasiswa = st.radio('Status Mahasiswa', ['Bekerja', 'Tidak Bekerja'], index=1) # index=1 sets 'Tidak Bekerja' as default
with col2:
    usia = st.number_input('Usia', min_value=17, max_value=70, value=20, help="Masukkan usia mahasiswa.")
    # Changed to st.radio for Status Pernikahan
    status_nikah = st.radio('Status Pernikahan', ['Belum Menikah', 'Menikah'], index=0) # index=0 sets 'Belum Menikah' as default

st.header('Nilai Akademik')
ips_values = {}
num_ips_cols = 4 # Number of columns for IPS inputs
cols_ips = st.columns(num_ips_cols)

for i in range(1, 9):
    with cols_ips[(i - 1) % num_ips_cols]:
        # st.number_input inherently includes increment/decrement buttons.
        # No direct option to remove them in Streamlit API.
        ips_values[f'IPS{i}'] = st.number_input(f'IPS Semester {i}', min_value=0.0, max_value=4.0, value=3.0, step=0.01, key=f'ips_{i}', help=f"Masukkan Indeks Prestasi Semester {i} (0.0 - 4.0).")

# Calculate IPK from the input IPS values
if ips_values:
    calculated_ipk = np.mean(list(ips_values.values()))
else:
    calculated_ipk = 0.0 # Default value if no IPS inputs (though this shouldn't happen with fixed loops)

# Display the calculated IPK before the prediction button
st.info(f"IPK yang Dihitung Otomatis: **{calculated_ipk:.2f}**")

st.header('Prediksi Kelulusan')
# --- Prediction Button ---
if st.button('Prediksi Kelulusan'):
    try:
        # Preprocessing input for categorical features using trained encoders
        gender_encoded = le_gender.transform([gender])[0]
        status_mahasiswa_encoded = le_status_mahasiswa.transform([status_mahasiswa])[0]
        status_nikah_encoded = le_status_nikah.transform([status_nikah])[0]

        # Prepare input data in dictionary format
        input_dict = {
            'Usia': usia,
            'IPK': calculated_ipk, # Using the calculated IPK
            'Gender_encoded': gender_encoded,
            'Status_Mahasiswa_encoded': status_mahasiswa_encoded,
            'Status_Nikah_encoded': status_nikah_encoded,
        }
        for i in range(1, 9): # Add all IPS values to the dictionary
            input_dict[f'IPS{i}'] = ips_values[f'IPS{i}']

        # Create DataFrame from user input, ensure column order matches model_features
        input_data_df = pd.DataFrame([input_dict])[model_features]

        # Perform Prediction
        prediction = model.predict(input_data_df)
        prediction_proba = model.predict_proba(input_data_df) # Probabilities for both classes (0 and 1)

        st.subheader('Hasil Prediksi & Skala Probabilitas')

        pass_proba = prediction_proba[0][1] * 100 # Probability of passing (class 1)
        fail_proba = prediction_proba[0][0] * 100 # Probability of failing (class 0)

        # Corrected: Separate the st.success/st.error calls to avoid DeltaGenerator print
        if prediction[0] == 1:
            st.success(f'Mahasiswa diprediksi **LULUS TEPAT WAKTU**! 🎉 Probabilitas: **{pass_proba:.2f}%**')
        else:
            st.error(f'Mahasiswa diprediksi **TERLAMBAT LULUS**! 😔 Probabilitas: **{fail_proba:.2f}%**')

        # Simple visual scale using st.progress
        st.write("Skala Probabilitas Lulus:")
        st.progress(pass_proba / 100) # st.progress expects a value between 0.0 and 1.0


    except Exception as e:
        # Catch any unexpected errors and display them
        st.error(f"Terjadi kesalahan: {e}. Mohon coba lagi. Jika masalah berlanjut, pastikan semua input valid dan model Anda dilatih dengan benar. Debugging Info: {e}")

st.markdown('---')
st.caption('Aplikasi ini dibuat menggunakan Streamlit dan scikit-learn. Model prediksi menggunakan data dummy; untuk hasil yang akurat, gantilah dengan model yang telah dilatih pada dataset asli Anda.')
