import streamlit as st
import pickle
import numpy as np
import Orange # Diperlukan untuk memuat model .pkcls dari Orange

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Iris (Orange Model)",
    page_icon="🌸"
)

# --- Fungsi untuk Memuat Model ---
# Menggunakan cache agar model tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model(model_path):
    """Memuat model .pkcls dari file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: File model '{model_path}' tidak ditemukan.")
        st.info("Pastikan file model Anda berada di direktori yang sama dengan `app.py` dan namanya sudah benar.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        st.info("Pastikan Anda memiliki library 'orange3' terinstal di environment Anda.")
        return None

# --- Mapping Nama Kelas ---
# Berdasarkan file .ows Anda, kelas target 'iris' memiliki 3 nilai:
# 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
# class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# --- Load Model ---
# Ganti 'nama_model_anda.pkcls' dengan nama file .pkcls Anda yang sebenarnya
MODEL_FILE = 'model_iris.pkcls' 
model = load_model(MODEL_FILE)

# --- Antarmuka Streamlit ---
st.title("🌸 Prediksi Kategori Bunga Iris")
st.write(f"""
Aplikasi ini menggunakan model **Neural Network** yang telah Anda train di Orange 
(dari file `.ows`) untuk memprediksi kategori bunga Iris berdasarkan fiturnya.
""")
st.write("---")

# --- Input Sidebar ---
st.sidebar.header("Masukkan Fitur Iris:")

# Buat slider untuk setiap fitur
sl = st.sidebar.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.4, step=0.1)
sw = st.sidebar.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.4, step=0.1)
pl = st.sidebar.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.3, step=0.1)
pw = st.sidebar.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2, step=0.1)

# Tombol Prediksi
if st.sidebar.button("Prediksi Kategori"):
    if model is not None:
        # --- Proses Prediksi ---
        
        # 1. Siapkan data input (harus dalam format numpy array 2D)
        input_data = np.array([[sl, sw, pl, pw]])
        
        # 2. Lakukan prediksi
        # Model Orange akan mengembalikan indeks kelas (0, 1, atau 2)
        prediction_index = model.predict(input_data)
        
        # 3. Dapatkan nama kelas dari indeks
        predicted_class_name = class_names[int(prediction_index[0])]
        
        # --- Tampilkan Hasil ---
        st.subheader("Hasil Prediksi:")
        
        # Tampilkan hasil dalam 'metric' box
        st.metric(label="Kategori Iris", value=predicted_class_name)
        
        # Tampilkan gambar berdasarkan hasil (opsional, tapi bagus)
        if predicted_class_name == 'Iris-setosa':
            st.image("https.upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa.jpg", caption="Iris Setosa", width=300)
        elif predicted_class_name == 'Iris-versicolor':
            st.image("https.upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Versicolor", width=300)
        else:
            st.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", caption="Iris Virginica", width=300)

    else:
        st.error("Model tidak dapat digunakan karena gagal dimuat.")
else:

    st.info("Silakan atur nilai fitur di sidebar dan klik 'Prediksi Kategori'.")
