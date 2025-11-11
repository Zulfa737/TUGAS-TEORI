import streamlit as st
import pickle
import numpy as np

# Judul Aplikasi
st.title("🌸 Iris Flower Classification App")
st.write("Prediksi kategori bunga Iris berdasarkan empat fitur utama.")

# Load Model
with open("TUGAS_TEORI_7_ZULFA.pkcls", "rb") as file:
    model = pickle.load(file)

# Input fitur dari user
st.header("Masukkan Nilai Fitur:")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Tombol Prediksi
if st.button("🔍 Prediksi"):
    # Buat array input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Lakukan prediksi
    prediction = model.predict(input_data)

    # Map hasil prediksi ke nama spesies
    iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    result = iris_classes[int(prediction[0])]

    # Tampilkan hasil
    st.success(f"Hasil prediksi: **{result}**")
