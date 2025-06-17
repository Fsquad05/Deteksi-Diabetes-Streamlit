# app.py
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

# File CSV untuk menyimpan riwayat
RIWAYAT_FILE = "riwayat_prediksi.csv"

# Inisialisasi file jika belum ada
if not os.path.exists(RIWAYAT_FILE):
    df_init = pd.DataFrame(columns=[
        "Tanggal Pemeriksaan","ID Pasien", "Nama Pasien", "Jenis Kelamin", "Usia", "BMI", "HbA1c",
        "Urea", "Cr", "Kolesterol", "TG", "HDL", "LDL", "VLDL",
        "Status", "Nama Pemeriksa"
    ])
    df_init.to_csv(RIWAYAT_FILE, index=False)

# Load model
with open('model_voting.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.title('🩺 Prediksi Diabetes Berdasarkan Data Medis')

# Dua bagian utama
col_kiri, col_kanan = st.columns(2)

# ========== KIRI: INPUT ==========
with col_kiri:
    st.subheader("🧾 Formulir Pemeriksaan")

    tanggal_pemeriksaan = st.date_input("📅 Tanggal Pemeriksaan")

    # Identitas pasien & pemeriksa
    row1 = st.columns(3)
    id_pasien = row1[0].text_input("ID Pasien")
    nama_pasien = row1[1].text_input("Nama Pasien")
    gender = row1[2].selectbox("Jenis Kelamin", ("-- Pilih --", "Perempuan", "Laki-laki"))

    row2 = st.columns(3)
    age = row2[0].number_input("Usia", min_value=0, format="%d")
    bmi = row2[1].number_input("BMI", min_value=0.0)
    hba1c = row2[2].number_input("HbA1c", min_value=0.0)

    row3 = st.columns(3)
    urea = row3[0].number_input("Urea", min_value=0.0)
    cr = row3[1].number_input("Kreatinin (Cr)", min_value=0.0)
    chol = row3[2].number_input("Kolesterol Total", min_value=0.0)

    row4 = st.columns(3)
    tg = row4[0].number_input("Trigliserida (TG)", min_value=0.0)
    hdl = row4[1].number_input("HDL", min_value=0.0)
    ldl = row4[2].number_input("LDL", min_value=0.0)

    row5 = st.columns(3)
    vldl = row5[0].number_input("VLDL", min_value=0.0)
    nama_pemeriksa = row5[1].text_input("Nama Pemeriksa")

    # Tombol di tengah
    tombol_row = st.columns([1, 1, 1])
    predict_btn = tombol_row[1].button("🔍 Prediksi")

# ========== KANAN: HASIL ==========
with col_kanan:
    st.subheader("📋 Hasil Pemeriksaan")

    if predict_btn:
        if not all([id_pasien, nama_pasien, nama_pemeriksa]):
            st.error("Mohon lengkapi semua isian formulir termasuk identitas pasien dan pemeriksa.")
        else:
            gender_encoded = 0 if gender == "Perempuan" else 1
            input_data = np.array([[gender_encoded, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]])

            # Prediksi
            prediction = model.predict(input_data)[0]
            pred_label = {0: "Normal", 1: "Pre-Diabetic", 2: "Diabetic"}
            result = pred_label[prediction]

            st.success(f"Hasil Prediksi: {result}")

            # Simpan ke file CSV
            data_baru = {
                "Tanggal Pemeriksaan": tanggal_pemeriksaan,
                "ID Pasien": id_pasien,
                "Nama Pasien": nama_pasien,
                "Jenis Kelamin": gender,
                "Usia": age,
                "BMI": bmi,
                "HbA1c": hba1c,
                "Urea": urea,
                "Cr": cr,
                "Kolesterol": chol,
                "TG": tg,
                "HDL": hdl,
                "LDL": ldl,
                "VLDL": vldl,
                "Status": result,
                "Nama Pemeriksa": nama_pemeriksa,
            }

            df_existing = pd.read_csv(RIWAYAT_FILE)
            df_existing = pd.concat([df_existing, pd.DataFrame([data_baru])], ignore_index=True)
            df_existing.to_csv(RIWAYAT_FILE, index=False)

        st.markdown("---")
    else:
        st.info("Isi data di sebelah kiri lalu klik tombol **Prediksi** untuk melihat hasil.")
