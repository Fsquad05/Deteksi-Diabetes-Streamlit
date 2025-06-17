import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

RIWAYAT_FILE = "riwayat_prediksi.csv"

st.set_page_config(page_title="Riwayat Prediksi", layout="wide")
st.title("🧾 Riwayat Prediksi Diabetes")

# Cek file
try:
    df = pd.read_csv(RIWAYAT_FILE)
except FileNotFoundError:
    st.warning("Belum ada data riwayat prediksi.")
    st.stop()

if df.empty:
    st.info("Belum ada riwayat prediksi.")
    st.stop()

# Pastikan kolom teks dalam bentuk string
df["ID Pasien"] = df["ID Pasien"].astype(str)
df["Nama Pasien"] = df["Nama Pasien"].astype(str)

# Baris pencarian & filter status (BERSAMPINGAN)
col1, col2 = st.columns([3, 3])
with col1:
    keyword = st.text_input("🔍 Cari ID/Nama Pasien").lower()
with col2:
    status_filter = st.multiselect("🩺 Filter Status", ["Normal", "Pre-Diabetic", "Diabetic"], default=["Normal", "Pre-Diabetic", "Diabetic"])

# Terapkan filter keyword & status
if keyword:
    df = df[df["ID Pasien"].str.lower().str.contains(keyword) |
            df["Nama Pasien"].str.lower().str.contains(keyword)]

df_filtered = df[df["Status"].isin(status_filter)]

# Header dan tombol download + hapus riwayat sejajar
header_col, download_col, hapus_col = st.columns([5, 1, 1])
with header_col:
    st.subheader("📜 Hasil Riwayat Prediksi")

with download_col:
    st.download_button(
        label="⬇️ Download CSV",
        data=df_filtered.to_csv(index=False).encode("utf-8"),
        file_name="riwayat_prediksi.csv",
        mime="text/csv"
    )

with hapus_col:
    if st.button("🗑 Hapus Riwayat"):
        os.remove(RIWAYAT_FILE)
        st.warning("Semua riwayat berhasil dihapus!")
        st.rerun()  # Gunakan ini sebagai pengganti experimental_rerun

# Tampilkan hasil
st.dataframe(df_filtered, use_container_width=True)

# Cek file dan tampilkan histogram
st.divider()
st.subheader("📊 Statistik Status Pasien")

# Cek apakah file tersedia
if os.path.exists(RIWAYAT_FILE):
    df = pd.read_csv(RIWAYAT_FILE)

    if not df.empty and "Status" in df.columns:
        status_count = df["Status"].value_counts()

        # Buat histogram di matplotlib
        fig, ax = plt.subplots()
        colors = ["#3cb371", "#ffa500", "#dc143c"]  # hijau, oranye, merah
        bars = ax.bar(status_count.index, status_count.values, color=colors)
        ax.set_title("Distribusi Pasien Berdasarkan Status Prediksi")
        ax.set_xlabel("Status")
        ax.set_ylabel("Jumlah Pasien")

        # Tampilkan histogram dan tabel bersampingan
        col1, col2 = st.columns([2, 1])  # proporsi 2:1
        with col1:
            st.pyplot(fig)
        with col2:
            st.markdown("#### 📌 Data Ringkasan")
            st.table(status_count.rename("Jumlah"))
    else:
        st.info("Belum ada data yang bisa ditampilkan.")
else:
    st.info("Data belum tersedia. Silakan lakukan prediksi terlebih dahulu.")