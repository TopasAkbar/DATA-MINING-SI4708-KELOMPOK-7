
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as stc
from sklearn.metrics import mean_absolute_error, r2_score

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Prediksi Jumlah Pasien HIV",
    page_icon="🧬",
    layout="wide",
)

# Load model
model = joblib.load("model.pkl")

# Title halaman
st.title("🧬 Dashboard Prediksi Jumlah Pasien HIV")

# Load data
columns = [
    "kode_kabupaten", "kode_kecamatan", "nama_kabupaten", "nama_kecamatan",
    "tahun", "jenis_kelamin", "keterangan", "jumlah_pasien_hiv", "satuan"
]
data = pd.read_csv("1703423829.csv", skiprows=2, names=columns)
data["jumlah_pasien_hiv"] = pd.to_numeric(data["jumlah_pasien_hiv"], errors="coerce")
data = data.dropna(subset=["jumlah_pasien_hiv"])

# Hitung agregasi
total_data = data.groupby(['nama_kecamatan', 'jenis_kelamin'])['jumlah_pasien_hiv'].sum().reset_index()
total_kecamatan = data.groupby('nama_kecamatan')['jumlah_pasien_hiv'].sum().sort_values(ascending=False)

# Pivot data untuk evaluasi model
X_agg = total_data.pivot(index='nama_kecamatan', columns='jenis_kelamin', values='jumlah_pasien_hiv').fillna(0)
for gender in ['Laki-Laki', 'Perempuan']:
    if gender not in X_agg.columns:
        X_agg[gender] = 0
X_agg = X_agg[['Laki-Laki', 'Perempuan']]

# Evaluasi model
st.markdown("---")
st.subheader("📈 Evaluasi Model")
if not X_agg.empty:
    y_true = X_agg.sum(axis=1)
    y_pred = model.predict(X_agg)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"MAE (Mean Absolute Error): **{mae:.2f}**")
    with col2:
        st.info(f"R² Score: **{r2:.2f}**")
else:
    st.warning("Data agregasi kosong. Tidak dapat mengevaluasi model.")

# Visualisasi
st.markdown("---")
st.subheader("📊 Visualisasi Data")
visual_choice = st.selectbox(
    "Pilih Visualisasi:",
    ["Bar Chart per Kecamatan", "Pie Chart Gender", "Top 5 Kecamatan", "Data Lengkap per Kecamatan"]
)

if visual_choice == "Bar Chart per Kecamatan":
    st.write("### Jumlah Pasien HIV per Kecamatan")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    total_kecamatan.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_ylabel("Jumlah Pasien HIV")
    ax1.set_title("Total Pasien HIV per Kecamatan")
    st.pyplot(fig1)

elif visual_choice == "Pie Chart Gender":
    st.write("### Distribusi Gender Pasien HIV")
    total_gender = data.groupby('jenis_kelamin')['jumlah_pasien_hiv'].sum()
    fig2, ax2 = plt.subplots()
    ax2.pie(total_gender, labels=total_gender.index, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
    ax2.axis('equal')
    ax2.set_title("Distribusi Pasien HIV berdasarkan Gender")
    st.pyplot(fig2)

elif visual_choice == "Top 5 Kecamatan":
    st.write("### Top 5 Kecamatan dengan Jumlah Pasien HIV Terbanyak")
    top5_kecamatan = total_kecamatan.head(5).reset_index()
    st.table(top5_kecamatan)

elif visual_choice == "Data Lengkap per Kecamatan":
    selected_kecamatan = st.selectbox("Pilih Kecamatan", sorted(data['nama_kecamatan'].unique()))
    filtered_data = data[data['nama_kecamatan'] == selected_kecamatan]
    st.write(f"### Data Lengkap untuk Kecamatan: **{selected_kecamatan}**")
    st.dataframe(filtered_data)

# Pencarian Sederhana
st.markdown("---")
st.subheader("🔍 Pencarian Data Pasien HIV Berdasarkan Gender")

# Filter berdasarkan jenis kelamin saja
gender_filter = st.selectbox("Pilih Jenis Kelamin", sorted(data['jenis_kelamin'].unique()))
filtered_gender = data[data['jenis_kelamin'] == gender_filter]

if not filtered_gender.empty:
    result = filtered_gender.groupby('nama_kecamatan')['jumlah_pasien_hiv'].sum().reset_index()
    st.write(f"### Hasil Pencarian: Jenis Kelamin **{gender_filter}**")
    st.dataframe(result)
else:
    st.warning("Data tidak ditemukan untuk gender ini.")

# Form prediksi interaktif
st.markdown("---")
st.subheader("🔮 Prediksi Jumlah Total Pasien HIV")
with st.form("prediction_form"):
    jumlah_laki = st.number_input("Jumlah Pasien Laki-Laki", min_value=0, step=1)
    jumlah_perempuan = st.number_input("Jumlah Pasien Perempuan", min_value=0, step=1)
    submit = st.form_submit_button("Prediksi Total Pasien HIV")

if submit:
    X_new = np.array([[jumlah_laki, jumlah_perempuan]])
    prediksi = model.predict(X_new)[0]
    st.success(f"Prediksi Jumlah Total Pasien HIV: **{prediksi:.0f}**")

# Simulasi Dampak Pencegahan
st.markdown("---")
st.subheader("🧪 Simulasi Dampak Pencegahan HIV")

# Hitung tren tahunan
trend_data = data.groupby('tahun')['jumlah_pasien_hiv'].sum().reset_index()

# Slider untuk skenario pengurangan
reduction_percent = st.slider(
    "Proyeksi Pengurangan Kasus HIV (%) karena Pencegahan",
    min_value=0, max_value=100, value=10, step=5
)

# Simulasi pengurangan
trend_data['setelah_pencegahan'] = trend_data['jumlah_pasien_hiv'] * (1 - reduction_percent / 100)

# Visualisasi
fig3, ax3 = plt.subplots()
ax3.plot(trend_data['tahun'], trend_data['jumlah_pasien_hiv'], label='Kasus Aktual', marker='o', color='red')
ax3.plot(trend_data['tahun'], trend_data['setelah_pencegahan'], label=f'Skenario Setelah Pencegahan ({reduction_percent}%)', marker='o', color='green')
ax3.set_xlabel("Tahun")
ax3.set_ylabel("Jumlah Pasien HIV")
ax3.set_title("Simulasi Dampak Pencegahan Terhadap Jumlah Kasus HIV")
ax3.legend()
st.pyplot(fig3)
