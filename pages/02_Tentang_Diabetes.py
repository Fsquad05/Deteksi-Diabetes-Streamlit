import streamlit as st

st.set_page_config(page_title="Tentang Diabetes", layout="wide")
st.title("📚 Informasi Tentang Diabetes dan Atribut Medis")

st.header("Apa Itu Diabetes?")
st.markdown("""
**Diabetes Mellitus** adalah penyakit kronis yang terjadi ketika tubuh tidak dapat memproduksi cukup insulin atau tidak dapat menggunakannya secara efektif. Insulin adalah hormon yang mengatur kadar gula (glukosa) dalam darah.

Jenis diabetes yang paling umum:
- **Diabetes Tipe 1**: Tubuh tidak memproduksi insulin.
- **Diabetes Tipe 2**: Tubuh tidak efektif menggunakan insulin.
- **Pre-Diabetes**: Kadar gula darah lebih tinggi dari normal, tetapi belum cukup tinggi untuk diklasifikasikan sebagai diabetes.
""")

st.header("📊 Penjelasan Atribut Medis")
st.markdown("""
| Atribut          | Penjelasan                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Jenis Kelamin** | Perempuan atau Laki-laki. Perbedaan biologis bisa mempengaruhi risiko.     |
| **Usia**          | Semakin tua, risiko diabetes tipe 2 meningkat.                             |
| **BMI**           | Body Mass Index: rasio berat badan terhadap tinggi badan. Obesitas = risiko tinggi. |
| **HbA1c**         | Kadar gula darah rata-rata dalam 2-3 bulan terakhir. Di atas 6.5% = Diabetes.|
| **Urea**          | Indikator fungsi ginjal. Kadar abnormal bisa mengindikasikan komplikasi.   |
| **Kreatinin (Cr)**| Digunakan untuk mengevaluasi fungsi ginjal.                                |
| **Kolesterol**    | Total kolesterol dalam darah. Tinggi = risiko komplikasi jantung.          |
| **TG (Trigliserida)**| Lemak dalam darah. Tinggi = risiko sindrom metabolik dan diabetes.       |
| **HDL**           | "Kolesterol baik". Rendah = risiko penyakit jantung dan diabetes.          |
| **LDL**           | "Kolesterol jahat". Tinggi = peningkatan risiko penyumbatan pembuluh darah.|
| **VLDL**          | Jenis lipoprotein pembawa trigliserida.                                   |
""")

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

st.header("📈 Akurasi Model Prediksi Diabetes")

# Load data
df = pd.read_csv("Dataset Of Diabetes.csv")
df = df[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'CLASS']]
df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()
df = df[df['CLASS'].isin(['N', 'P', 'Y'])]
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})

X = df.drop(columns=['CLASS'])
y = df['CLASS']
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model
svm = SVC(kernel='linear', C=1.0, gamma='scale')
knn = KNeighborsClassifier(n_neighbors=3)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Latih model
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Voting model
voting = VotingClassifier(estimators=[
    ('svm', svm),
    ('knn', knn),
    ('rf', rf)
], voting='hard')
voting.fit(X_train, y_train)

# Hitung akurasi
svm_acc = accuracy_score(y_test, svm.predict(X_test))
knn_acc = accuracy_score(y_test, knn.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))
voting_acc = accuracy_score(y_test, voting.predict(X_test))

# Tampilkan ke UI
col1, col2 = st.columns(2)
with col1:
    st.metric("🎯 Akurasi SVM", f"{svm_acc:.2%}")
    st.metric("🎯 Akurasi KNN", f"{knn_acc:.2%}")
with col2:
    st.metric("🎯 Akurasi Random Forest", f"{rf_acc:.2%}")
    st.metric("🗳️ Akurasi Voting Classifier", f"{voting_acc:.2%}")
