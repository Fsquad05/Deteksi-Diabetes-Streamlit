import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle

# === 1. Load dataset ===
df = pd.read_csv('Dataset Of Diabetes.csv')  # pastikan nama file benar dan diunggah ke Colab

# === 2. Pilih dan bersihkan kolom ===
df = df[['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI', 'CLASS']]
df['CLASS'] = df['CLASS'].astype(str).str.strip().str.upper()
df = df[df['CLASS'].isin(['N', 'P', 'Y'])]

# Mapping Gender dan Class
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
df['CLASS'] = df['CLASS'].map({'N': 0, 'P': 1, 'Y': 2})

# === 3. Pisahkan fitur dan label ===
X = df.drop(columns=['CLASS'])
y = df['CLASS']

# === 4. Tangani missing value ===
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# === 5. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Inisialisasi model ===
model_svm = SVC(probability=True, random_state=42)
model_knn = KNeighborsClassifier(n_neighbors=5)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# === 7. Voting classifier ===
voting_model = VotingClassifier(estimators=[
    ('svm', model_svm),
    ('knn', model_knn),
    ('rf', model_rf)
], voting='hard')

# === 8. Latih model ensemble ===
voting_model.fit(X_train, y_train)

# === 9. Simpan model ke file .pkl ===
with open('model_voting.pkl', 'wb') as f:
    pickle.dump(voting_model, f)

print("✅ Model Voting berhasil disimpan sebagai 'model_voting.pkl'")