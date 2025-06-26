# 📦 Step 1: Upload ZIP
from google.colab import files
uploaded = files.upload()  # Upload archive.zip

# 📂 Step 2: Extract CSV
import zipfile
import os

with zipfile.ZipFile("archive.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

# Find CSV file name
csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]
print("Found CSV:", csv_files)

# 🧠 Step 3: Load Data
import pandas as pd

df = pd.read_csv(f"data/{csv_files[0]}")
print("Dataset loaded. First 5 rows:")
print(df.head())

# 🔍 Step 4: Explore
print("Info:")
print(df.info())
print("Fraud count:\n", df['Class'].value_counts())

# 🔧 Step 5: Preprocess
from sklearn.preprocessing import StandardScaler

df['normAmount'] = StandardScaler().fit_transform(df[['Amount']])
df = df.drop(['Time', 'Amount'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# ✂️ Step 6: Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 🤖 Step 7: Train Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📊 Step 8: Evaluate
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# 🌳 Step 9: Random Forest (Optional)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Results:")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# 💾 Step 10: Save Model (Optional)
import joblib

joblib.dump(rf, "fraud_model.pkl")
files.download("fraud_model.pkl")
