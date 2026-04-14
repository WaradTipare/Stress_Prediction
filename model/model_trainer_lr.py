#Using logistic regression

import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\warad\OneDrive\Desktop\projects\stress_prediction\data\data1.csv")

# Features & Target
X = df[['sleep', 'study', 'screen', 'workload', 'mood']]
y = df['stress']

# ML tools
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 🔹 Model
model = LogisticRegression(max_iter=300, class_weight='balanced')

# 🔹 Train
model.fit(X_train, y_train)

# 🔹 Predict
y_pred = model.predict(X_test)

# 🔹 Evaluation
print("===== Logistic Regression Results =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 🔹 Cross Validation (improved version)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=kf)

print("\nCross Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean())

import joblib

# Save model
joblib.dump(model, "model.pkl")

# Save scaler
joblib.dump(scaler, "scaler.pkl")
