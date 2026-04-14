import pandas as pd

# Load dataset
df = pd.read_csv(r"C:\Users\warad\OneDrive\Desktop\projects\stress_prediction\data\data1.csv")

# Features & Target
X = df[['sleep', 'study', 'screen', 'workload', 'mood']]
y = df['stress']

# ML tools
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Grid Search Parameters
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# 🔹 Grid Search
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    params,
    cv=5
)

grid.fit(X_train, y_train)

print("===== Random Forest Results =====")
print("Best Parameters:", grid.best_params_)

# 🔹 Best Model
model = grid.best_estimator_

# 🔹 Train
model.fit(X_train, y_train)

# 🔹 Predict
y_pred = model.predict(X_test)

# 🔹 Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 🔹 Cross Validation (consistent version)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)

print("\nCross Validation Scores:", scores)
print("Average CV Accuracy:", scores.mean())