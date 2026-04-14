import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

n = 500  # number of samples

# Generate features
sleep = np.random.uniform(3, 9, n)              # hours
study = np.random.uniform(1, 10, n)             # hours
screen = np.random.uniform(1, 14, n)            # hours
workload = np.random.randint(1, 15, n)          # tasks
mood = np.random.uniform(1, 10, n)              # score

# Add noise
noise = np.random.normal(0, 1, n)

# Create stress logic (important)
stress_score = (
    (10 - sleep) * 0.5 +
    study * 0.2 +
    screen * 0.3 +
    workload * 0.3 +
    (10 - mood) * 0.5 +
    noise
)

# Convert to categories
stress_level = []
for s in stress_score:
    if s < 7:
        stress_level.append(0)  # Low
    elif s < 10:
        stress_level.append(1)  # Medium
    else:
        stress_level.append(2)  # High

# Create DataFrame
df = pd.DataFrame({
    'sleep': sleep,
    'study': study,
    'screen': screen,
    'workload': workload,
    'mood': mood,
    'stress': stress_level
})

# Save dataset to your folder
df.to_csv(r"C:\Users\warad\OneDrive\Desktop\projects\stress_prediction\data\data1.csv", index=False)

print("Dataset created successfully!")
print(df.head())

# Check class balance
print("\nClass Distribution:")
print(df['stress'].value_counts())


