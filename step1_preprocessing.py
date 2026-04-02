import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset directly from URL (no download needed)
url = "https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv"
df = pd.read_csv(url)

print("✅ Dataset Loaded")
print("Shape:", df.shape)

# Step 2: Basic Cleaning
# Drop unwanted column
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
# Drop rows where target is missing
df = df.dropna(subset=['Price'])

# Step 3: Handle Missing Values
# Numerical → Median
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical → Mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Missing Values Handled")

# Step 4: Feature Scaling (VERY IMPORTANT for ML)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("✅ Data Scaled Successfully")

# Step 5: Save Clean Data
df.to_csv("cleaned_cars.csv", index=False)

print("🎯 Step 1 Completed: cleaned_cars.csv created")