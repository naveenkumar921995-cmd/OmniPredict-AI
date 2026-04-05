import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv("cleaned_cars.csv")

print("✅ Data Loaded")

# --- Classification Setup ---
df['Price_Class'] = (df['Price'] > df['Price'].median()).astype(int)
df = df.drop(columns=['Price'])

# Encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Price_Class'])
y = df['Price_Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data Prepared")

# --- ENSEMBLE MODELS ---
models = {
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

print("\n🔥 Ensemble Model Results:")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {acc:.4f}")

# --- K-MEANS CLUSTERING ---
print("\n📊 K-Means Clustering")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

print("Cluster Labels (first 10):")
print(clusters[:10])

# --- PCA ---
print("\n📉 PCA Dimensionality Reduction")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("PCA Shape:", X_pca.shape)