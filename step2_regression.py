import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

# Load cleaned data
df = pd.read_csv("cleaned_cars.csv")

print("✅ Data Loaded for Regression")

# Select target
target = "Price"

# Convert categorical → numerical (simple encoding)
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data Split Completed")

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "SVR": SVR()
}

# Store results
results = []

# Train & Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    results.append([name, r2, mae])

    print(f"\n{name}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

# Create comparison table
results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "MAE"])

print("\n🏆 Model Comparison Table:")
print(results_df.sort_values(by="R2 Score", ascending=False)) 