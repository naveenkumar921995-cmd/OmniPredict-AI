import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load cleaned data
df = pd.read_csv("cleaned_cars.csv")

print("✅ Data Loaded for Classification")

# Convert Price into Category (High / Low)
df['Price_Class'] = (df['Price'] > df['Price'].median()).astype(int)

# Drop original Price
df = df.drop(columns=['Price'])

# Convert categorical → numerical
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop(columns=['Price_Class'])
y = df['Price_Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("✅ Data Split Completed")

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier()
}

results = []

# Train & Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append([name, acc, f1])

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

# Comparison table
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score"])

print("\n🏆 Classification Comparison Table:")
print(results_df.sort_values(by="Accuracy", ascending=False))