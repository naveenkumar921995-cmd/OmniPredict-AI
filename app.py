import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score, accuracy_score, confusion_matrix

# Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from xgboost import XGBClassifier

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="OmniPredict AI", layout="wide")

st.title("🚀 ModelVerse AI")
st.markdown("### Explore, Compare & Understand Machine Learning Models")
st.info("🔍 Covers Regression | Classification | Ensemble | Clustering in one platform")
# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_cars.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Data Overview",
    "📈 Regression Models",
    "📉 Classification Models",
    "🔵 Clustering (KMeans + PCA)",
    "🏆 Model Leaderboard",
    "📚 ML Concept Breakdown"
])

# =========================================================
# 📊 DATA OVERVIEW
# =========================================================
if page == "📊 Data Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =========================================================
# 📈 REGRESSION
# =========================================================
elif page == "📈 Regression Models":
    st.subheader("Regression Models (Price Prediction)")

    df_reg = pd.get_dummies(df, drop_first=True)

    X = df_reg.drop("Price", axis=1)
    y = df_reg["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model_name = st.selectbox("Select Model", [
        "Linear Regression",
        "Ridge (L2)",
        "Lasso (L1)",
        "ElasticNet"
    ])

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge (L2)":
        model = Ridge()
    elif model_name == "Lasso (L1)":
        model = Lasso()
    else:
        model = ElasticNet()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.success(f"R2 Score: {r2_score(y_test, preds):.4f}")

# =========================================================
# 📉 CLASSIFICATION
# =========================================================
elif page == "📉 Classification Models":
    st.subheader("Classification Models (High vs Low Price)")

    df_clf = df.copy()
    df_clf["Target"] = (df_clf["Price"] > df_clf["Price"].median()).astype(int)
    df_clf = df_clf.drop("Price", axis=1)
    df_clf = pd.get_dummies(df_clf, drop_first=True)

    X = df_clf.drop("Target", axis=1)
    y = df_clf["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model_name = st.selectbox("Select Model", [
        "Logistic Regression",
        "Naive Bayes",
        "Decision Tree",
        "KNN",
        "SVM",
        "Random Forest",
        "XGBoost"
    ])

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "SVM":
        model = SVC()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    else:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    st.success(f"Accuracy: {acc*100:.2f}%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Feature Importance (Tree Models Only)
    if model_name in ["Decision Tree", "Random Forest", "XGBoost"]:
        st.subheader("Feature Importance")
        importance = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values("Importance", ascending=False).head(10)

        fig, ax = plt.subplots()
        sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)

# =========================================================
# 🔵 CLUSTERING
# =========================================================
elif page == "🔵 Clustering (KMeans + PCA)":
    st.subheader("Clustering using KMeans")

    df_c = pd.get_dummies(df, drop_first=True)

    k = st.slider("Select Number of Clusters", 2, 10, 3)

    kmeans = KMeans(n_clusters=k)
    clusters = kmeans.fit_predict(df_c)

    pca = PCA(n_components=2)
    comp = pca.fit_transform(df_c)

    pca_df = pd.DataFrame(comp, columns=["PC1", "PC2"])
    pca_df["Cluster"] = clusters

    st.scatter_chart(pca_df, x="PC1", y="PC2", color="Cluster")

# =========================================================
# 🏆 LEADERBOARD
# =========================================================
elif page == "🏆 Model Leaderboard":
    st.subheader("Model Performance Comparison")

    df_clf = df.copy()
    df_clf["Target"] = (df_clf["Price"] > df_clf["Price"].median()).astype(int)
    df_clf = df_clf.drop("Price", axis=1)
    df_clf = pd.get_dummies(df_clf, drop_first=True)

    X = df_clf.drop("Target", axis=1)
    y = df_clf["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results.append([name, acc])

    result_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    st.dataframe(result_df.sort_values("Accuracy", ascending=False))

# =========================================================
# 📚 ML BREAKDOWN (YOUR SYLLABUS)
# =========================================================
elif page == "📚 ML Concept Breakdown":
    st.subheader("Machine Learning Complete Breakdown")

    st.markdown("""
### 🔵 REGRESSION
- Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Ridge (L2)
- Lasso (L1)
- ElasticNet

### 🔴 CLASSIFICATION
- Logistic Regression (Logit)
- Naive Bayes
- Decision Tree
- KNN
- SVM

### 🟢 ENSEMBLE LEARNING
- Random Forest (Bagging)
- XGBoost (Boosting)

### 🟣 CLUSTERING
- KMeans
- PCA (Dimensionality Reduction)

👉 This project covers complete ML lifecycle.
""")