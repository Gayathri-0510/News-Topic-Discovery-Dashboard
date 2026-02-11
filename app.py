# ==========================================
# News Topic Discovery Dashboard
# Upload-based Version
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(
    page_title="News Topic Discovery Dashboard",
    layout="wide"
)

# ------------------------------------------
# Title Section
# ------------------------------------------
st.title("🟣 News Topic Discovery Dashboard")
st.markdown(
    "This system uses **Hierarchical Clustering** to automatically group "
    "similar news articles based on textual similarity."
)

# ------------------------------------------
# Sidebar: Dataset Handling
# ------------------------------------------
st.sidebar.header("📂 Dataset Handling")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ------------------------------------------
# If no file uploaded → STOP
# ------------------------------------------
if uploaded_file is None:
    st.warning("Please upload a CSV file to start.")
    st.stop()

# ------------------------------------------
# Load Dataset
# ------------------------------------------
df = pd.read_csv(uploaded_file, encoding="latin1")

# Auto-detect text column (longest text column)
text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
texts = df[text_col].astype(str)

st.sidebar.success(f"Detected text column:\n{text_col}")

# ------------------------------------------
# Sidebar: Text Vectorization
# ------------------------------------------
st.sidebar.header("📝 Text Vectorization")

max_features = st.sidebar.slider(
    "Maximum TF-IDF Features",
    100, 2000, 1000, step=100
)

use_stopwords = st.sidebar.checkbox(
    "Use English Stopwords",
    value=True
)

ngram_choice = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

if ngram_choice == "Unigrams":
    ngram_range = (1, 1)
elif ngram_choice == "Bigrams":
    ngram_range = (2, 2)
else:
    ngram_range = (1, 2)

# ------------------------------------------
# Sidebar: Hierarchical Clustering
# ------------------------------------------
st.sidebar.header("🌳 Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

if linkage_method == "ward":
    st.sidebar.info("Ward linkage always uses Euclidean distance.")

dendro_size = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, 200, 100
)

# ------------------------------------------
# TF-IDF Vectorization
# ------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if use_stopwords else None,
    ngram_range=ngram_range
)

X = vectorizer.fit_transform(texts)

# ------------------------------------------
# Dendrogram Section
# ------------------------------------------
st.subheader("🌳 Dendrogram")

if st.button("🟦 Generate Dendrogram"):
    X_subset = X[:dendro_size].toarray()

    if linkage_method == "ward":
        Z = linkage(X_subset, method="ward")
    else:
        Z = linkage(X_subset, method=linkage_method, metric="euclidean")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax)
    ax.set_xlabel("Article Index")
    ax.set_ylabel("Distance")

    st.pyplot(fig)

    st.info(
        "Inspect **large vertical gaps** to decide the number of clusters."
    )

# ------------------------------------------
# Apply Clustering
# ------------------------------------------
st.subheader("🟩 Apply Clustering")

n_clusters = st.slider(
    "Select Number of Clusters (based on dendrogram)",
    2, 10, 4
)

if linkage_method == "ward":
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )
else:
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric="euclidean"
    )

labels = model.fit_predict(X.toarray())
df["Cluster"] = labels

# ------------------------------------------
# PCA Visualization
# ------------------------------------------
st.subheader("📊 Cluster Visualization (PCA)")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")

st.pyplot(fig)

# ------------------------------------------
# Cluster Summary
# ------------------------------------------
st.subheader("📋 Cluster Summary")

feature_names = np.array(vectorizer.get_feature_names_out())
rows = []

for c in range(n_clusters):
    idx = np.where(labels == c)[0]
    tfidf_mean = X[idx].mean(axis=0).A1
    top_terms = feature_names[tfidf_mean.argsort()[-10:]][::-1]

    rows.append({
        "Cluster ID": c,
        "Number of Articles": len(idx),
        "Top Keywords": ", ".join(top_terms),
        "Sample Article": texts.iloc[idx[0]][:150] + "..."
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ------------------------------------------
# Validation
# ------------------------------------------
st.subheader("📊 Validation")

sil = silhouette_score(X, labels)
st.metric("Silhouette Score", f"{sil:.4f}")

# ------------------------------------------
# User Guidance
# ------------------------------------------
st.info(
    "Articles grouped in the same cluster share similar vocabulary and themes. "
    "These clusters can be used for **automatic tagging**, "
    "**recommendations**, and **content organization**."
)