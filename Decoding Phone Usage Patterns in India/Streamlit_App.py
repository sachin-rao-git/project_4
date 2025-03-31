import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Data load
df = pd.read_csv('phone_usage_india.csv')  # Apna file name daal dena

# Clustering (Step 4 ka part)
features = ['Screen Time (hrs/day)', 'Data Usage (GB/month)', 'Social Media Time (hrs/day)', 
            'Streaming Time (hrs/day)', 'Gaming Time (hrs/day)']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters assume kiya
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Classification (Step 5 ka part)
X_class = df[features]  # Same features for simplicity
y_class = df['Primary Use']
rf = RandomForestClassifier(random_state=42)
rf.fit(X_class, y_class)  # Model train kar rahe hai

# Streamlit App
st.title("Decoding Phone Usage Patterns in India")

# Sidebar mein options
st.sidebar.header("Explore Data")
option = st.sidebar.selectbox("Choose Analysis", ["EDA", "Clusters", "Prediction"])

if option == "EDA":
    st.subheader("Screen Time Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Screen Time (hrs/day)'], bins=20, ax=ax)
    st.pyplot(fig)

    st.subheader("Primary Use Count")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Primary Use', data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif option == "Clusters":
    st.subheader("User Clusters")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Screen Time (hrs/day)', y='Social Media Time (hrs/day)', hue='Cluster', data=df, ax=ax)
    st.pyplot(fig)

elif option == "Prediction":
    st.subheader("Predict Primary Use")
    screen_time = st.slider("Screen Time (hrs/day)", 0.0, 24.0, 5.0)
    data_usage = st.slider("Data Usage (GB/month)", 0.0, 50.0, 20.0)
    social_media = st.slider("Social Media Time (hrs/day)", 0.0, 24.0, 2.0)
    streaming = st.slider("Streaming Time (hrs/day)", 0.0, 24.0, 1.0)
    gaming = st.slider("Gaming Time (hrs/day)", 0.0, 24.0, 1.0)

    # Input data for prediction
    input_data = [[screen_time, data_usage, social_media, streaming, gaming]]
    pred = rf.predict(input_data)
    st.write(f"Predicted Primary Use: {pred[0]}")

# Run karna: `streamlit run app.py`