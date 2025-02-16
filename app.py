import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import subprocess
from streamlit_ace import st_ace
import uuid
import os
from dotenv import load_dotenv
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN

# Load environment variables
load_dotenv()
key = os.getenv("GEMINI_API_KEY")

if not key:
    st.error("Error: Gemini API key not found. Set GEMINI_API_KEY in a .env file.")
    st.stop()

genai.configure(api_key=key)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

st.title("🔍 AI-Powered Data Analysis & ML Toolkit")

# Sidebar for different tools
option = st.sidebar.radio(
    "Choose a tool:",
    ["Python Code Review", "MySQL Query Review", "Data Analysis", "Machine Learning"]
)

# Python Code Review
if option == "Python Code Review":
    st.subheader("Python Code Review & Execution")

    code = st_ace(language="python", theme="monokai", height=300)
    
    if st.button("Run Code"):
        with open("temp.py", "w") as f:
            f.write(code)
        
        result = subprocess.run(["python3", "temp.py"], capture_output=True, text=True)
        st.code(result.stdout if result.stdout else result.stderr)

    if st.button("Review Code"):
        response = model.generate_content(f"Review this Python code and suggest improvements:\n{code}")
        st.markdown(response.text)

# MySQL Query Review
elif option == "MySQL Query Review":
    st.subheader("MySQL Query Review")
    query = st.text_area("Enter MySQL Query")

    if st.button("Review Query"):
        response = model.generate_content(f"Review this MySQL query and suggest improvements:\n{query}")
        st.markdown(response.text)

# Data Analysis Tools
elif option == "Data Analysis":
    st.subheader("📊 Data Analysis Toolkit")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())

        analysis_type = st.radio("Choose analysis type:", ["EDA", "A/B Testing", "Hypothesis Testing", "Time Series Analysis"])

        # Exploratory Data Analysis
        if analysis_type == "EDA":
            st.write("### Summary Statistics")
            st.write(df.describe())

            st.write("### Missing Values")
            st.write(df.isnull().sum())

            st.write("### Correlation Matrix")
            st.write(df.corr())

        # A/B Testing
        elif analysis_type == "A/B Testing":
            st.write("### A/B Test (T-Test)")
            col1, col2 = st.columns(2)
            with col1:
                group_a = st.selectbox("Select Group A Column", df.columns)
            with col2:
                group_b = st.selectbox("Select Group B Column", df.columns)

            if st.button("Run A/B Test"):
                t_stat, p_value = stats.ttest_ind(df[group_a].dropna(), df[group_b].dropna())
                st.write(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
                if p_value < 0.05:
                    st.success("Significant difference between groups.")
                else:
                    st.warning("No significant difference found.")

        # Hypothesis Testing
        elif analysis_type == "Hypothesis Testing":
            st.write("### Hypothesis Testing (Chi-Square)")
            col1, col2 = st.columns(2)
            with col1:
                cat_var1 = st.selectbox("Select Categorical Variable 1", df.columns)
            with col2:
                cat_var2 = st.selectbox("Select Categorical Variable 2", df.columns)

            if st.button("Run Hypothesis Test"):
                contingency_table = pd.crosstab(df[cat_var1], df[cat_var2])
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                st.write(f"Chi-Square Statistic: {chi2_stat:.4f}, P-Value: {p_value:.4f}")
                if p_value < 0.05:
                    st.success("Significant association found.")
                else:
                    st.warning("No significant association.")

        # Time Series Analysis
        elif analysis_type == "Time Series Analysis":
            time_col = st.selectbox("Select Time Column", df.columns)
            value_col = st.selectbox("Select Value Column", df.columns)

            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)

            st.line_chart(df[value_col])

# Machine Learning Methods
elif option == "Machine Learning":
    st.subheader("🤖 Machine Learning Models")

    uploaded_file = st.file_uploader("Upload CSV for ML", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())

        ml_type = st.radio("Choose ML type:", ["Regression", "Classification", "Clustering"])

        target = st.selectbox("Select Target Column", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Regression
        if ml_type == "Regression":
            model_choice = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"])
            if st.button("Train Model"):
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor()
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                st.write(f"Model R² Score: {score:.4f}")

        # Classification
        elif ml_type == "Classification":
            model_choice = st.selectbox("Choose Model", ["Logistic Regression", "SVM"])
            if st.button("Train Model"):
                if model_choice == "Logistic Regression":
                    model = LogisticRegression()
                else:
                    model = SVC()
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                st.write(f"Model Accuracy: {score:.4f}")

        # Clustering
        elif ml_type == "Clustering":
            model_choice = st.selectbox("Choose Model", ["K-Means", "DBSCAN"])
            n_clusters = st.slider("Number of Clusters", 2, 10, 3) if model_choice == "K-Means" else None
            if st.button("Train Model"):
                if model_choice == "K-Means":
                    model = KMeans(n_clusters=n_clusters)
                else:
                    model = DBSCAN()
                labels = model.fit_predict(X)
                df["Cluster"] = labels
                st.write("### Clustered Data", df.head())
