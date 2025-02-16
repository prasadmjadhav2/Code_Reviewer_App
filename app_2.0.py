import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
import subprocess
from dotenv import load_dotenv
from streamlit_ace import st_ace
import plotly.express as px
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from prophet import Prophet
import featuretools as ft
import optuna

# Load API Key from .env file
load_dotenv()
key = os.getenv("GEMINI_API_KEY")

if not key:
    st.error("⚠️ Gemini API key not found. Set GEMINI_API_KEY in a .env file.")
    st.stop()

genai.configure(api_key=key)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

st.title("🚀 AI-Powered Data Science & ML Toolkit")

# Sidebar options
option = st.sidebar.radio(
    "Choose a tool:",
    ["🔍 Python Code Review", "📊 Data Analysis", "🤖 Machine Learning", "🧠 AI Chat Assistant"]
)

# 📌 Python Code Review & Execution
if option == "🔍 Python Code Review":
    st.subheader("Python Code Review & Execution")
    code = st_ace(language="python", theme="monokai", height=300)

    col1, col2 = st.columns(2)
    
    if col1.button("Run Code"):
        with open("temp.py", "w") as f:
            f.write(code)
        result = subprocess.run(["python3", "temp.py"], capture_output=True, text=True)
        st.code(result.stdout if result.stdout else result.stderr)

    if col2.button("Optimize Code"):
        response = model.generate_content(f"Optimize this Python code for efficiency:\n{code}")
        st.markdown(response.text)

# 📌 Data Analysis
elif option == "📊 Data Analysis":
    st.subheader("📊 Data Analysis Toolkit")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())

        analysis_type = st.radio("Choose analysis type:", ["EDA", "A/B Testing", "Hypothesis Testing", "Time Series Forecasting"])

        # 🔎 Exploratory Data Analysis (EDA)
        if analysis_type == "EDA":
            st.write("### Summary Statistics")
            st.write(df.describe())

            st.write("### Missing Values")
            st.write(df.isnull().sum())

            st.write("### Correlation Matrix")
            st.write(df.corr())

            # Interactive Visualization
            col_x, col_y = st.columns(2)
            with col_x:
                x_axis = st.selectbox("X-Axis", df.columns)
            with col_y:
                y_axis = st.selectbox("Y-Axis", df.columns)

            fig = px.scatter(df, x=x_axis, y=y_axis, title="Scatter Plot")
            st.plotly_chart(fig)

        # 🔬 A/B Testing
        elif analysis_type == "A/B Testing":
            col1, col2 = st.columns(2)
            with col1:
                group_a = st.selectbox("Select Group A Column", df.columns)
            with col2:
                group_b = st.selectbox("Select Group B Column", df.columns)

            if st.button("Run A/B Test"):
                t_stat, p_value = stats.ttest_ind(df[group_a].dropna(), df[group_b].dropna())
                st.write(f"T-Statistic: {t_stat:.4f}, P-Value: {p_value:.4f}")
                st.success("Significant Difference" if p_value < 0.05 else "No Significant Difference")

        # 🧪 Hypothesis Testing
        elif analysis_type == "Hypothesis Testing":
            cat_var1 = st.selectbox("Categorical Variable 1", df.columns)
            cat_var2 = st.selectbox("Categorical Variable 2", df.columns)

            if st.button("Run Hypothesis Test"):
                contingency_table = pd.crosstab(df[cat_var1], df[cat_var2])
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                st.write(f"Chi-Square Statistic: {chi2_stat:.4f}, P-Value: {p_value:.4f}")
                st.success("Significant Association" if p_value < 0.05 else "No Significant Association")

        # 📈 Time Series Forecasting
        elif analysis_type == "Time Series Forecasting":
            time_col = st.selectbox("Time Column", df.columns)
            value_col = st.selectbox("Value Column", df.columns)

            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)

            model = Prophet()
            df_prophet = df[[value_col]].reset_index().rename(columns={time_col: "ds", value_col: "y"})
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            fig = px.line(forecast, x="ds", y="yhat", title="Time Series Forecast")
            st.plotly_chart(fig)

# 📌 Machine Learning
elif option == "🤖 Machine Learning":
    st.subheader("🚀 Auto ML Model Selection & Hyperparameter Tuning")

    uploaded_file = st.file_uploader("Upload CSV for ML", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview", df.head())

        ml_type = st.radio("Choose ML type:", ["Regression", "Classification", "Clustering"])

        target = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 📌 Auto Feature Engineering
        st.write("🚀 Running Auto Feature Engineering...")
        es = ft.EntitySet(id="data")
        es = es.entity_from_dataframe(entity_id="df", dataframe=df, index="index")
        features, feature_defs = ft.dfs(entityset=es, target_entity="df")
        st.write("Generated Features:", features.head())

        # 📌 Hyperparameter Tuning
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 10, 500)
            max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train, y_train)
            return model.score(X_test, y_test)

        if ml_type == "Regression":
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=10)
            st.write("Best Hyperparameters:", study.best_params)

# 📌 AI Chat Assistant
elif option == "🧠 AI Chat Assistant":
    st.subheader("💡 AI Chat for Queries")
    user_query = st.text_input("Ask me anything related to AI & Data Science:")
    
    if st.button("Ask AI"):
        response = model.generate_content(user_query)
        st.markdown(response.text)
