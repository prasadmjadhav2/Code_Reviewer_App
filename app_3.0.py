import streamlit as st
import requests
import google.generativeai as genai
import subprocess
import uuid
import sqlite3
import hashlib
import os
from dotenv import load_dotenv
import io
from datetime import datetime
from PIL import Image

# Load environment variables
load_dotenv()
key = os.getenv("GEMINI_API_KEY") 

if not key:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY as an environment variable.")
    st.stop()

genai.configure(api_key=key)
model = genai.GenerativeModel("gemini-pro")
model2 = genai.GenerativeModel("gemini-pro-vision")

# Database setup
def init_db():
    conn = sqlite3.connect('code_review.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS reviews 
                 (id TEXT PRIMARY KEY, username TEXT, code TEXT, review_output TEXT,
                  run_output TEXT, fixed_code TEXT, timestamp TEXT, 
                  FOREIGN KEY (username) REFERENCES users(username))''')
    conn.commit()
    conn.close()

# Authentication Functions
def hash_password(password):
    return hashlib.sha256((password + "secure_salt").encode()).hexdigest()

def authenticate(username, password):
    conn = sqlite3.connect('code_review.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('SELECT password FROM users WHERE username=?', (username,))
    result = c.fetchone()
    conn.close()
    return result and result[0] == hash_password(password)

def register_user(username, password):
    if not username or not password:
        return False
    conn = sqlite3.connect('code_review.db', check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users VALUES (?, ?)', (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Function to execute Python code
def run_code(code):
    """Execute Python code and capture output."""
    try:
        temp_file = "temp_script.py"
        with open(temp_file, "w") as f:
            f.write(code)
        
        result = subprocess.run(["python3", temp_file], capture_output=True, text=True, timeout=30)
        os.remove(temp_file)
        
        output = result.stdout
        if result.stderr:
            output += "\nErrors:\n" + result.stderr
        
        return output
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30 second limit)"
    except Exception as e:
        return f"Error: {str(e)}"

# Extract code from image using Gemini AI
def extract_code_from_image(uploaded_image):
    try:
        image = Image.open(io.BytesIO(uploaded_image.getvalue()))
        response = model2.generate_content(["Extract only the programming code from this image:", image])
        return response.text.strip() if response.text else ""
    except Exception as e:
        st.error(f"Error extracting code: {str(e)}")
        return ""

# Review code using Gemini AI
def review_code(code):
    """Send code to Gemini AI for review."""
    try:
        prompt = f"Review this code and provide fixes:\n{code}"
        response = model.generate_content(prompt)
        return response.text if response.text else "No review generated."
    except Exception as e:
        return f"Error during review: {str(e)}"

# Streamlit UI
st.title("AI-Powered Code Review & Execution")

# Authentication Section
if "username" not in st.session_state:
    st.session_state["username"] = None

if st.session_state["username"]:
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state["username"] = None
        st.experimental_rerun()
else:
    choice = st.sidebar.radio("Login / Register", ["Login", "Register"])
    
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if choice == "Login":
        if st.sidebar.button("Login"):
            if authenticate(username, password):
                st.session_state["username"] = username
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid credentials.")
    
    elif choice == "Register":
        if st.sidebar.button("Register"):
            if register_user(username, password):
                st.sidebar.success("Account created. Please login.")
            else:
                st.sidebar.error("Username already exists.")

# Main App Interface
if st.session_state["username"]:
    tab = st.selectbox("Select Mode", ["Code Review", "Run Code", "Upload Image"])

    if tab == "Code Review":
        st.subheader("Paste your code below")
        code = st.text_area("Your Code", height=200)

        if st.button("Review Code"):
            if code:
                review_output = review_code(code)
                st.text_area("AI Review", review_output, height=200)
            else:
                st.warning("Please enter code for review.")

    elif tab == "Run Code":
        st.subheader("Run Python Code")
        code = st.text_area("Your Code", height=200)

        if st.button("Execute Code"):
            if code:
                output = run_code(code)
                st.text_area("Execution Output", output, height=200)
            else:
                st.warning("Please enter code to execute.")

    elif tab == "Upload Image":
        st.subheader("Extract Code from Image")
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file and st.button("Extract Code"):
            extracted_code = extract_code_from_image(uploaded_file)
            st.text_area("Extracted Code", extracted_code, height=200)

# Initialize Database
init_db()
