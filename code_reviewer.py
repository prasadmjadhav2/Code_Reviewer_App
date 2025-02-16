import streamlit as st
import google.generativeai as genai
import subprocess
from streamlit_ace import st_ace
import uuid
import os
from datetime import datetime
from dotenv import load_dotenv  # Load .env file

# Load environment variables securely
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set GEMINI_API_KEY as an environment variable.")
    st.stop()

# Initialize Gemini AI model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")


# Initialize session state
def init_session_state():
    if "tabs" not in st.session_state:
        new_tab_id = str(uuid.uuid4())
        st.session_state["tabs"] = {
            new_tab_id: {
                "code": "",
                "review_output": "",
                "run_output": "",
                "fixed_code": "",
                "editor_key": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        st.session_state["current_tab"] = new_tab_id


# Execute Python code securely
def run_code(code, tab_id):
    temp_file = f"temp_{tab_id}.py"
    
    try:
        with open(temp_file, "w") as f:
            f.write(code)
        
        result = subprocess.run(
            ["python3", temp_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout
        if result.stderr:
            output += "\nErrors:\n" + result.stderr
        
        return output
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30 second limit)"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# Review and fix Python code using Gemini AI
def review_code(code, tab_id):
    try:
        prompt = f"Review and provide fixes for the following Python code:\n{code}"
        response = model.generate_content(prompt)
        review_text = response.text if response else "No response from Gemini AI."
        
        st.session_state["tabs"][tab_id]["review_output"] = review_text

        # Extract fixed code if available
        if "```python" in review_text:
            fixed_code = review_text.split("```python")[1].split("```")[0].strip()
            st.session_state["tabs"][tab_id]["fixed_code"] = fixed_code if fixed_code else ""
    except Exception as e:
        st.error(f"Error during code review: {str(e)}")


# Create a new tab
def create_new_tab():
    new_tab_id = str(uuid.uuid4())
    st.session_state["current_tab"] = new_tab_id
    st.session_state["tabs"][new_tab_id] = {
        "code": "",
        "review_output": "",
        "run_output": "",
        "fixed_code": "",
        "editor_key": 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# Delete a tab
def delete_tab(tab_id):
    if tab_id in st.session_state["tabs"]:
        if tab_id == st.session_state["current_tab"]:
            remaining_tabs = [t for t in st.session_state["tabs"].keys() if t != tab_id]
            st.session_state["current_tab"] = remaining_tabs[0] if remaining_tabs else create_new_tab()
        
        del st.session_state["tabs"][tab_id]


# Apply fixed code
def apply_fixed_code(tab_id):
    if st.session_state["tabs"][tab_id]["fixed_code"]:
        st.session_state["tabs"][tab_id]["code"] = st.session_state["tabs"][tab_id]["fixed_code"]
        st.session_state["tabs"][tab_id]["editor_key"] += 1


# Initialize session state
init_session_state()

# Sidebar for managing tabs
with st.sidebar:
    st.title("Code Review History")
    
    if st.button("New Review", type="primary"):
        create_new_tab()
        st.rerun()
    
    st.divider()
    
    for tab_id, tab_data in st.session_state["tabs"].items():
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"Review from {tab_data['timestamp']}", key=f"history_{tab_id}", use_container_width=True):
                st.session_state["current_tab"] = tab_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{tab_id}"):
                delete_tab(tab_id)
                st.rerun()


# Main content area
current_tab = st.session_state.get("current_tab", None)
current_tab_data = st.session_state["tabs"][current_tab]

st.title("Python Code Reviewer")

# Code Editor
code = st_ace(
    language="python",
    theme="monokai",
    height=300,
    value=current_tab_data["code"],
    key=f"editor_{current_tab}_{current_tab_data['editor_key']}"
)

# Update code in session state
st.session_state["tabs"][current_tab]["code"] = code

# Run Code Section
if st.button("Run Code", key=f"run_{current_tab}"):
    if code.strip():
        result = run_code(code, current_tab)
        st.session_state["tabs"][current_tab]["run_output"] = result
    else:
        st.warning("Please enter some code.")

if current_tab_data["run_output"]:
    st.markdown("#### Output:")
    st.code(current_tab_data["run_output"])

# Code Review Section
if st.button("Review Code", key=f"review_{current_tab}"):
    if code.strip():
        review_code(code, current_tab)
    else:
        st.warning("Please enter some code.")

if current_tab_data["review_output"]:
    st.markdown("#### Review Feedback:")
    st.markdown(current_tab_data["review_output"])
    
    if current_tab_data["fixed_code"]:
        if st.button("Apply Fixed Code", key=f"apply_{current_tab}"):
            apply_fixed_code(current_tab)
            st.rerun()
