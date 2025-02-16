import streamlit as st
import google.generativeai as genai
import subprocess
from streamlit_ace import st_ace
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv  # Secure environment variable handling

# Load environment variables
load_dotenv()
key = os.getenv("GEMINI_API_KEY")

if not key:
    st.error("Error: Gemini API key not found. Please set GEMINI_API_KEY in a .env file.")
    st.stop()

# Initialize session state
def init_session_state():
    if "tabs" not in st.session_state:
        new_tab_id = str(uuid.uuid4())
        st.session_state["tabs"] = {
            new_tab_id: {
                "code": "",
                "query": "",
                "review_output": "",
                "run_output": "",
                "fixed_code": "",
                "editor_key": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": "python"
            }
        }
        st.session_state["current_tab"] = new_tab_id

# Run Python code
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
        return "Error: Code execution timed out (30-second limit)"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# Review code or MySQL query using Gemini AI
def review_content(content, tab_id, content_type):
    try:
        prompt = f"Review this {content_type} and suggest improvements:\n{content}"
        response = model.generate_content(prompt)
        st.session_state["tabs"][tab_id]["review_output"] = response.text

        if content_type == "Python code" and "```python" in response.text:
            fixed_code = response.text.split("```python")[1].split("```")[0].strip()
            if fixed_code:
                st.session_state["tabs"][tab_id]["fixed_code"] = fixed_code
    except Exception as e:
        st.error(f"Error during review: {str(e)}")

# Create a new tab
def create_new_tab():
    new_tab_id = str(uuid.uuid4())
    st.session_state["tabs"][new_tab_id] = {
        "code": "",
        "query": "",
        "review_output": "",
        "run_output": "",
        "fixed_code": "",
        "editor_key": 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "python"
    }
    st.session_state["current_tab"] = new_tab_id

# Delete a tab
def delete_tab(tab_id):
    if tab_id in st.session_state["tabs"]:
        del st.session_state["tabs"][tab_id]
        if st.session_state["tabs"]:
            st.session_state["current_tab"] = list(st.session_state["tabs"].keys())[0]
        else:
            create_new_tab()

# Apply fixed code
def apply_fixed_code(tab_id):
    if st.session_state["tabs"][tab_id]["fixed_code"]:
        st.session_state["tabs"][tab_id]["code"] = st.session_state["tabs"][tab_id]["fixed_code"]
        st.session_state["tabs"][tab_id]["editor_key"] += 1

# Initialize session state
init_session_state()

# Configure Gemini model
genai.configure(api_key=key)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Sidebar for managing tabs
with st.sidebar:
    st.title("Review History")

    if st.button("New Review", type="primary"):
        create_new_tab()
        st.rerun()

    st.divider()

    for tab_id, tab_data in st.session_state["tabs"].items():
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f"Review from {tab_data['timestamp']} ({tab_data['type']})",
                key=f"history_{tab_id}",
                use_container_width=True
            ):
                st.session_state["current_tab"] = tab_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{tab_id}"):
                delete_tab(tab_id)
                st.rerun()

# Main content area
current_tab = st.session_state.get("current_tab", None)
current_tab_data = st.session_state["tabs"][current_tab]

st.title("Python & MySQL Code/Query Reviewer")

# Select content type
content_type = st.radio(
    "Select content type:",
    ["Python Code", "MySQL Query"],
    index=0 if current_tab_data["type"] == "python" else 1
)
st.session_state["tabs"][current_tab]["type"] = "python" if content_type == "Python Code" else "mysql"

if content_type == "Python Code":
    code = st_ace(
        language="python",
        theme="monokai",
        height=300,
        value=current_tab_data["code"],
        key=f"editor_{current_tab}_{current_tab_data['editor_key']}"
    )
    st.session_state["tabs"][current_tab]["code"] = code

    # Run Python Code
    if st.button("Run Code", key=f"run_{current_tab}"):
        if code.strip():
            result = run_code(code, current_tab)
            st.session_state["tabs"][current_tab]["run_output"] = result
        else:
            st.warning("Please enter some Python code.")

    if current_tab_data["run_output"]:
        st.markdown("#### Output:")
        st.code(current_tab_data["run_output"])

else:  # MySQL Query
    query = st.text_area(
        "Enter MySQL Query",
        value=current_tab_data["query"],
        height=150,
        key=f"query_{current_tab}"
    )
    st.session_state["tabs"][current_tab]["query"] = query

# Review Section
if st.button("Review", key=f"review_{current_tab}"):
    if content_type == "Python Code" and code.strip():
        review_content(code, current_tab, "Python code")
    elif content_type == "MySQL Query" and query.strip():
        review_content(query, current_tab, "SQL query")
    else:
        st.warning(f"Please enter some {content_type}.")

if current_tab_data["review_output"]:
    st.markdown("#### Review Feedback:")
    st.markdown(current_tab_data["review_output"])

    if content_type == "Python Code" and current_tab_data["fixed_code"]:
        if st.button("Apply Fixed Code", key=f"apply_{current_tab}"):
            apply_fixed_code(current_tab)
            st.rerun()
