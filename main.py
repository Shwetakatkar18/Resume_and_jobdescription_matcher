import streamlit as st

st.set_page_config(page_title="AI-Powered Talent Matcher", layout="centered")

st.title("Resume Matcher")
st.subheader("Choose an Option:")

# These files must be inside the "pages/" folder
st.page_link("pages/1_Resume_analysis.py", label="Resume Analysis", icon="📄")
st.page_link("pages/2_company_profilling.py", label="Company Profiling", icon="🏆")
