# Import required libraries
import os
import io
import base64
import streamlit as st
from PIL import Image
import fitz  # To read PDF files
from sklearn.metrics.pairwise import cosine_similarity  # To compare text similarity
from sentence_transformers import SentenceTransformer  # For converting text to vectors
import pdf2image  # (Optional) Convert PDF pages to images if needed

# LangChain imports for LLM usage
from langchain_ollama import OllamaLLM  # To use local LLMs like Gemma via Ollama
from langchain.prompts import PromptTemplate  # To create prompt templates
from langchain_core.runnables import RunnableSequence  # To chain prompts

# Load sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Fast, small model to get text embeddings

# Load Gemma model using Ollama locally
llm = OllamaLLM(model="gemma:2b")  # Make sure Gemma is pulled using: ollama pull gemma:2b

# ========== Helper Functions ==========

# Function to extract text from uploaded PDF resume
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])  # Combine all page texts

# Function to calculate similarity score between two texts
def get_similarity_score(text1, text2):
    vec1 = embedder.encode(text1)  # Convert text1 to vector
    vec2 = embedder.encode(text2)  # Convert text2 to vector
    return cosine_similarity([vec1], [vec2])[0][0]  # Return similarity score

# Function to extract relevant category text from a resume or JD
def extract_category(text, category_keywords):
    text_lower = text.lower()  # Convert text to lowercase
    return "\n".join([line for line in text_lower.split('\n') if any(word in line for word in category_keywords)])

# ========== LLM Prompt Templates ==========

# Prompt to explain the match score
score_explainer_prompt = PromptTemplate.from_template("""
You are an ATS evaluation expert. The resume scored {score}% against the following job description.

Categories considered: technical skills, soft skills, experience, package, location.

Resume:
{resume}

Job Description:
{jd}

Explain briefly why this score was given and what can be improved.
""")

# Prompt to generate improvement suggestions
improvement_prompt = PromptTemplate.from_template("""
You are a resume and job match improvement expert.

Your task is to give highly specific, actionable improvement suggestions to align the resume more closely with the job description.

Resume:
{resume}

Job Description:
{jd}

Now give category-wise recommendations as follows:

1. Technical Skills - Mention which job-required technical skills are missing from the resume and how to include them effectively.
2. Soft Skills - Check if relevant soft skills like communication, leadership, etc., are covered in resume. If not, recommend how to insert them.
3. Experience - Are years of experience, roles, or project details aligning? If not, mention what to reword or add.
4. Location - Does the resume mention the preferred location or relocation readiness? Suggest phrasing.
5. Grammar & Formatting - Point out grammatical issues or formatting flaws that may reduce professionalism.

Respond in bullet format under each category.
""")

# Create the full chains using LLM + prompts
score_explainer_chain = score_explainer_prompt | llm
improvement_chain = improvement_prompt | llm

# ========== Streamlit Web App UI ==========

# Setup page title
st.set_page_config(page_title="ATS Resume Expert")

# App header
st.title("Resume vs JD Matcher (LLM + AI Based)")

# Input: Job Description
jd_text = st.text_area("Paste Job Description")  # User pastes JD

# Input: Upload Resume (PDF only)
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Check if file is uploaded
if uploaded_file:
    st.success("Resume uploaded successfully!")

    # Button to trigger matching
    if st.button("Get Match Score & LLM Feedback"):

        # Extract text from uploaded resume
        resume_text = extract_text_from_pdf(uploaded_file)

        # Calculate and display overall similarity score
        overall_score = get_similarity_score(resume_text, jd_text)
        formatted_score = f"{overall_score * 100:.2f}"
        st.markdown(f"### Overall Match Score: `{formatted_score}%`")

        # Category keywords for technical, soft skills, etc.
        categories = {
            "Technical Skills": ["python", "java", "sql", "flask", "fastapi", "ml", "api", "kafka", "pyspark", "etl"],
            "Soft Skills": ["communication", "team", "collaboration", "leadership", "adaptability", "problem-solving"],
            "Experience": ["experience", "worked", "handled", "years", "projects", "internship", "role"],
            "Location": ["location", "remote", "onsite", "hyderabad", "bangalore", "delhi", "pune", "mumbai"],
        }

        # Section to display category-wise similarity
        st.markdown("Category-wise Match Scores")

        # Store category-specific matched sections
        category_sections = {}

        for category, keywords in categories.items():
            resume_section = extract_category(resume_text, keywords)
            jd_section = extract_category(jd_text, keywords)

            category_sections[category] = (resume_section, jd_section)

            if resume_section.strip() == "" or jd_section.strip() == "":
                score = 0.0  # If no content found for category
            else:
                score = get_similarity_score(resume_section, jd_section)

            # Show category score
            st.markdown(f"**{category}:** `{score * 100:.2f}%`")

        # --------- LLM Generated Feedback ---------
        st.markdown("Recommendations and Improvements")

        # 1. Why the score was given
        explanation = score_explainer_chain.invoke({
            "resume": resume_text,
            "jd": jd_text,
            "score": formatted_score
        })
        st.subheader("Why This Score Was Given")
        st.write(explanation)

        # 2. Resume optimization tips
        improvement = improvement_chain.invoke({
            "resume": resume_text,
            "jd": jd_text
        })
        st.subheader("Resume Optimization Suggestions")
        st.write(improvement)

else:
    st.warning("Please upload your resume PDF.")  # Show warning if no file uploaded
