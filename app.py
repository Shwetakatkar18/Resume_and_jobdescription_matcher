import os
import io
import base64
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF for reading PDF files
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pdf2image

#LangChain imports
from langchain_ollama import OllamaLLM #class to interact with gemma through ollama 
from langchain.prompts import PromptTemplate #for structured prompt tempelate for consistent input to llm
from langchain_core.runnables import RunnableSequence #enable channing of prompts and llm together

#embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#create a Gemma model
llm = OllamaLLM(model="gemma:2b")

#Extract all text from uploaded PDF resume
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

#Calculate cosine similarity between embeddings
def get_similarity_score(text1, text2):
    vec1 = embedder.encode(text1)
    vec2 = embedder.encode(text2)
    return cosine_similarity([vec1], [vec2])[0][0]

#Prompt to explain the match score
score_explainer_prompt = PromptTemplate.from_template("""
You are an ATS evaluation expert. The resume scored {score}% against the following job description.

Resume:
{resume}

Job Description:
{jd}

Explain briefly why this score was given.
""")

#Runnable chain for score explanation
score_explainer_chain = score_explainer_prompt | llm

#prompt to suggest resume improvements
improvement_prompt = PromptTemplate.from_template("""
You are a resume optimization expert.

Resume:
{resume}

Job Description:
{jd}

Suggest detailed improvements to make the resume a better fit for this job.
""")

#Create Runnable chain for improvement suggestions
improvement_chain = improvement_prompt | llm

#Streamlit UI setup
st.set_page_config(page_title="ATS Resume Expert")
st.title("Resume vs JD Matcher")

#Input job description
jd_text = st.text_area("Paste Job Description")

#Upload resume PDF
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

#If resume uploaded
if uploaded_file:
    st.success("Resume uploaded successfully!")

    #Button to evaluate match
    if st.button("Get Match Score & LLM Feedback"):
        resume_text = extract_text_from_pdf(uploaded_file)
        score = get_similarity_score(resume_text, jd_text)
        formatted_score = f"{score * 100:.2f}"

        #Display score
        st.markdown(f"### Match Score: `{formatted_score}%`")

        #Get LLM feedback and suggestions
        explanation = score_explainer_chain.invoke({
            "resume": resume_text,
            "jd": jd_text,
            "score": formatted_score
        })

        improvement = improvement_chain.invoke({
            "resume": resume_text,
            "jd": jd_text
        })

        #Display results
        st.markdown("### Score Explanation")
        st.write(explanation)

        st.markdown("### Resume Improvement Suggestions")
        st.write(improvement)

#If no file uploaded
else:
    st.warning("Please upload your resume PDF.")

#Gemini Vision placeholder (not used with Gemma)
st.markdown("---")
st.button("Analyze with Gemini Vision (Disabled for Gemma)", disabled=True)