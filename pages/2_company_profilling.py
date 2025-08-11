# ========== Import all required libraries ==========

import os  # To work with folders and file paths
import fitz  # PyMuPDF - to extract text from PDF files
import streamlit as st  # To create the web interface
from sentence_transformers import SentenceTransformer  # To create embeddings from text
from sklearn.metrics.pairwise import cosine_similarity  # (Not used directly here)
import chromadb  # To store and search resume embeddings
from chromadb.config import Settings  # For ChromaDB settings (optional here)

# ========== Setup Streamlit and file paths ==========

# Set up the page title and layout
st.set_page_config(page_title="Company Profiling - Resume Matcher", layout="centered")

# Folder where all resume PDFs are stored
RESUME_FOLDER = 'D:\\resume_web\\resume_samples'  # Change if your folder is different

# Folder where ChromaDB will store the embeddings
CHROMA_PATH = "chroma_db"

# ========== Load models and setup ChromaDB ==========

# Load the SentenceTransformer model to create embeddings for text
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create a new collection or connect to an existing one called "resumes"
collection = chroma_client.get_or_create_collection(name="resumes")

# ========== Function to Embed Resumes from PDF ==========

def embed_and_store_resumes():
    """
    This function reads all PDF resumes from the folder, extracts the text,
    creates embeddings, and stores them into ChromaDB.
    """
    for file in os.listdir(RESUME_FOLDER):
        if file.endswith(".pdf"):
            full_path = os.path.join(RESUME_FOLDER, file)

            try:
                with open(full_path, "rb") as f:
                    pdf_doc = fitz.open("pdf", f.read())
                    # Combine text from all pages
                    text = " ".join([page.get_text() for page in pdf_doc])
                
                # Convert resume text into a vector (embedding)
                embedding = embedder.encode(text).tolist()
                doc_id = file  # Use file name as unique ID

                # Add document, its embedding, and ID to ChromaDB
                collection.add(documents=[text], embeddings=[embedding], ids=[doc_id])

            except Exception as e:
                st.error(f"Failed to embed {file}: {e}")

# ========== Function to Search Top Matching Resumes ==========

def search_top_k_resumes(jd_text, top_k=5):
    """
    This function takes a job description and finds top_k similar resumes
    by comparing their embeddings using ChromaDB.
    """
    jd_vector = embedder.encode(jd_text).tolist()  # Embed the job description
    results = collection.query(query_embeddings=[jd_vector], n_results=top_k)
    return results  # Returns IDs and similarity scores

# ========== Streamlit UI Starts ==========

# App title
st.title("Company Profiling - Resume Matcher (RAG + AI)")

# Welcome message
st.markdown("Welcome! Upload resumes into the database and search for the best matches based on the job description.")

# --- STEP 1: Embed Resumes Section ---
with st.expander("Step 1: Embed Resumes from Folder"):
    st.write(f"Resume folder path: `{RESUME_FOLDER}`")
    st.warning("Make sure the folder contains only `.pdf` files.")

    # Button to start the embedding process
    if st.button("Embed All Resumes"):
        embed_and_store_resumes()
        st.success("All resumes have been successfully embedded into the database!")

# --- STEP 2: Paste Job Description ---
st.divider()  # Adds a horizontal line separator
jd_input = st.text_area("step 2: Paste the Job Description", height=200, placeholder="Paste the JD here...")

# --- STEP 3: Enter number of top resumes and search ---
if jd_input:
    st.divider()
    st.subheader("Step 3: Get Top Matching Resumes")

    # User can choose how many top resumes to fetch
    top_k = st.number_input("Enter the number of top matching resumes you want to see:", min_value=1, max_value=50, value=5)

    # Search resumes when button is clicked
    if st.button("Find Top Matching Resumes"):
        results = search_top_k_resumes(jd_input, top_k=top_k)

        if results and results["ids"]:
            st.markdown("Top Matching Resumes:")

            # Display each matching resume with similarity score
            for i, (doc_id, score) in enumerate(zip(results['ids'][0], results['distances'][0])):
                match_percent = (1 - score) * 100  # Convert distance to percentage
                st.markdown(f"""
                ### {i+1}.  File: `{doc_id}`
                -  **Match Score:** `{match_percent:.2f}%`
                """)
                st.markdown("---")  # Divider between resumes

        else:
            st.warning("No matching resumes found. Please ensure resumes are embedded and JD is valid.")

else:
    st.info("Paste a job description to search for matching resumes.")
