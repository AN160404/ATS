from dotenv import load_dotenv
from docx import Document
from PyPDF2 import PdfReader
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
import os
api_key = os.getenv('api_key')

# LLMs
llm = GoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")

# Reading docx
def docx_read(uploaded_file):
    text1 = []
    doc = Document(uploaded_file)
    for para in doc.paragraphs:
        text1.append(para.text)
    return "\n".join(text1)

# Reading pdf
def pdf_read(uploaded_file):
    text2 = ""
    pdf = PdfReader(uploaded_file)
    for page in pdf.pages:
        text2 += page.extract_text()
    return text2

# parsing uploaded file
def resume_parse(uploaded_file):
    if uploaded_file.name.endswith(".docx"):
        return docx_read(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        return pdf_read(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Upload either a PDF or DOCX file.")

# Dividing into chunks
def create_chunks(resume_text):
    resume_chunks = []
    current_section = []
    section_headers = ["Experience", "Education", "Skills", "Projects", "Awards"]  

    # Iterate through lines and group them by sections
    lines = resume_text.split("\n")
    for line in lines:
        line = line.strip()
        if any(header in line for header in section_headers):  
            if current_section:  
                resume_chunks.append("\n".join(current_section))
            current_section = [line]  
        else:
            current_section.append(line)  

    if current_section:
        resume_chunks.append("\n".join(current_section))  

    return resume_chunks

# Creating embeddings and storing 
def embeddings_store(resume_chunks):
    vector = embeddings.embed_documents(resume_chunks)
    index = faiss.IndexFlatL2(len(vector[0]))
    vector_store = FAISS(
        embedding_function=embeddings,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        index=index)
    vector_store.add_texts(resume_chunks)
    return vector_store

# Final Model
def create_model(vector_store):
    prompt_template = PromptTemplate(
        template=""" 
You are an AI-powered Applicant Tracking System (ATS) assistant. Your task is to review resumes, provide a detailed evaluation, and offer career advice. 

**Evaluation Criteria**:
- Keywords Match
- Experience and Projects
- Education
- Skills and Certifications
- Formatting
- Industry Fit
- Technical Depth
- Soft Skills
- Potential Career Growth Areas

**Response Format**:
1. **ATS Score**: [Score out of 100]
2. **Strengths**: List 2-3 positive aspects of the resume.
3. **Improvement Tips**: Suggest 2-3 improvements to make the resume better.
4. **Recommended Job Roles**: Based on the resume content, suggest 2-3 roles the individual can apply for.
5. **Suggested Industries**: Recommend 2-3 industries that align with the candidate's profile.
6. **Potential Growth Areas**: Suggest specific skills, tools, or certifications to enhance career opportunities.
7. **Networking Tips**: Provide recommendations to build a stronger professional network.

Here's the resume content: 
{context}

Provide a response in a clear and easy-to-understand style.
""",
        input_variables=["context"]
    )

    retriever = vector_store.as_retriever()
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # Allows user's query
        | prompt_template
        | llm)
    return qa_chain


st.markdown(
    """
    <h1 style="text-align: center; font-size: 40px; color: black; margin-top: 20px;">
        AI based ATS Scorer
    </h1>
    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Upload a PDF or DOC file", type=["pdf", "docx"])
if uploaded_file:
    resume_text = resume_parse(uploaded_file)
    resume_chunks = create_chunks(resume_text)
    vector_store = embeddings_store(resume_chunks)
    qa_chain = create_model(vector_store)
    
    # Take query and generate a response
    query = st.text_input("Enter Query: ")
    if query:
        response = qa_chain.invoke(query)
        st.subheader("ATS Feedback")
        st.write(response)




