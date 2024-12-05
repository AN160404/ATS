from dotenv import load_dotenv

load_dotenv()
import streamlit as st
import os
from PIL import Image
import pdf2image
import google.generativeai as genai
import io
import base64

genai.configure(api_key=os.getenv("Gemini"))

def get_response(input,pdf_content,prompt):
    model=genai.GenerativeModel('gemini-pro-vision')
    response=model.generate_content([input,pdf_content[0],prompt])
    return response.text

def input_pdf(uploaded_file):
    if uploaded_file is not None:
        images=pdf2image.convert_from_bytes(uploaded_file.read())
        first_page=images[0]
        img_byte_arr=io.BytesIO()
        first_page.save(img_byte_arr,format='JPEG')
        img_byte_arr=img_byte_arr.getvalue()

        pdf_parts=[
            {
                "mime_type": "image/jpeg",
                "data":base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")
    

st.set_page_config(page_title="ATS Review")
st.header("ATS Tracking System")
imput_text=st.text_area("Job Description: ",key="input")
uploaded_file=st.file_uploader("Upload your Resume(PDF)",type["pdf"])

if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

