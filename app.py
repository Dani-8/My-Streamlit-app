import streamlit as st
import PyPDF2
from PIL import Image
import pytesseract
import pandas as pd
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Load dataset
df = pd.read_csv('healthcare_dataset.csv')
df.columns = df.columns.str.strip()
df['Medical Condition'] = df['Medical Condition'].str.lower().str.strip()
df['Test Results'] = df['Test Results'].str.lower().str.strip()

# Train model
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
pipeline.fit(df['Medical Condition'], df['Test Results'])

def keyword_analyze_text(text, df):
    conditions = df['Medical Condition'].unique()
    found_conditions = [c for c in conditions if c in text.lower()]
    if "glycosylated hemoglobin" in text.lower() or "hba1c" in text.lower() or "hbaic" in text.lower():
        found_conditions.append("diabetes")
    found_results = []
    
    # Directly check for HbA1c value in the text
    if "hbaic 6" in text.lower():
        found_results.append("inconclusive")  # 6% is in 5.7-6.4 range
    
    return found_conditions, found_results

def analyze_text(text, df, model):
    conditions, results = keyword_analyze_text(text, df)
    predicted_result = model.predict([text])[0]
    return conditions, results, predicted_result

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.lower()

def extract_text_from_image(file):
    image = Image.open(file).convert('L')  # Improve OCR with grayscale
    text = pytesseract.image_to_string(image)
    return text.lower()

st.title("Medical Report Checker")
uploaded_files = st.file_uploader("Upload Medical Reports (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing: {uploaded_file.name}")
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
            if "digns" in text or len(text.split()) < 10:
                st.error("OCR Error: Unreadable text. Upload a clearer document.")
                continue
            st.subheader("Extracted Text from PDF")
            st.write(text)
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            text = extract_text_from_image(uploaded_file)
            if len(text.split()) < 10:
                st.error("OCR Error: Unreadable text. Upload a clearer document.")
                continue
            st.subheader("Uploaded Image")
            st.image(uploaded_file, caption="Medical Report Image")
            st.subheader("Extracted Text from Image")
            st.write(text)
        
        conditions, results, predicted_result = analyze_text(text, df, pipeline)
        st.subheader("Analysis")
        if conditions:
            st.write(f"Detected Medical Conditions: {', '.join(conditions)}")
        else:
            st.write("No medical conditions detected.")
        if results:
            st.write(f"Detected Test Results: {', '.join(results)}")
        else:
            st.write("No test results detected.")
        st.write(f"Predicted Test Result: {predicted_result}")
else:
    st.write("Upload medical reports to analyze!")