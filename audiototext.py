
import whisper
import sqlite3
import pandas as pd
import streamlit as st
from transformers import pipeline
import spacy
import os

# Load models
whisper_model = whisper.load_model("base")  # Change to "large" for better accuracy
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
nlp = spacy.load("en_core_web_sm")  # For extracting location and urgency

# Database setup
DB_PATH = "complaints.db"

def init_db():
    """Initialize the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints_db (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint TEXT,
            category TEXT,
            location TEXT,
            urgency TEXT
        )
    """)
    conn.commit()
    conn.close()

# Function to transcribe audio
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Function to classify complaint
def classify_complaint(complaint_text):
    categories = ["Flight Delay", "Baggage Issue", "Immigration", "Security", "Other"]
    prediction = classifier(complaint_text, candidate_labels=categories)
    return prediction["labels"][0]  # Return the highest scoring category

# Function to extract urgency and location
def extract_details(complaint_text):
    doc = nlp(complaint_text)
    location = None
    urgency = "Normal"  # Default urgency

    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            location = ent.text
            break

    urgency_keywords = ["urgent", "immediately", "critical", "high priority", "emergency"]
    if any(word in complaint_text.lower() for word in urgency_keywords):
        urgency = "High"

    return location, urgency

# Function to store complaint in the database
def store_complaint_in_db(complaint_text, category, location, urgency):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO complaints_db (complaint, category, location, urgency) VALUES (?, ?, ?, ?)",
                   (complaint_text, category, location, urgency))
    conn.commit()
    conn.close()

# Streamlit App
def main():
    st.title("📞 Call Complaint Classification System")
    st.write("Upload an audio file of a complaint call to analyze its category and details.")

    uploaded_file = st.file_uploader("Upload an MP3 file", type=["mp3"])

    if uploaded_file is not None:
        file_path = "uploaded_audio.mp3"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(file_path, format="audio/mp3")

        # Process the uploaded audio file
        with st.spinner("Transcribing the call..."):
            complaint_text = transcribe_audio(file_path)
        st.subheader("Transcribed Text")
        st.write(complaint_text)

        # Classification
        category = classify_complaint(complaint_text)
        st.subheader("Complaint Category")
        st.write(f"🔹 {category}")

        # Extract location & urgency
        location, urgency = extract_details(complaint_text)
        st.subheader("Extracted Details")
        st.write(f"📍 Location: {location if location else 'Not detected'}")
        st.write(f"⚠️ Urgency: {urgency}")

        # Store in database
        store_complaint_in_db(complaint_text, category, location, urgency)
        st.success("✅ Complaint logged successfully!")

    # Display complaints
    st.subheader("📊 Complaints Dashboard")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM complaints_db", conn)
    conn.close()

    if not df.empty:
        st.write(df)
        st.bar_chart(df["category"].value_counts())
        st.bar_chart(df["urgency"].value_counts())
    else:
        st.write("No complaints recorded yet.")

if __name__ == "__main__":
    init_db()
    main()
