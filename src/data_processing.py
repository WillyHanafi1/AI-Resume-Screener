import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text: str) -> str:
    """Clean resume/JD text for ML processing."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, emails, phone numbers
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words]
    
    return ' '.join(cleaned_words)

def process_datasets(resume_path: str, jd_path: str, output_dir: str):
    """Load, clean, and save datasets."""
    print("Loading datasets...")
    df_resume = pd.read_csv(resume_path)
    df_jd = pd.read_csv(jd_path, on_bad_lines='skip')
    
    # Process Resumes
    print(f"Processing {len(df_resume)} resumes...")
    # snehaanbhawal/resume-dataset contains 'Resume_str' and 'Category'
    df_resume = df_resume[['ID', 'Category', 'Resume_str']].dropna()
    df_resume.rename(columns={'Resume_str': 'text', 'Category': 'label'}, inplace=True)
    df_resume['cleaned_text'] = df_resume['text'].apply(clean_text)
    
    # Process Job Descriptions
    print(f"Processing {len(df_jd)} job descriptions...")
    # andrewmvd/data-scientist-jobs contains 'Job Title' and 'Job Description'
    df_jd = df_jd[['index', 'Job Title', 'Job Description']].dropna()
    df_jd.rename(columns={'index': 'ID', 'Job Title': 'title', 'Job Description': 'text'}, inplace=True)
    df_jd['cleaned_text'] = df_jd['text'].apply(clean_text)
    
    # Save processed data
    print("Saving processed datasets...")
    os.makedirs(output_dir, exist_ok=True)
    df_resume.to_csv(os.path.join(output_dir, 'resumes_cleaned.csv'), index=False)
    df_jd.to_csv(os.path.join(output_dir, 'jd_cleaned.csv'), index=False)
    print(f"Done! Files saved to {output_dir}")

if __name__ == "__main__":
    RESUME_CSV = 'data/raw/Resume.csv'
    JD_CSV = 'data/raw/DataScientist.csv'
    OUTPUT_DIR = 'data/processed'
    
    process_datasets(RESUME_CSV, JD_CSV, OUTPUT_DIR)
