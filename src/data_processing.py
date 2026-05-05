import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text: str) -> str:
    """Clean resume/JD text for ML processing."""
    # Remove URLs, emails, phone numbers
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    # Remove special characters, keep letters and numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Lowercase and remove extra whitespace
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text

def extract_skills(text: str, skill_list: list) -> list:
    """Extract matching skills from text against a predefined skill list."""
    text_lower = text.lower()
    found_skills = []
    for skill in skill_list:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    return found_skills

def extract_years_experience(text: str) -> int:
    """Extract years of experience from resume text using regex."""
    patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s+)?experience',
        r'experience\s*:\s*(\d+)\+?\s*years?',
        r'(\d+)\+?\s*years?\s*in\s+',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return 0
