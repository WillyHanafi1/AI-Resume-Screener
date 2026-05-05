from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.data_processing import extract_skills, extract_years_experience

# Predefined skill taxonomy
TECH_SKILLS = [
    'python', 'java', 'javascript', 'typescript', 'react', 'angular',
    'node.js', 'fastapi', 'django', 'flask', 'docker', 'kubernetes',
    'aws', 'gcp', 'azure', 'postgresql', 'mongodb', 'redis',
    'machine learning', 'deep learning', 'nlp', 'pytorch', 'tensorflow',
    'scikit-learn', 'pandas', 'numpy', 'sql', 'git', 'ci/cd',
]

def build_features(resume_text: str, jd_text: str) -> dict:
    """
    Build feature vector from resume + job description pair.
    Returns dict of engineered features for ML model.
    """
    # 1. TF-IDF Cosine Similarity
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    # If the text is completely empty or just spaces, handle error gracefully
    if not resume_text.strip() or not jd_text.strip():
        tfidf_similarity = 0.0
    else:
        try:
            tfidf_matrix = tfidf.fit_transform([resume_text, jd_text])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except ValueError: # Case where after removing stop words, nothing is left
            tfidf_similarity = 0.0
    
    # 2. Skill Overlap
    resume_skills = set(extract_skills(resume_text, TECH_SKILLS))
    jd_skills = set(extract_skills(jd_text, TECH_SKILLS))
    
    if len(jd_skills) > 0:
        skill_overlap_ratio = len(resume_skills & jd_skills) / len(jd_skills)
    else:
        skill_overlap_ratio = 0.0
    
    skill_count_resume = len(resume_skills)
    skill_count_jd = len(jd_skills)
    common_skills = len(resume_skills & jd_skills)
    
    # 3. Experience
    years_exp = extract_years_experience(resume_text)
    
    # 4. Text Length Features
    resume_word_count = len(resume_text.split())
    jd_word_count = len(jd_text.split())
    
    return {
        'tfidf_similarity': tfidf_similarity,
        'skill_overlap_ratio': skill_overlap_ratio,
        'skill_count_resume': skill_count_resume,
        'skill_count_jd': skill_count_jd,
        'common_skills': common_skills,
        'years_experience': years_exp,
        'resume_word_count': resume_word_count,
        'jd_word_count': jd_word_count,
    }
