import streamlit as st
import joblib
import os
import sys
import pdfplumber
import docx

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pytorch_pipeline import SemanticMatcher
from src.feature_engineering import build_features
from src.data_processing import clean_text

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🎯 AI Resume Screening & Matching System")

# Load models
@st.cache_resource
def load_models():
    # If a model doesn't exist yet, we handle it gracefully
    if os.path.exists('models/xgb_model.pkl') and os.path.exists('models/tfidf.pkl') and os.path.exists('models/encoder.pkl'):
        xgb_model = joblib.load('models/xgb_model.pkl')
        tfidf = joblib.load('models/tfidf.pkl')
        encoder = joblib.load('models/encoder.pkl')
    else:
        xgb_model, tfidf, encoder = None, None, None
    
    semantic_matcher = SemanticMatcher()
    return xgb_model, tfidf, encoder, semantic_matcher

xgb_model, tfidf, encoder, semantic_matcher = load_models()

# Sidebar
st.sidebar.title("Configuration")
if not xgb_model:
    st.sidebar.warning("ML models not found in models/ directory. Run 06_create_model.ipynb first. For now, only Semantic Matching will be available.")

def read_file(file):
    if file.name.endswith('.pdf'):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode('utf-8', errors='ignore')

# Main UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Job Description")
    jd_text = st.text_area("Paste job description here", height=300)

with col2:
    st.subheader("📋 Resume(s)")
    uploaded_files = st.file_uploader("Upload resume(s) (.txt, .pdf, .docx)", accept_multiple_files=True, type=['txt', 'pdf', 'docx'])

if st.button("🔍 Analyze & Rank") and jd_text and uploaded_files:
    results = []
    
    # Show loading spinner
    with st.spinner('Processing and evaluating candidates...'):
        for file in uploaded_files:
            resume_text = read_file(file)
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(jd_text)
            
            # Extract basic features for display
            features = build_features(resume_clean, jd_clean)
            
            # Semantic score (Match between JD and Resume)
            semantic_score = semantic_matcher.compute_similarity(resume_clean, jd_clean) * 100
            
            # Predict Resume Category using XGBoost
            if xgb_model is not None and tfidf is not None and encoder is not None:
                resume_tfidf = tfidf.transform([resume_clean])
                pred_idx = xgb_model.predict(resume_tfidf)[0]
                pred_category = encoder.inverse_transform([pred_idx])[0]
                confidence = max(xgb_model.predict_proba(resume_tfidf)[0]) * 100
            else:
                pred_category = "Unknown (Model missing)"
                confidence = 0.0
                
            results.append({
                'file': file.name,
                'category': pred_category,
                'confidence': confidence,
                'match_score': semantic_score,
                'skills': features,
            })
    
    # Display ranked results based on JD Match Score
    results.sort(key=lambda x: x['match_score'], reverse=True)
    
    st.subheader("🏆 Candidate Rankings")
    for i, r in enumerate(results, 1):
        with st.expander(f"#{i} — {r['file']} (Match Score: {r['match_score']:.1f}/100)"):
            col_a, col_b = st.columns(2)
            col_a.metric("Predicted Profession", f"{r['category']}")
            col_b.metric("AI Confidence", f"{r['confidence']:.1f}%")
            
            st.markdown("**Engineered Features:**")
            st.json(r['skills'])
