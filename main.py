import streamlit as st
from nlp_parser import parse_job_prompt
import apis
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Initialize NLP pipelines (load once)
@st.cache_resource
def load_nlp_pipelines():
    # Zero-shot classifier for job type detection
    classifier = pipeline("zero-shot-classification", 
                        model="typeform/distilbert-base-uncased-mnli")
    # Sentiment analysis for experience level estimation
    sentiment = pipeline("sentiment-analysis", 
                       model="distilbert-base-uncased-finetuned-sst-2-english")
    return classifier, sentiment

classifier, sentiment = load_nlp_pipelines()

st.title("🧠 AI-Powered Job & Candidate Finder Pro")

# Salary prediction model (simplified for demo)
def predict_salary(role, experience, location):
    """Predict salary based on role, experience and location"""
    # Base salaries (in LPA) for different roles in India
    base_salaries = {
        "developer": {"entry": 4, "mid": 8, "senior": 15},
        "data analyst": {"entry": 5, "mid": 10, "senior": 18},
        "full stack": {"entry": 6, "mid": 12, "senior": 20},
        "default": {"entry": 3, "mid": 7, "senior": 12}
    }
    
    # Location multipliers
    location_factors = {
        "bangalore": 1.3,
        "mumbai": 1.25,
        "delhi": 1.2,
        "hyderabad": 1.15,
        "pune": 1.1,
        "default": 1.0
    }
    
    role_key = role.lower() if role.lower() in base_salaries else "default"
    location_key = location.lower() if location.lower() in location_factors else "default"
    
    base = base_salaries[role_key][experience]
    multiplier = location_factors[location_key]
    
    # Add some random variation (+/- 15%)
    variation = 1 + (np.random.rand() * 0.3 - 0.15)
    return round(base * multiplier * variation, 1)

# Enhanced job relevance scoring
def enhanced_relevance_score(job, parsed_data):
    """Improved scoring with NLP features"""
    title = job.get('job_title', '').lower()
    desc = job.get('job_description', '').lower()
    text = title + " " + desc
    
    # Basic keyword matching
    score = 0
    if parsed_data['role'] and parsed_data['role'].lower() in text:
        score += 5
        
    for skill in parsed_data['skills']:
        if skill.lower() in text:
            score += 2
            
    # Location matching
    if parsed_data['location'] and parsed_data['location'].lower() in text:
        score += 3
        
    # Work preference matching
    if parsed_data['work_preference'] and parsed_data['work_preference'].lower() in text:
        score += 2
        
    # Use zero-shot classification to match job type
    candidate_labels = ["technology", "business", "design", "marketing", "finance"]
    result = classifier(text, candidate_labels)
    if parsed_data['role'] and result['labels'][0] in parsed_data['role'].lower():
        score += 3
        
    return score

# AI-powered job search
def ai_job_search(parsed_data):
    """Enhanced job search with AI features"""
    url = "https://jsearch.p.rapidapi.com/search"
    query = f"{parsed_data['role']} in {parsed_data['location']}" if parsed_data['location'] else parsed_data['role']
    
    params = {
        "query": query,
        "num_pages": "2",  # Get more results for better matching
        "page": "1",
        "employment_types": "FULLTIME",
        "remote_jobs_only": "true" if parsed_data['work_preference'] == "Remote" else "false"
    }
    
    headers = {
        "X-RapidAPI-Key": apis.JSEARCH_API,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        jobs = response.json().get("data", [])
        
        # Score and sort jobs by relevance
        scored_jobs = [(job, enhanced_relevance_score(job, parsed_data)) for job in jobs]
        scored_jobs.sort(key=lambda x: x[1], reverse=True)
        
        return [job for job, score in scored_jobs[:10]]  # Return top 10
    else:
        st.error(f"API Error {response.status_code}: {response.text}")
        return []

# Experience level detection
def detect_experience(text):
    """Estimate experience level from text"""
    result = sentiment(text)
    if "senior" in text.lower():
        return "senior"
    elif "mid" in text.lower() or "intermediate" in text.lower():
        return "mid"
    elif result[0]['label'] == 'POSITIVE' and "year" in text.lower():
        return "mid"
    else:
        return "entry"

tabs = st.tabs(["🔍 AI Job Search", "📌 Smart Recruiter", "💰 Salary Predictor"])

with tabs[0]:
    st.subheader("AI-Powered Job Search")
    prompt = st.text_area("Describe your ideal job:", 
                         placeholder="e.g. I want a remote Python developer job in Bangalore with 3 years experience...")
    
    if st.button("Find My Dream Job"):
        if not prompt.strip():
            st.warning("Please describe your job preferences")
        else:
            with st.spinner("Analyzing your preferences with AI..."):
                parsed = parse_job_prompt(prompt)
                st.subheader("🤖 AI Analysis")
                st.json(parsed)
                
                # Detect experience level
                exp_level = detect_experience(prompt)
                parsed['experience'] = exp_level
                st.info(f"Detected Experience Level: {exp_level.title()}")
                
                # Predict salary
                if parsed['role'] and parsed['location']:
                    salary = predict_salary(parsed['role'], exp_level, parsed['location'])
                    st.success(f"💵 Predicted Salary Range: {salary-2} - {salary+2} LPA")
                
                # Find matching jobs
                jobs = ai_job_search(parsed)
                if not jobs:
                    st.warning("No matching jobs found. Try broadening your search.")
                else:
                    st.subheader("🎯 Best Matching Jobs")
                    for idx, job in enumerate(jobs, 1):
                        with st.expander(f"{idx}. {job.get('job_title', 'N/A')}"):
                            st.markdown(f"*Company:* {job.get('employer_name', 'N/A')}")
                            st.markdown(f"*Location:* {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}")
                            st.markdown(f"*Type:* {job.get('job_employment_type', 'N/A')}")
                            
                            # Show salary if available
                            if job.get('job_min_salary') and job.get('job_max_salary'):
                                st.markdown(f"*Salary:* ₹{job['job_min_salary']/1000}k - ₹{job['job_max_salary']/1000}k")
                            else:
                                st.markdown("*Salary:* Not specified")
                            
                            st.markdown(f"[🔗 Apply Here]({job.get('job_apply_link', '#')})")
                            
                            # Show job highlights
                            if job.get('job_highlights'):
                                st.markdown("*Highlights:*")
                                for highlight, items in job['job_highlights'].items():
                                    st.markdown(f"- {highlight}:")
                                    for item in items:
                                        st.markdown(f"  - {item}")

with tabs[1]:
    st.subheader("Smart Recruiter Pro")
    prompt = st.text_area("Describe your ideal candidate:", 
                         placeholder="e.g. We need a senior React developer with Redux experience in Mumbai...")
    
    if st.button("Find Top Candidates"):
        if not prompt.strip():
            st.warning("Please describe your candidate requirements")
        else:
            with st.spinner("Analyzing candidate requirements..."):
                parsed = parse_job_prompt(prompt)
                st.subheader("🤖 AI Analysis")
                st.json(parsed)
                
                # Build search query
                role = parsed.get('role', 'developer')
                skills = " ".join(parsed.get('skills', []))
                location = parsed.get('location', '')
                
                query = f"{role} {skills} site:linkedin.com/in OR site:upwork.com/freelancers"
                if location:
                    query += f" location:{location}"
                
                # Search with SerpAPI
                serp_url = "https://serpapi.com/search.json"
                serp_params = {
                    "q": query,
                    "hl": "en",
                    "api_key": apis.SERP_API,
                    "num": 5  # Get top 5 results
                }
                
                response = requests.get(serp_url, params=serp_params)
                if response.status_code == 200:
                    results = response.json().get("organic_results", [])
                    if not results:
                        st.warning("No candidates found. Try different keywords.")
                    else:
                        st.subheader("👥 Top Matching Candidates")
                        for idx, result in enumerate(results, 1):
                            with st.expander(f"{idx}. {result.get('title', 'N/A')}"):
                                st.write(result.get("snippet", "No description available"))
                                st.markdown(f"[🔗 View Profile]({result.get('link', '#')})")
                                
                                # Extract potential experience
                                snippet = result.get("snippet", "").lower()
                                exp_level = detect_experience(snippet)
                                st.info(f"Estimated Experience: {exp_level.title()}")
                                
                                # Show skills match
                                matched_skills = [s for s in parsed.get('skills', []) if s.lower() in snippet]
                                if matched_skills:
                                    st.success(f"Skills Match: {', '.join(matched_skills)}")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

with tabs[2]:
    st.subheader("💰 AI Salary Predictor")
    st.write("Get salary estimates based on role, experience and location")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        role = st.selectbox("Job Role", 
                           ["Developer", "Data Analyst", "Full Stack", "UX Designer", "Product Manager"])
    with col2:
        experience = st.selectbox("Experience Level", 
                                ["Entry", "Mid", "Senior"])
    with col3:
        location = st.selectbox("Location", 
                              ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Other"])
    
    if st.button("Predict Salary"):
        with st.spinner("Calculating salary prediction..."):
            salary = predict_salary(role, experience.lower(), location)
            st.success(f"💵 Predicted Salary Range for {role} in {location}:")
            st.markdown(f"## ₹{salary-2} - {salary+2} LPA")
            
            # Show comparison
            st.subheader("💰 Salary Comparison")
            base_salary = predict_salary(role, experience.lower(), "Other")
            if location != "Other":
                st.write(f"- {location} salary is {round((salary/base_salary-1)*100)}% higher than average")
            st.write(f"- Senior level typically earns {round(predict_salary(role, 'senior', location)/salary*100)}% more")
            st.write(f"- Entry level typically earns {round(predict_salary(role, 'entry', location)/salary*50)}% less")

# Add some styling
st.markdown("""
<style>
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .stExpander:hover {
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)