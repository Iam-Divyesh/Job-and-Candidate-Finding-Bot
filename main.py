# main.py
import streamlit as st
from nlp_parser import parse_prompt_gpt
from dotenv import load_dotenv
import os
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from openai import AzureOpenAI
import json
import re

# Load environment variables
load_dotenv()

# Azure OpenAI Setup for GPT-4.1-mini
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # Should be gpt-4.1-mini

client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

# External APIs
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")

@st.cache_resource
def load_nlp_pipelines():
    try:
        classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        return classifier, sentiment
    except Exception as e:
        st.error(f"Failed to load NLP pipelines: {e}")
        return None, None

classifier, sentiment = load_nlp_pipelines()

st.title("🧠 AI-Powered Job & Candidate Finder Pro")

def validate_location_with_api(location):
    """Validate and enhance location using OpenCage Geocoding API"""
    if not location or not OPENCAGE_API_KEY:
        return location
    
    try:
        url = f"https://api.opencagedata.com/geocode/v1/json"
        params = {
            'q': location,
            'key': OPENCAGE_API_KEY,
            'limit': 1,
            'countrycode': 'in',
            'no_annotations': 1
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                result = data['results'][0]
                formatted_location = result['formatted']
                components = result['components']
                
                city = components.get('city') or components.get('town') or components.get('village')
                state = components.get('state')
                
                if city and state:
                    return f"{city}, {state}"
                elif city:
                    return city
                else:
                    return formatted_location
        
        return location
    except Exception as e:
        st.warning(f"Location validation failed: {e}")
        return location

def predict_salary_dynamically(parsed_data):
    """Dynamic salary prediction based on parsed job data"""
    role = parsed_data.get('role', '').lower()
    experience = parsed_data.get('experience', 'entry')
    location = parsed_data.get('location', '').lower()
    skills = parsed_data.get('skills', [])
    
    # Base salary calculation with skill multipliers
    base_ranges = {
        'entry': {'min': 3, 'max': 8},
        'mid': {'min': 7, 'max': 15},
        'senior': {'min': 15, 'max': 30},
        'lead': {'min': 25, 'max': 45}
    }
    
    # Role-based multipliers (dynamic detection)
    role_multipliers = {
        'data scientist': 1.4, 'machine learning': 1.4, 'ai engineer': 1.4,
        'product manager': 1.3, 'architect': 1.3, 'technical lead': 1.3,
        'full stack': 1.2, 'devops': 1.2, 'cloud': 1.2,
        'frontend': 1.0, 'backend': 1.1, 'mobile': 1.1,
        'qa': 0.9, 'test': 0.9, 'support': 0.8
    }
    
    # Location multipliers (dynamic)
    location_multipliers = {
        'bangalore': 1.3, 'mumbai': 1.25, 'delhi': 1.2, 'gurgaon': 1.2,
        'hyderabad': 1.15, 'pune': 1.1, 'chennai': 1.1, 'noida': 1.15,
        'remote': 1.0, 'anywhere': 1.0
    }
    
    # Skill-based bonus
    high_value_skills = ['react', 'node.js', 'python', 'aws', 'docker', 'kubernetes', 
                        'machine learning', 'data science', 'blockchain', 'ai']
    skill_bonus = sum(0.1 for skill in skills if any(hvs in skill.lower() for hvs in high_value_skills))
    
    # Calculate base salary
    base_range = base_ranges.get(experience, base_ranges['entry'])
    
    # Apply multipliers
    role_mult = 1.0
    for role_key, mult in role_multipliers.items():
        if role_key in role:
            role_mult = mult
            break
    
    location_mult = location_multipliers.get(location, 1.0)
    
    # Final calculation
    min_salary = base_range['min'] * role_mult * location_mult * (1 + skill_bonus)
    max_salary = base_range['max'] * role_mult * location_mult * (1 + skill_bonus)
    
    return {
        'min': round(min_salary, 1),
        'max': round(max_salary, 1),
        'currency': 'INR',
        'period': 'annual'
    }

def fetch_jobs(parsed):
    if not JSEARCH_API_KEY:
        st.warning("⚠️ JSEARCH_API_KEY not found in environment variables")
        return []
    
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": JSEARCH_API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    
    # Build intelligent search query
    query_parts = []
    
    if parsed.get('role'):
        query_parts.append(parsed['role'])
    
    if parsed.get('skills') and len(parsed['skills']) > 0:
        query_parts.extend(parsed['skills'][:2])
    
    if parsed.get('location') and parsed['location'].lower() not in ['remote', 'anywhere']:
        validated_location = validate_location_with_api(parsed['location'])
        query_parts.append(f"in {validated_location}")
    
    query = " ".join(query_parts) if query_parts else "software developer"
    
    querystring = {
        "query": query,
        "page": "1",
        "num_pages": "1",
        "employment_types": "FULLTIME,PARTTIME,CONTRACTOR"
    }

    try:
        st.info(f"🔍 Searching for: **{query}**")
        response = requests.get(url, headers=headers, params=querystring, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and data['data']:
            st.success(f"✅ Found {len(data['data'])} jobs")
            return data['data']
        else:
            st.warning("⚠️ No jobs found in API response")
            return []
            
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Please try again.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Network error: {e}")
        return []
    except Exception as e:
        st.error(f"❌ Failed to fetch jobs: {e}")
        return []

def fetch_candidates(parsed):
    if not SERP_API_KEY:
        st.warning("⚠️ SERP_API_KEY not found in environment variables")
        return []
    
    search_parts = []
    if parsed.get('role'):
        search_parts.append(parsed['role'])
    if parsed.get('skills') and len(parsed['skills']) > 0:
        search_parts.extend(parsed['skills'][:3])
    
    search_query = f"{' '.join(search_parts)} site:linkedin.com/in/"
    if parsed.get('location') and parsed['location'].lower() not in ['remote', 'anywhere']:
        validated_location = validate_location_with_api(parsed['location'])
        search_query += f" {validated_location}"
    
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": search_query,
        "api_key": SERP_API_KEY,
        "hl": "en",
        "gl": "in",
        "num": 10
    }

    try:
        st.info(f"🔍 Searching for: **{search_query}**")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        results = response.json()
        candidates = results.get("organic_results", [])
        
        if candidates:
            st.success(f"✅ Found {len(candidates)} potential candidates")
        else:
            st.warning("⚠️ No candidates found")
            
        return candidates
        
    except Exception as e:
        st.error(f"❌ Failed to fetch candidates: {e}")
        return []

def display_parsed_results(parsed, user_type="jobseeker"):
    """Display parsed results in a structured format"""
    st.subheader("🎯 AI Analysis Results")
    
    if user_type == "jobseeker":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Role", parsed.get('role') or "Not detected", 
                     help="Job role/position you're looking for")
            st.metric("Experience Level", (parsed.get('experience') or "entry").title(),
                     help="Your experience level")
        with col2:
            st.metric("Location", parsed.get('location') or "Not specified",
                     help="Preferred work location")
            st.metric("Work Preference", parsed.get('work_preference') or "Not specified",
                     help="Remote/On-site/Hybrid preference")
        with col3:
            skills_count = len(parsed.get('skills', []))
            st.metric("Skills Found", skills_count,
                     help="Technical skills identified")
            st.metric("Job Type", parsed.get('job_type') or "Not specified",
                     help="Full-time/Part-time/Contract")
        
        # Additional jobseeker details
        if parsed.get('salary'):
            st.info(f"💰 Expected Salary: {parsed['salary']}")
        if parsed.get('industry'):
            st.info(f"🏭 Target Industry: {parsed['industry']}")
        if parsed.get('platform'):
            st.info(f"📱 Platform Preference: {parsed['platform']}")
    
    else:  # recruiter
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Target Role", parsed.get('role') or "Not specified")
            st.metric("Required Experience", parsed.get('experience') or "Not specified")
        with col2:
            st.metric("Location", parsed.get('location') or "Any")
            skills_count = len(parsed.get('skills', []))
            st.metric("Required Skills", skills_count)
    
    # Skills display
    if parsed.get('skills'):
        st.write("**Skills:** " + ", ".join(parsed['skills']))
    
    # Detailed analysis
    with st.expander("🔧 Detailed AI Analysis"):
        st.json(parsed)

# Main UI
tabs = st.tabs(["🔍 AI Job Search", "📌 Smart Recruiter", "💰 Salary Predictor"])

with tabs[0]:
    st.subheader("AI-Powered Job Search")
    
    with st.expander("💡 Example Prompts - Click to expand"):
        st.markdown("""
        **Job Search Examples:**
        - 'I want a Python developer job in Bangalore with 3 years experience, full-time, 8-12 LPA'
        - 'Looking for remote React developer position with Redux skills, hybrid work'
        - 'Senior data analyst role in Mumbai with SQL and Power BI, fintech industry'
        - 'Full stack developer job in Surat with Node.js and MongoDB, startup platform'
        - 'Entry level Java developer position in Pune, part-time preferred'
        - 'DevOps engineer role with AWS and Docker experience in Hyderabad, 15+ LPA'
        """)
    
    prompt = st.text_area(
        "Describe your ideal job:", 
        placeholder="e.g. I want a Python developer job in Bangalore with 3 years experience, Django skills, full-time, 8-12 LPA salary...",
        height=100
    )

    if st.button("🚀 Find My Dream Job", type="primary"):
        if not prompt.strip():
            st.warning("⚠️ Please describe your job preferences")
        else:
            with st.spinner("🤖 Analyzing your preferences with GPT-4.1-mini..."):
                parsed = parse_prompt_gpt(prompt, query_type="jobseeker")
                
                display_parsed_results(parsed, "jobseeker")
                
                # Dynamic salary prediction
                if parsed.get('role'):
                    salary_prediction = predict_salary_dynamically(parsed)
                    location_text = f" in {parsed['location']}" if parsed.get('location') else ""
                    st.success(f"💰 Predicted Salary Range: **₹{salary_prediction['min']} - ₹{salary_prediction['max']} LPA**{location_text}")

                # Fetch and display jobs
                st.subheader("🔍 Job Search Results")
                jobs = fetch_jobs(parsed)
                
                if not jobs:
                    st.error("❌ No matching jobs found")
                    st.markdown("### 💡 Suggestions to improve your search:")
                    suggestions = [
                        "• Be more specific about the job role",
                        "• Try alternative skill names or technologies",
                        "• Consider expanding location preferences",
                        "• Use more common industry terms"
                    ]
                    for suggestion in suggestions:
                        st.markdown(suggestion)
                else:
                    st.success(f"🎉 Found **{len(jobs)}** matching jobs!")
                    
                    for idx, job in enumerate(jobs[:10], 1):
                        with st.expander(f"**{idx}. {job.get('job_title', 'Job Title Not Available')}**"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**🏢 Company:** {job.get('employer_name', 'N/A')}")
                                st.markdown(f"**📍 Location:** {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}")
                                st.markdown(f"**💼 Type:** {job.get('job_employment_type', 'N/A')}")
                                if job.get('job_posted_at_datetime_utc'):
                                    st.markdown(f"**📅 Posted:** {job['job_posted_at_datetime_utc'][:10]}")
                            
                            with col2:
                                if job.get('job_min_salary') and job.get('job_max_salary'):
                                    min_sal = job['job_min_salary']
                                    max_sal = job['job_max_salary']
                                    currency = job.get('job_salary_currency', 'USD')
                                    if currency == 'USD':
                                        st.markdown(f"**💵 Salary:** ${min_sal:,} - ${max_sal:,}")
                                    else:
                                        st.markdown(f"**💵 Salary:** {currency} {min_sal:,} - {max_sal:,}")
                                else:
                                    st.markdown("**💵 Salary:** Not specified")
                                
                                if job.get('job_is_remote'):
                                    st.markdown("**🏠 Remote:** ✅ Yes")
                                else:
                                    st.markdown("**🏠 Remote:** ❌ No")
                            
                            if job.get('job_description'):
                                desc = job['job_description']
                                if len(desc) > 300:
                                    desc = desc[:300] + "..."
                                st.markdown(f"**📋 Description:** {desc}")
                            
                            if job.get('job_apply_link'):
                                st.markdown(f"[🔗 **Apply Now**]({job['job_apply_link']})")

with tabs[1]:
    st.subheader("Smart Recruiter Pro")
    
    with st.expander("💡 Example Prompts for Recruiters"):
        st.markdown("""
        **Candidate Search Examples:**
        - 'We need a senior React developer with Redux experience in Mumbai, 5+ years'
        - 'Looking for Python data scientist with ML and TensorFlow experience, remote OK'
        - 'Hiring full stack developer with Node.js and MongoDB skills in Bangalore'
        - 'Need DevOps engineer with AWS and Kubernetes experience, senior level'
        - 'Searching for UI/UX designer with Figma skills in Delhi, 3-5 years experience'
        """)
    
    prompt = st.text_area(
        "Describe your ideal candidate:", 
        placeholder="e.g. We need a senior React developer with Redux experience in Mumbai, 5+ years experience...",
        height=100
    )

    if st.button("🔍 Find Top Candidates", type="primary"):
        if not prompt.strip():
            st.warning("⚠️ Please describe your candidate requirements")
        else:
            with st.spinner("🤖 Using GPT-4.1-mini to find LinkedIn profiles..."):
                parsed = parse_prompt_gpt(prompt, query_type="recruiter")
                
                display_parsed_results(parsed, "recruiter")

                candidates = fetch_candidates(parsed)
                if not candidates:
                    st.warning("❌ No relevant candidates found")
                    st.markdown("### 💡 Tips to find better candidates:")
                    st.markdown("• Try broader skill terms")
                    st.markdown("• Remove location restrictions for wider search")
                    st.markdown("• Use more common job titles")
                else:
                    st.subheader(f"👥 Found {len(candidates)} Potential Candidates")
                    
                    for idx, candidate in enumerate(candidates[:10], 1):
                        title = candidate.get("title", "Professional")
                        link = candidate.get("link", "#")
                        snippet = candidate.get("snippet", "")
                        
                        with st.expander(f"**{idx}. {title}**"):
                            if snippet:
                                st.markdown(f"**📋 Profile Preview:** {snippet}")
                            st.markdown(f"[🔗 **View LinkedIn Profile**]({link})")
                            if candidate.get("date"):
                                st.markdown(f"**📅 Last Updated:** {candidate['date']}")

with tabs[2]:
    st.subheader("💰 AI Salary Predictor")
    st.markdown("Get intelligent salary predictions using AI analysis")
    
    salary_prompt = st.text_area(
        "Describe the role for salary prediction:",
        placeholder="e.g. Senior Python developer with Django and AWS skills in Bangalore with 5 years experience...",
        height=80
    )
    
    if st.button("💰 Predict Salary", type="primary"):
        if not salary_prompt.strip():
            st.warning("⚠️ Please describe the role")
        else:
            with st.spinner("🔮 Calculating intelligent salary prediction..."):
                parsed = parse_prompt_gpt(salary_prompt, query_type="salary")
                salary_prediction = predict_salary_dynamically(parsed)
                
                st.success(f"💵 **Predicted Salary Range:** ₹{salary_prediction['min']} - ₹{salary_prediction['max']} LPA")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📊 Analysis Factors")
                    st.write(f"**Role:** {parsed.get('role', 'Not specified')}")
                    st.write(f"**Experience:** {parsed.get('experience', 'entry').title()}")
                    st.write(f"**Location:** {parsed.get('location', 'India average')}")
                    if parsed.get('skills'):
                        st.write(f"**Skills:** {', '.join(parsed['skills'])}")
                
                with col2:
                    st.markdown("### 💡 Factors Affecting Salary")
                    st.write("• High-demand skills increase salary")
                    st.write("• Location significantly impacts compensation")
                    st.write("• Experience level is a major factor")
                    st.write("• Industry and company size matter")

# Enhanced CSS
st.markdown("""
<style>
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .stExpander:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: #1f77b4;
        transition: all 0.3s ease;
    }
    .stAlert > div {
        padding: 1rem;
        border-radius: 8px;
    }
    .stMetric {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)