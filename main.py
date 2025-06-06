# main.py
import streamlit as st
from nlp_parser import parse_prompt_gpt, extract_with_regex
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

# Azure OpenAI Setup
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)

# External APIs
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

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

def smart_merge_results(gpt_result, regex_result):
    """Intelligently merge GPT and regex parsing results"""
    merged = gpt_result.copy()
    
    # Use regex result if GPT result is null/empty
    for key in ['role', 'location', 'skills', 'work_preference', 'experience']:
        if not merged.get(key) or merged.get(key) in [None, [], ""]:
            if regex_result.get(key) and regex_result[key] not in [None, [], ""]:
                merged[key] = regex_result[key]
        
        # Special handling for skills - merge both results
        if key == 'skills':
            gpt_skills = merged.get('skills', []) or []
            regex_skills = regex_result.get('skills', []) or []
            combined_skills = list(set(gpt_skills + regex_skills))
            merged['skills'] = combined_skills
    
    return merged

def predict_salary(role, experience, location):
    base_salaries = {
        "python developer": {"entry": 5, "mid": 10, "senior": 18},
        "java developer": {"entry": 4.5, "mid": 9, "senior": 16},
        "react developer": {"entry": 5.5, "mid": 11, "senior": 19},
        "full stack developer": {"entry": 6, "mid": 12, "senior": 20},
        "data analyst": {"entry": 5, "mid": 10, "senior": 18},
        "data scientist": {"entry": 7, "mid": 14, "senior": 25},
        "software engineer": {"entry": 4.5, "mid": 9, "senior": 16},
        "frontend developer": {"entry": 4, "mid": 8, "senior": 15},
        "backend developer": {"entry": 5, "mid": 10, "senior": 18},
        "devops engineer": {"entry": 6, "mid": 12, "senior": 22},
        "product manager": {"entry": 8, "mid": 16, "senior": 30},
        "ux designer": {"entry": 4, "mid": 8, "senior": 15},
        "developer": {"entry": 4, "mid": 8, "senior": 15},
        "engineer": {"entry": 4.5, "mid": 9, "senior": 16},
        "analyst": {"entry": 4, "mid": 8, "senior": 14},
        "default": {"entry": 3, "mid": 7, "senior": 12}
    }
    
    location_factors = {
        "bangalore": 1.3, "mumbai": 1.25, "delhi": 1.2, "gurgaon": 1.2,
        "hyderabad": 1.15, "pune": 1.1, "chennai": 1.1, "noida": 1.15,
        "surat": 0.9, "ahmedabad": 0.95, "jaipur": 0.9, "kolkata": 0.95,
        "remote": 1.0, "default": 1.0
    }
    
    # Find matching role (case insensitive, partial match)
    role_key = "default"
    role_lower = role.lower() if role else ""
    
    for salary_role in base_salaries.keys():
        if salary_role in role_lower or any(word in role_lower for word in salary_role.split()):
            role_key = salary_role
            break
    
    location_key = location.lower() if location and location.lower() in location_factors else "default"
    
    base = base_salaries[role_key][experience]
    multiplier = location_factors[location_key]
    variation = 1 + (np.random.rand() * 0.3 - 0.15)
    return round(base * multiplier * variation, 1)

def detect_experience(text):
    try:
        text_lower = text.lower()
        
        # Senior indicators
        if any(word in text_lower for word in ["senior", "lead", "principal", "staff", "architect"]):
            return "senior"
        elif any(pattern in text_lower for pattern in ["5+ years", "5 years", "6+ years", "6 years", "7+ years", "experienced"]):
            return "senior"
        
        # Mid-level indicators
        elif any(word in text_lower for word in ["mid", "intermediate", "associate"]):
            return "mid"
        elif any(pattern in text_lower for pattern in ["2-4 years", "3 years", "4 years", "2+ years", "3+ years"]):
            return "mid"
        
        # Entry level indicators
        elif any(word in text_lower for word in ["entry", "junior", "fresher", "graduate", "trainee"]):
            return "entry"
        elif any(pattern in text_lower for pattern in ["0-2 years", "1 year", "new grad", "fresh"]):
            return "entry"
        
        # Default to entry if no clear indicators
        else:
            return "entry"
            
    except Exception:
        return "entry"

def fetch_jobs(parsed):
    if not JSEARCH_API_KEY:
        st.warning("⚠️ JSEARCH_API_KEY not found in environment variables")
        return []
    
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": JSEARCH_API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    
    # Build search query more intelligently
    query_parts = []
    
    if parsed.get('role'):
        query_parts.append(parsed['role'])
    else:
        query_parts.append("software developer")  # Default fallback
    
    if parsed.get('location') and parsed['location'].lower() != 'remote':
        query_parts.append(f"in {parsed['location']}")
    
    # Add top skills to query
    if parsed.get('skills'):
        query_parts.extend(parsed['skills'][:2])
    
    # Add experience level if specified
    if parsed.get('experience') and parsed['experience'] in ['senior', 'mid']:
        query_parts.insert(0, parsed['experience'])
    
    query = " ".join(query_parts)
    
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
    
    # Build search query
    search_parts = []
    if parsed.get('role'):
        search_parts.append(parsed['role'])
    if parsed.get('skills'):
        search_parts.extend(parsed['skills'][:3])
    
    search_query = f"{' '.join(search_parts)} site:linkedin.com/in/"
    if parsed.get('location') and parsed['location'].lower() != 'remote':
        search_query += f" {parsed['location']}"
    
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

# Main UI
tabs = st.tabs(["🔍 AI Job Search", "📌 Smart Recruiter", "💰 Salary Predictor"])

with tabs[0]:
    st.subheader("AI-Powered Job Search")
    
    # Add example prompts
    with st.expander("💡 Example Prompts - Click to expand"):
        st.markdown("""
        **Job Search Examples:**
        - 'I want a Python developer job in Bangalore with 3 years experience'
        - 'Looking for remote React developer position'
        - 'Senior data analyst role in Mumbai'
        - 'Full stack developer job in Surat with Node.js and MongoDB'
        - 'Entry level Java developer position in Pune'
        - 'DevOps engineer role with AWS and Docker experience'
        """)
    
    prompt = st.text_area(
        "Describe your ideal job:", 
        placeholder="e.g. I want a Python developer job in Bangalore with 3 years experience...",
        height=100
    )

    if st.button("🚀 Find My Dream Job", type="primary"):
        if not prompt.strip():
            st.warning("⚠️ Please describe your job preferences")
        else:
            with st.spinner("🤖 Analyzing your preferences with AI..."):
                # Use both GPT and regex parsing
                gpt_result = parse_prompt_gpt(prompt)
                regex_result = extract_with_regex(prompt)
                
                # Show parsing methods used
                col1, col2 = st.columns(2)
                with col1:
                    st.info("🤖 GPT Analysis Complete")
                with col2:
                    st.info("🔍 Regex Analysis Complete")
                
                # Merge results intelligently
                parsed = smart_merge_results(gpt_result, regex_result)
                
                st.subheader("🎯 Final AI Analysis")
                
                # Display results in a nice format
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Role", parsed.get('role') or "Not detected")
                    st.metric("Location", parsed.get('location') or "Not detected")
                with col2:
                    st.metric("Experience", (parsed.get('experience') or "entry").title())
                    st.metric("Work Style", parsed.get('work_preference') or "Not specified")
                with col3:
                    skills_count = len(parsed.get('skills', []))
                    st.metric("Skills Found", skills_count)
                    if skills_count > 0:
                        st.write("**Skills:** " + ", ".join(parsed['skills']))

                # Show detailed JSON for debugging
                with st.expander("🔧 Detailed Analysis (Debug Info)"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**GPT Result:**")
                        st.json(gpt_result)
                    with col2:
                        st.write("**Regex Result:**")
                        st.json(regex_result)

                # Detect experience level
                exp_level = detect_experience(prompt)
                parsed['experience'] = exp_level

                # Show salary prediction if we have role and location
                if parsed.get('role') and parsed.get('location'):
                    salary = predict_salary(parsed['role'], exp_level, parsed['location'])
                    st.success(f"💰 Predicted Salary Range: **₹{salary-2} - ₹{salary+2} LPA**")
                elif parsed.get('role'):
                    salary = predict_salary(parsed['role'], exp_level, "default")
                    st.info(f"💰 Average Salary Range: **₹{salary-2} - ₹{salary+2} LPA** (location not specified)")
                else:
                    st.warning("⚠️ Cannot predict salary - role not detected")

                # Fetch jobs
                st.subheader("🔍 Job Search Results")
                jobs = fetch_jobs(parsed)
                
                if not jobs:
                    st.error("❌ No matching jobs found")
                    
                    # Provide suggestions
                    st.markdown("### 💡 Suggestions to improve your search:")
                    suggestions = []
                    if not parsed.get('role'):
                        suggestions.append("• Be more specific about the job role (e.g., 'Python Developer', 'Data Analyst')")
                    if not parsed.get('location'):
                        suggestions.append("• Specify a location (e.g., 'in Mumbai', 'remote position')")
                    if not parsed.get('skills'):
                        suggestions.append("• Mention specific skills (e.g., 'with React', 'using Python')")
                    
                    suggestions.extend([
                        "• Try broader terms (e.g., 'developer' instead of 'senior full-stack developer')",
                        "• Check if your API keys are properly configured",
                        "• Try different location names"
                    ])
                    
                    for suggestion in suggestions:
                        st.markdown(suggestion)
                        
                else:
                    st.success(f"🎉 Found **{len(jobs)}** matching jobs!")
                    
                    # Filter and display jobs
                    for idx, job in enumerate(jobs[:10], 1):  # Show top 10 jobs
                        with st.expander(f"**{idx}. {job.get('job_title', 'Job Title Not Available')}**"):
                            
                            # Job details in columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**🏢 Company:** {job.get('employer_name', 'N/A')}")
                                st.markdown(f"**📍 Location:** {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}")
                                st.markdown(f"**💼 Type:** {job.get('job_employment_type', 'N/A')}")
                                
                                # Posted date
                                if job.get('job_posted_at_datetime_utc'):
                                    st.markdown(f"**📅 Posted:** {job['job_posted_at_datetime_utc'][:10]}")
                            
                            with col2:
                                # Salary information
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
                                
                                # Remote/onsite info
                                if job.get('job_is_remote'):
                                    st.markdown("**🏠 Remote:** ✅ Yes")
                                else:
                                    st.markdown("**🏠 Remote:** ❌ No")
                            
                            # Job description
                            if job.get('job_description'):
                                desc = job['job_description']
                                if len(desc) > 300:
                                    desc = desc[:300] + "..."
                                st.markdown(f"**📋 Description:** {desc}")
                            
                            # Apply button
                            if job.get('job_apply_link'):
                                st.markdown(f"[🔗 **Apply Now**]({job['job_apply_link']})")
                            
                            # Job highlights
                            if job.get('job_highlights'):
                                highlights = job['job_highlights']
                                if highlights.get('Qualifications'):
                                    st.markdown("**🎯 Key Qualifications:**")
                                    for qual in highlights['Qualifications'][:3]:
                                        st.markdown(f"• {qual}")

with tabs[1]:
    st.subheader("Smart Recruiter Pro")
    
    with st.expander("💡 Example Prompts for Recruiters"):
        st.markdown("""
        **Candidate Search Examples:**
        - 'We need a senior React developer with Redux experience in Mumbai'
        - 'Looking for Python data scientist with ML experience'
        - 'Hiring full stack developer with Node.js and MongoDB skills'
        - 'Need DevOps engineer with AWS and Kubernetes experience in Bangalore'
        """)
    
    prompt = st.text_area(
        "Describe your ideal candidate:", 
        placeholder="e.g. We need a senior React developer with Redux experience in Mumbai...",
        height=100
    )

    if st.button("🔍 Find Top Candidates", type="primary"):
        if not prompt.strip():
            st.warning("⚠️ Please describe your candidate requirements")
        else:
            with st.spinner("🤖 Using AI to find LinkedIn profiles..."):
                # Parse requirements
                gpt_result = parse_prompt_gpt(prompt)
                regex_result = extract_with_regex(prompt)
                parsed = smart_merge_results(gpt_result, regex_result)
                parsed['query_type'] = 'candidate'
                
                st.subheader("🎯 Candidate Requirements Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Target Role", parsed.get('role') or "Not specified")
                with col2:
                    st.metric("Preferred Location", parsed.get('location') or "Any")
                with col3:
                    skills_count = len(parsed.get('skills', []))
                    st.metric("Required Skills", skills_count)
                
                if parsed.get('skills'):
                    st.write("**Key Skills:** " + ", ".join(parsed['skills']))

                candidates = fetch_candidates(parsed)
                if not candidates:
                    st.warning("❌ No relevant candidates found")
                    
                    st.markdown("### 💡 Tips to find better candidates:")
                    st.markdown("• Try broader skill terms")
                    st.markdown("• Remove location restrictions")
                    st.markdown("• Use more common job titles")
                    st.markdown("• Check if your SERP API key is configured")
                    
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
                            
                            # Extract additional info if available
                            if candidate.get("date"):
                                st.markdown(f"**📅 Last Updated:** {candidate['date']}")

with tabs[2]:
    st.subheader("💰 AI Salary Predictor")
    
    st.markdown("Get accurate salary predictions based on role, experience, and location in India.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        role = st.selectbox(
            "Job Role", 
            ["Python Developer", "Java Developer", "React Developer", "Full Stack Developer", 
             "Data Analyst", "Data Scientist", "Software Engineer", "Frontend Developer",
             "Backend Developer", "DevOps Engineer", "Product Manager", "UX Designer"]
        )
    with col2:
        experience = st.selectbox("Experience Level", ["Entry", "Mid", "Senior"])
    with col3:
        location = st.selectbox(
            "Location", 
            ["Bangalore", "Mumbai", "Delhi", "Gurgaon", "Hyderabad", "Pune", 
             "Chennai", "Noida", "Surat", "Ahmedabad", "Jaipur", "Kolkata", "Remote", "Other"]
        )

    if st.button("💰 Predict Salary", type="primary"):
        with st.spinner("🔮 Calculating salary prediction..."):
            salary = predict_salary(role, experience.lower(), location)
            
            # Main prediction
            st.success(f"💵 **Predicted Salary Range for {role} in {location}:** ₹{salary-2} - ₹{salary+2} LPA")
            
            # Detailed comparison
            st.subheader("📊 Salary Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🌍 Location Comparison")
                base_salary = predict_salary(role, experience.lower(), "Other")
                
                if location != "Other" and location != "Remote":
                    percentage_diff = round((salary/base_salary-1)*100)
                    if percentage_diff > 0:
                        st.write(f"📈 **{location}** salary is **{percentage_diff}%** higher than national average")
                    elif percentage_diff < 0:
                        st.write(f"📉 **{location}** salary is **{abs(percentage_diff)}%** lower than national average")
                    else:
                        st.write(f"📊 **{location}** salary is at national average")
                
                # Show top paying cities for this role
                top_cities = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune"]
                st.markdown("**Top Paying Cities:**")
                for city in top_cities:
                    city_salary = predict_salary(role, experience.lower(), city)
                    st.write(f"• {city}: ₹{city_salary-1}-{city_salary+1} LPA")
            
            with col2:
                st.markdown("### 📈 Experience Level Comparison")
                
                entry_salary = predict_salary(role, 'entry', location)
                mid_salary = predict_salary(role, 'mid', location)
                senior_salary = predict_salary(role, 'senior', location)
                
                st.write(f"👶 **Entry Level:** ₹{entry_salary-1}-{entry_salary+1} LPA")
                st.write(f"👔 **Mid Level:** ₹{mid_salary-1}-{mid_salary+1} LPA")
                st.write(f"🎖️ **Senior Level:** ₹{senior_salary-1}-{senior_salary+1} LPA")
                
                if experience.lower() != 'senior':
                    growth = round((senior_salary/salary-1)*100)
                    st.info(f"💡 You could earn **{growth}%** more at senior level!")

# Enhanced CSS with better styling
st.markdown("""
<style>
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #fafafa;
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
        background-color: #f8f9fa;
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
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)