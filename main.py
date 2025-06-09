# main.py
import streamlit as st
from nlp_parser import parse_prompt_gpt, process_query
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

def enhanced_parse_prompt_gpt(user_input):
    """Enhanced GPT parsing with better prompts and field extraction"""
    if not client:
        return fallback_parser(user_input)
    
    try:
        # Enhanced system prompt for better field extraction
        system_prompt = """You are an expert job search assistant that extracts structured information from user queries. 
        
        Extract the following fields from user input and return ONLY a valid JSON object:

        Fields to extract:
        - role: Job title/position (e.g., "Software Engineer", "Data Scientist")
        - skills: Array of technical skills mentioned (e.g., ["Python", "React", "AWS"])
        - experience: Experience level - MUST be one of: "entry", "mid", "senior", "lead"
        - location: Location preference (city, state, country, or "remote")
        - work_preference: Work mode - one of: "remote", "onsite", "hybrid", null
        - salary: Salary expectation if mentioned (number in lakhs for India)
        - job_type: Employment type - one of: "full-time", "part-time", "contract", "internship", null
        - industry: Industry/domain if specified (e.g., "fintech", "healthcare", "e-commerce")
        - platform: Specific platform mentioned (e.g., "linkedin", "naukri", "indeed")
        - error: null (only set if parsing fails)

        Experience level mapping:
        - "entry": 0-2 years, fresher, graduate, junior
        - "mid": 3-5 years, intermediate, mid-level
        - "senior": 6-10 years, senior, experienced
        - "lead": 10+ years, lead, manager, architect, principal

        Return ONLY valid JSON. Do not include explanations or additional text."""

        user_prompt = f"""Extract job search information from this query: "{user_input}"

        Examples of good extraction:
        
        Query: "I'm looking for a senior Python developer role in Bangalore with 7 years experience"
        Output: {{"role": "Python Developer", "skills": ["Python"], "experience": "senior", "location": "Bangalore", "work_preference": null, "salary": null, "job_type": null, "industry": null, "platform": null, "error": null}}

        Query: "Find me remote React jobs, part-time, 15 LPA salary"
        Output: {{"role": "React Developer", "skills": ["React"], "experience": "mid", "location": "remote", "work_preference": "remote", "salary": 15, "job_type": "part-time", "industry": null, "platform": null, "error": null}}

        Query: "Fresh graduate looking for full-time software engineer positions in fintech companies"
        Output: {{"role": "Software Engineer", "skills": [], "experience": "entry", "location": null, "work_preference": null, "salary": null, "job_type": "full-time", "industry": "fintech", "platform": null, "error": null}}

        Now extract from: "{user_input}" """

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean and parse JSON
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
        
        parsed = json.loads(content)
        
        # Validate and clean the parsed data
        validated = validate_parsed_data(parsed)
        validated['parsing_status'] = 'success'
        
        return validated
        
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error: {e}")
        return fallback_parser(user_input)
    except Exception as e:
        st.warning(f"GPT parsing error: {e}")
        return fallback_parser(user_input)

def validate_parsed_data(parsed):
    """Validate and clean parsed data"""
    # Set defaults for missing fields
    defaults = {
        "role": None,
        "skills": [],
        "experience": "entry",
        "location": None,
        "work_preference": None,
        "salary": None,
        "job_type": None,
        "industry": None,
        "platform": None,
        "error": None
    }
    
    # Merge with defaults
    for key, default_value in defaults.items():
        if key not in parsed or parsed[key] == "":
            parsed[key] = default_value
    
    # Validate experience level
    valid_experience = ["entry", "mid", "senior", "lead"]
    if parsed["experience"] not in valid_experience:
        parsed["experience"] = "entry"
    
    # Validate work preference
    valid_work_prefs = ["remote", "onsite", "hybrid", None]
    if parsed["work_preference"] not in valid_work_prefs:
        parsed["work_preference"] = None
    
    # Validate job type
    valid_job_types = ["full-time", "part-time", "contract", "internship", None]
    if parsed["job_type"] not in valid_job_types:
        parsed["job_type"] = None
    
    # Ensure skills is a list
    if not isinstance(parsed["skills"], list):
        parsed["skills"] = []
    
    # Clean skills (remove empty strings)
    parsed["skills"] = [skill.strip() for skill in parsed["skills"] if skill and skill.strip()]
    
    return parsed

def fallback_parser(user_input):
    """Enhanced fallback parser using regex and keyword matching"""
    result = {
        "role": None,
        "skills": [],
        "experience": "entry",
        "location": None,
        "work_preference": None,
        "salary": None,
        "job_type": None,
        "industry": None,
        "platform": None,
        "error": "AI parsing unavailable - using fallback response",
        "parsing_status": "fallback_used"
    }
    
    text = user_input.lower()
    
    # Extract role
    role_patterns = [
        r'looking for\s+(?:a\s+)?(.+?)\s+(?:role|position|job)',
        r'(?:want|need)\s+(?:a\s+)?(.+?)\s+(?:role|position|job)',
        r'searching for\s+(.+?)\s+(?:opportunities|positions)',
        r'find\s+(?:me\s+)?(.+?)\s+jobs?',
        r'apply\s+for\s+(.+?)\s+(?:role|position)'
    ]
    
    for pattern in role_patterns:
        match = re.search(pattern, text)
        if match:
            role = match.group(1).strip()
            # Clean common words
            role = re.sub(r'\b(the|a|an)\b', '', role).strip()
            if role and len(role) > 2:
                result["role"] = role.title()
                break
    
    # Extract experience
    exp_patterns = {
        "entry": r'\b(?:fresh|fresher|graduate|entry|junior|0-2|new)\b',
        "mid": r'\b(?:3-5|mid|intermediate|experienced)\b',
        "senior": r'\b(?:senior|6-10|7-8|experienced)\b',
        "lead": r'\b(?:lead|manager|architect|principal|10\+|senior)\b'
    }
    
    # Check for years of experience
    years_match = re.search(r'(\d+)\s*(?:\+)?\s*years?\s+(?:of\s+)?experience', text)
    if years_match:
        years = int(years_match.group(1))
        if years <= 2:
            result["experience"] = "entry"
        elif years <= 5:
            result["experience"] = "mid"
        elif years <= 10:
            result["experience"] = "senior"
        else:
            result["experience"] = "lead"
    else:
        for exp_level, pattern in exp_patterns.items():
            if re.search(pattern, text):
                result["experience"] = exp_level
                break
    
    # Extract location
    location_patterns = [
        r'in\s+([a-zA-Z\s,]+?)(?:\s|$|,)',
        r'at\s+([a-zA-Z\s,]+?)(?:\s|$|,)',
        r'based\s+in\s+([a-zA-Z\s,]+?)(?:\s|$|,)',
        r'location:?\s*([a-zA-Z\s,]+?)(?:\s|$|,)'
    ]
    
    if 'remote' in text:
        result["location"] = "remote"
        result["work_preference"] = "remote"
    else:
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                location = match.group(1).strip().rstrip(',')
                if len(location) > 2 and location not in ['the', 'a', 'an']:
                    result["location"] = location.title()
                    break
    
    # Extract work preference
    if 'remote' in text:
        result["work_preference"] = "remote"
    elif 'hybrid' in text:
        result["work_preference"] = "hybrid"
    elif 'onsite' in text or 'on-site' in text or 'office' in text:
        result["work_preference"] = "onsite"
    
    # Extract job type
    if 'part-time' in text or 'part time' in text:
        result["job_type"] = "part-time"
    elif 'contract' in text or 'freelance' in text:
        result["job_type"] = "contract"
    elif 'internship' in text or 'intern' in text:
        result["job_type"] = "internship"
    elif 'full-time' in text or 'full time' in text:
        result["job_type"] = "full-time"
    
    # Extract salary
    salary_patterns = [
        r'(\d+)\s*(?:lpa|lakhs?|l)',
        r'(\d+)\s*lakhs?\s*per\s*annum',
        r'salary\s*:?\s*(\d+)',
        r'(\d+)\s*(?:k|thousand)\s*per\s*month'
    ]
    
    for pattern in salary_patterns:
        match = re.search(pattern, text)
        if match:
            salary = float(match.group(1))
            if 'k' in text or 'thousand' in text:
                salary = salary * 12 / 100000  # Convert monthly thousands to annual lakhs
            result["salary"] = salary
            break
    
    # Extract skills (basic skill detection)
    common_skills = [
        'python', 'java', 'javascript', 'react', 'node.js', 'angular', 'vue',
        'php', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift', 'kotlin',
        'html', 'css', 'sql', 'mongodb', 'mysql', 'postgresql',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
        'git', 'linux', 'django', 'flask', 'spring', 'express',
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'sklearn'
    ]
    
    found_skills = []
    for skill in common_skills:
        if skill.lower() in text:
            found_skills.append(skill.title())
    
    result["skills"] = found_skills
    
    # Extract industry
    industries = {
        'fintech': r'\b(?:fintech|financial|banking|finance)\b',
        'healthcare': r'\b(?:healthcare|medical|health|pharma)\b',
        'e-commerce': r'\b(?:ecommerce|e-commerce|retail|shopping)\b',
        'gaming': r'\b(?:gaming|game|games)\b',
        'education': r'\b(?:education|edtech|learning)\b',
        'travel': r'\b(?:travel|tourism|booking)\b'
    }
    
    for industry, pattern in industries.items():
        if re.search(pattern, text):
            result["industry"] = industry
            break
    
    # Extract platform
    platforms = {
        'linkedin': r'\blinkedin\b',
        'naukri': r'\bnaukri\b',
        'indeed': r'\bindeed\b',
        'glassdoor': r'\bglassdoor\b'
    }
    
    for platform, pattern in platforms.items():
        if re.search(pattern, text):
            result["platform"] = platform
            break
    
    return result

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
    """Enhanced dynamic salary prediction based on parsed job data"""
    role = parsed_data.get('role', '').lower()
    experience = parsed_data.get('experience', 'entry')
    location = parsed_data.get('location', '').lower()
    skills = parsed_data.get('skills', [])
    
    # Enhanced base salary calculation
    base_ranges = {
        'entry': {'min': 3, 'max': 8},
        'mid': {'min': 7, 'max': 15},
        'senior': {'min': 15, 'max': 30},
        'lead': {'min': 25, 'max': 45}
    }
    
    # Enhanced role-based multipliers with more categories
    role_multipliers = {
        # High-demand roles
        'data scientist': 1.5, 'machine learning': 1.5, 'ai engineer': 1.4,
        'blockchain': 1.4, 'cybersecurity': 1.3, 'cloud architect': 1.4,
        
        # Leadership roles
        'product manager': 1.3, 'technical lead': 1.3, 'architect': 1.3,
        'engineering manager': 1.4, 'team lead': 1.2,
        
        # Development roles
        'full stack': 1.2, 'fullstack': 1.2, 'devops': 1.3, 'cloud': 1.2,
        'frontend': 1.0, 'backend': 1.1, 'mobile': 1.1, 'ios': 1.2, 'android': 1.2,
        
        # Specialized roles
        'qa': 0.9, 'test': 0.9, 'support': 0.8, 'intern': 0.6,
        'designer': 1.0, 'ui/ux': 1.1, 'business analyst': 0.95
    }
    
    # Enhanced location multipliers
    location_multipliers = {
        # Tier 1 cities
        'bangalore': 1.3, 'bengaluru': 1.3, 'mumbai': 1.25, 'delhi': 1.2, 
        'gurgaon': 1.2, 'gurugram': 1.2, 'noida': 1.15,
        
        # Tier 2 cities
        'hyderabad': 1.15, 'pune': 1.1, 'chennai': 1.1, 'kolkata': 1.0,
        'ahmedabad': 1.0, 'surat': 0.9, 'jaipur': 0.9, 'lucknow': 0.9,
        
        # Remote work
        'remote': 1.05, 'anywhere': 1.0, 'work from home': 1.05
    }
    
    # Enhanced skill-based bonus calculation
    skill_categories = {
        'high_demand': ['react', 'node.js', 'python', 'aws', 'docker', 'kubernetes', 
                       'machine learning', 'data science', 'blockchain', 'ai', 'tensorflow',
                       'pytorch', 'microservices', 'system design'],
        'cloud_skills': ['aws', 'azure', 'gcp', 'kubernetes', 'docker', 'terraform'],
        'modern_frontend': ['react', 'vue', 'angular', 'typescript', 'next.js'],
        'backend_skills': ['node.js', 'django', 'spring', 'express', 'fastapi'],
        'data_skills': ['python', 'r', 'sql', 'pandas', 'numpy', 'spark'],
        'mobile_skills': ['react native', 'flutter', 'swift', 'kotlin']
    }
    
    skill_bonus = 0
    skills_lower = [skill.lower() for skill in skills]
    
    for category, category_skills in skill_categories.items():
        category_bonus = sum(0.1 for skill in skills_lower 
                            if any(cs in skill for cs in category_skills))
        if category == 'high_demand':
            skill_bonus += category_bonus * 1.5  # Higher weight for high-demand skills
        else:
            skill_bonus += category_bonus
    
    # Calculate base salary
    base_range = base_ranges.get(experience, base_ranges['entry'])
    
    # Apply role multiplier
    role_mult = 1.0
    for role_key, mult in role_multipliers.items():
        if role_key in role:
            role_mult = mult
            break
    
    # Apply location multiplier
    location_mult = 1.0
    for loc_key, mult in location_multipliers.items():
        if loc_key in location:
            location_mult = mult
            break
    
    # Cap skill bonus to prevent unrealistic salaries
    skill_bonus = min(skill_bonus, 1.0)
    
    # Final calculation with bounds checking
    min_salary = base_range['min'] * role_mult * location_mult * (1 + skill_bonus)
    max_salary = base_range['max'] * role_mult * location_mult * (1 + skill_bonus)
    
    # Ensure minimum salary makes sense
    min_salary = max(min_salary, 3.0)
    max_salary = max(max_salary, min_salary + 2.0)
    
    return {
        'min': round(min_salary, 1),
        'max': round(max_salary, 1),
        'currency': 'INR',
        'period': 'annual',
        'factors': {
            'role_multiplier': role_mult,
            'location_multiplier': location_mult,
            'skill_bonus': round(skill_bonus, 2)
        }
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
        # Add top 2 most relevant skills
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
    """Enhanced display of parsed results with better formatting"""
    st.subheader("🎯 AI Analysis Results")
    
    # Show parsing status
    if parsed.get('parsing_status') == 'fallback_used':
        st.warning("⚠️ AI parsing failed - using fallback analysis")
    elif parsed.get('parsing_status') == 'success':
        st.success("✅ Successfully analyzed your query using AI")
    
    if user_type == "jobseeker":
        col1, col2, col3 = st.columns(3)
        with col1:
            role_display = parsed.get('role') or "❌ Not detected"
            st.metric("🎯 Target Role", role_display, 
                     help="Job role/position you're looking for")
            
            exp_display = (parsed.get('experience') or "entry").title()
            exp_help = {
                "Entry": "0-2 years experience",
                "Mid": "3-5 years experience", 
                "Senior": "6+ years experience",
                "Lead": "10+ years experience"
            }
            st.metric("📈 Experience Level", exp_display,
                     help=exp_help.get(exp_display, "Your experience level"))
        
        with col2:
            location_display = parsed.get('location') or "❌ Not specified"
            st.metric("📍 Location", location_display,
                     help="Preferred work location")
            
            work_pref = parsed.get('work_preference') or "❌ Not specified"
            st.metric("🏠 Work Mode", work_pref.title() if work_pref != "❌ Not specified" else work_pref,
                     help="Remote/On-site/Hybrid preference")
        
        with col3:
            skills_count = len(parsed.get('skills', []))
            skills_display = f"{skills_count} skills" if skills_count > 0 else "❌ None found"
            st.metric("🛠️ Skills Found", skills_display,
                     help="Technical skills identified from your query")
            
            job_type = parsed.get('job_type') or "❌ Not specified"
            st.metric("💼 Job Type", job_type.title() if job_type != "❌ Not specified" else job_type,
                     help="Full-time/Part-time/Contract preference")
        
        # Additional information in expandable sections
        if parsed.get('salary'):
            st.info(f"💰 **Salary Expectation:** ₹{parsed['salary']} LPA")
        
        if parsed.get('industry'):
            st.info(f"🏢 **Industry:** {parsed['industry'].title()}")
        
        if parsed.get('platform'):
            st.info(f"🔍 **Platform:** {parsed['platform'].title()}")
        
        # Skills details
        if parsed.get('skills'):
            with st.expander("🛠️ Skills Details"):
                skills_str = ", ".join(parsed['skills'])
                st.write(f"**Identified Skills:** {skills_str}")
        
        # Salary prediction
        with st.expander("💰 AI Salary Prediction"):
            salary_pred = predict_salary_dynamically(parsed)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💵 Predicted Range", 
                         f"₹{salary_pred['min']}-{salary_pred['max']} LPA")
            with col2:
                factors = salary_pred['factors']
                st.write("**Factors:**")
                st.write(f"• Role multiplier: {factors['role_multiplier']}x")
                st.write(f"• Location multiplier: {factors['location_multiplier']}x")
                st.write(f"• Skills bonus: +{factors['skill_bonus']*100:.0f}%")
    
    else:  # Recruiter view
        col1, col2 = st.columns(2)
        with col1:
            role_display = parsed.get('role') or "❌ Not specified"
            st.metric("🎯 Looking for", role_display)
            
            location_display = parsed.get('location') or "❌ Any location"
            st.metric("📍 Location", location_display)
        
        with col2:
            exp_display = (parsed.get('experience') or "entry").title()
            st.metric("📈 Experience Level", exp_display)
            
            skills_count = len(parsed.get('skills', []))
            skills_display = f"{skills_count} skills" if skills_count > 0 else "❌ None specified"
            st.metric("🛠️ Required Skills", skills_display)

def display_jobs(jobs, parsed_data):
    """Display job results with enhanced formatting"""
    if not jobs:
        st.warning("🔍 No jobs found. Try adjusting your search criteria.")
        return
    
    st.subheader(f"💼 Found {len(jobs)} Job Opportunities")
    
    for i, job in enumerate(jobs):
        with st.container():
            # Job header
            # Job header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### 🏢 [{job.get('job_title', 'N/A')}]({job.get('job_apply_link', '#')})")
                st.markdown(f"**🏢 Company:** {job.get('employer_name', 'N/A')}")
            with col2:
                if job.get('job_posted_at_datetime_utc'):
                    st.markdown(f"**📅 Posted:** {job.get('job_posted_at_datetime_utc', 'N/A')[:10]}")
                st.markdown(f"**🔗 [Apply Now]({job.get('job_apply_link', '#')})**")
            
            # Job details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**📍 Location:** {job.get('job_city', 'N/A')}, {job.get('job_state', 'N/A')}")
                st.markdown(f"**💼 Type:** {job.get('job_employment_type', 'N/A')}")
            with col2:
                if job.get('job_min_salary') and job.get('job_max_salary'):
                    currency = job.get('job_salary_currency', 'USD')
                    period = job.get('job_salary_period', 'YEAR')
                    st.markdown(f"**💰 Salary:** {currency} {job.get('job_min_salary', 0):,} - {job.get('job_max_salary', 0):,} / {period}")
                else:
                    st.markdown("**💰 Salary:** Not disclosed")
            with col3:
                if job.get('job_required_experience'):
                    required_exp = job.get('job_required_experience', {})
                    if isinstance(required_exp, dict):
                        exp_text = f"{required_exp.get('required_experience_in_months', 0)//12} years"
                    else:
                        exp_text = str(required_exp)
                    st.markdown(f"**📈 Experience:** {exp_text}")
                else:
                    st.markdown("**📈 Experience:** Not specified")
            
            # Job description
            if job.get('job_description'):
                with st.expander(f"📄 Job Description - {job.get('job_title', 'N/A')}"):
                    # Clean and truncate description
                    description = job.get('job_description', '')[:1000]
                    if len(job.get('job_description', '')) > 1000:
                        description += "..."
                    st.write(description)
            
            # Match score calculation
            if parsed_data and classifier:
                try:
                    job_text = f"{job.get('job_title', '')} {job.get('job_description', '')}"
                    user_skills = parsed_data.get('skills', [])
                    user_role = parsed_data.get('role', '')
                    
                    if user_skills or user_role:
                        labels = user_skills + [user_role] if user_role else user_skills
                        if labels and job_text.strip():
                            result = classifier(job_text[:512], labels)  # Limit text length
                            max_score = max(result['scores']) if result['scores'] else 0
                            match_percentage = int(max_score * 100)
                            
                            if match_percentage >= 70:
                                st.success(f"🎯 **Match Score: {match_percentage}%** - Excellent match!")
                            elif match_percentage >= 50:
                                st.info(f"🎯 **Match Score: {match_percentage}%** - Good match")
                            else:
                                st.warning(f"🎯 **Match Score: {match_percentage}%** - Partial match")
                except Exception as e:
                    st.warning(f"Could not calculate match score: {e}")
            
            st.divider()

def display_candidates(candidates, parsed_data):
    """Display candidate results with enhanced formatting"""
    if not candidates:
        st.warning("🔍 No candidates found. Try adjusting your search criteria.")
        return
    
    st.subheader(f"👥 Found {len(candidates)} Potential Candidates")
    
    for i, candidate in enumerate(candidates):
        with st.container():
            # Candidate header
            col1, col2 = st.columns([3, 1])
            with col1:
                title = candidate.get('title', 'N/A')
                link = candidate.get('link', '#')
                st.markdown(f"### 👤 [{title}]({link})")
            with col2:
                st.markdown(f"**🔗 [View Profile]({link})**")
            
            # Candidate details
            if candidate.get('snippet'):
                st.markdown(f"**📝 Summary:** {candidate.get('snippet', 'N/A')}")
            
            # Match score for candidates
            if parsed_data and classifier:
                try:
                    candidate_text = f"{candidate.get('title', '')} {candidate.get('snippet', '')}"
                    user_skills = parsed_data.get('skills', [])
                    user_role = parsed_data.get('role', '')
                    
                    if user_skills or user_role:
                        labels = user_skills + [user_role] if user_role else user_skills
                        if labels and candidate_text.strip():
                            result = classifier(candidate_text[:512], labels)
                            max_score = max(result['scores']) if result['scores'] else 0
                            match_percentage = int(max_score * 100)
                            
                            if match_percentage >= 70:
                                st.success(f"🎯 **Relevance Score: {match_percentage}%** - Highly relevant!")
                            elif match_percentage >= 50:
                                st.info(f"🎯 **Relevance Score: {match_percentage}%** - Good match")
                            else:
                                st.warning(f"🎯 **Relevance Score: {match_percentage}%** - Partial match")
                except Exception as e:
                    st.warning(f"Could not calculate relevance score: {e}")
            
            st.divider()

def main():
    """Main application logic"""
    # Sidebar for user type selection
    st.sidebar.title("🎭 User Type")
    user_type = st.sidebar.radio(
        "Are you a:",
        ["Job Seeker 🔍", "Recruiter 🎯"],
        help="Select your role to get personalized results"
    )
    
    # User input
    if user_type == "Job Seeker 🔍":
        st.markdown("### 🔍 Find Your Dream Job")
        placeholder_text = "e.g., 'Looking for a senior Python developer role in Bangalore with 5 years experience'"
        help_text = "Describe your job requirements in natural language"
        user_type_key = "jobseeker"
    else:
        st.markdown("### 🎯 Find Perfect Candidates")
        placeholder_text = "e.g., 'Need a React developer with 3+ years experience for our startup in Mumbai'"
        help_text = "Describe the candidate you're looking for"
        user_type_key = "recruiter"
    
    user_input = st.text_area(
        "💬 Describe what you're looking for:",
        placeholder=placeholder_text,
        help=help_text,
        height=100
    )
    
    # Search button
    if st.button("🚀 AI-Powered Search", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("🧠 AI is analyzing your query..."):
                # Parse the input
                parsed = enhanced_parse_prompt_gpt(user_input)
                
                # Display parsed results
                display_parsed_results(parsed, user_type_key)
                
                # Search based on user type
                if user_type_key == "jobseeker":
                    with st.spinner("🔍 Searching for jobs..."):
                        jobs = fetch_jobs(parsed)
                        display_jobs(jobs, parsed)
                else:
                    with st.spinner("🔍 Searching for candidates..."):
                        candidates = fetch_candidates(parsed)
                        display_candidates(candidates, parsed)
        else:
            st.error("⚠️ Please enter your requirements to start searching!")
    
    # Additional features in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Pro Tips")
    
    if user_type_key == "jobseeker":
        st.sidebar.markdown("""
        **For better results, mention:**
        - 🎯 Specific job title
        - 📍 Preferred location
        - 🛠️ Key skills
        - 📈 Years of experience
        - 💰 Salary expectations
        - 🏠 Work preference (remote/onsite)
        """)
    else:
        st.sidebar.markdown("""
        **For better results, mention:**
        - 🎯 Required role/position
        - 🛠️ Must-have skills
        - 📈 Experience level needed
        - 📍 Work location
        - 🏢 Industry/domain
        - 💼 Employment type
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Status")
    
    # Check API availability
    api_status = []
    if JSEARCH_API_KEY:
        api_status.append("✅ Job Search API")
    else:
        api_status.append("❌ Job Search API")
    
    if SERP_API_KEY:
        api_status.append("✅ Candidate Search API")
    else:
        api_status.append("❌ Candidate Search API")
    
    if client:
        api_status.append("✅ AI Parser")
    else:
        api_status.append("❌ AI Parser")
    
    if classifier and sentiment:
        api_status.append("✅ NLP Models")
    else:
        api_status.append("❌ NLP Models")
    
    for status in api_status:
        st.sidebar.markdown(status)
    
    # About section
    with st.sidebar.expander("ℹ️ About This App"):
        st.markdown("""
        **AI-Powered Job & Candidate Finder Pro** uses advanced NLP and multiple APIs to:
        
        - 🧠 Parse natural language queries
        - 🎯 Match jobs/candidates intelligently  
        - 💰 Predict salary ranges
        - 📍 Validate locations
        - 🔍 Search across multiple platforms
        
        Built with Streamlit, OpenAI, and various job APIs.
        """)

if __name__ == "__main__":
    main()
