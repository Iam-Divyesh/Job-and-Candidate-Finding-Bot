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

st.title("AI-Powered Job & Candidate Finder Pro")

def detect_user_location():
    """Detect user's location using IP geolocation and OpenCage API"""
    try:
        # First try to get IP-based location
        ip_response = requests.get('https://ipapi.co/json/', timeout=5)
        if ip_response.status_code == 200:
            ip_data = ip_response.json()
            city = ip_data.get('city')
            region = ip_data.get('region')
            country = ip_data.get('country_name')
            
            if city and region:
                detected_location = f"{city}, {region}"
                if country and country != 'India':
                    detected_location += f", {country}"
                
                # Validate with OpenCage if available
                if OPENCAGE_API_KEY:
                    validated_location = validate_location_with_api(detected_location)
                    return validated_location if validated_location != detected_location else detected_location
                
                return detected_location
        
        return None
    except Exception as e:
        st.warning(f"Could not detect location automatically: {e}")
        return None

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
        
        # Auto-detect location if not provided
        if not validated.get('location') or validated['location'] == 'null':
            detected_location = detect_user_location()
            if detected_location:
                validated['location'] = detected_location
                st.info(f"Auto-detected your location: {detected_location}")
        
        return validated
        
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error: {e}")
        return fallback_parser(user_input)
    except Exception as e:
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
    
    # Extract role with improved patterns
    role_patterns = [
        r'looking for\s+(?:a\s+)?(.+?)\s+(?:role|position|job)',
        r'(?:want|need)\s+(?:a\s+)?(.+?)\s+(?:role|position|job)',
        r'searching for\s+(.+?)\s+(?:opportunities|positions)',
        r'find\s+(?:me\s+)?(.+?)\s+jobs?',
        r'apply\s+for\s+(.+?)\s+(?:role|position)',
        r'(.+?)\s+developer',
        r'(.+?)\s+engineer',
        r'(.+?)\s+analyst'
    ]
    
    for pattern in role_patterns:
        match = re.search(pattern, text)
        if match:
            role = match.group(1).strip()
            # Clean common words
            role = re.sub(r'\b(the|a|an|some|any)\b', '', role).strip()
            if role and len(role) > 2:
                result["role"] = role.title()
                break
    
    # Enhanced skills extraction with more comprehensive list
    skill_keywords = {
        'Python': ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
        'JavaScript': ['javascript', 'js', 'typescript', 'ts', 'node.js', 'nodejs'],
        'React': ['react', 'reactjs', 'react.js', 'jsx'],
        'Angular': ['angular', 'angularjs'],
        'Vue': ['vue', 'vuejs', 'vue.js'],
        'Java': ['java', 'spring', 'springboot', 'hibernate'],
        'C#': ['c#', 'csharp', '.net', 'dotnet', 'asp.net'],
        'PHP': ['php', 'laravel', 'symfony', 'codeigniter'],
        'Ruby': ['ruby', 'rails', 'ruby on rails'],
        'Go': ['golang', 'go'],
        'Rust': ['rust'],
        'Swift': ['swift', 'ios development'],
        'Kotlin': ['kotlin', 'android development'],
        'HTML': ['html', 'html5'],
        'CSS': ['css', 'css3', 'sass', 'scss', 'less'],
        'SQL': ['sql', 'mysql', 'postgresql', 'sqlite', 'oracle'],
        'MongoDB': ['mongodb', 'mongo', 'nosql'],
        'Redis': ['redis'],
        'Docker': ['docker', 'containerization'],
        'Kubernetes': ['kubernetes', 'k8s'],
        'AWS': ['aws', 'amazon web services', 'ec2', 's3', 'lambda'],
        'Azure': ['azure', 'microsoft azure'],
        'GCP': ['gcp', 'google cloud', 'gcp'],
        'Git': ['git', 'github', 'gitlab', 'version control'],
        'Machine Learning': ['machine learning', 'ml', 'ai', 'artificial intelligence'],
        'Data Science': ['data science', 'data analysis', 'analytics', 'data mining'],
        'DevOps': ['devops', 'ci/cd', 'jenkins', 'automation'],
        'Bootstrap': ['bootstrap'],
        'Tailwind': ['tailwind', 'tailwindcss'],
        'Express': ['express', 'expressjs'],
        'Next.js': ['next.js', 'nextjs'],
        'GraphQL': ['graphql'],
        'REST API': ['rest api', 'restful', 'api development'],
        'Microservices': ['microservices', 'microservice architecture'],
        'TensorFlow': ['tensorflow', 'tf'],
        'PyTorch': ['pytorch'],
        'Spark': ['apache spark', 'spark'],
        'Hadoop': ['hadoop'],
        'Power BI': ['power bi', 'powerbi'],
        'Tableau': ['tableau'],
        'Excel': ['excel', 'advanced excel'],
        'Linux': ['linux', 'unix'],
        'Agile': ['agile', 'scrum', 'kanban'],
        'Testing': ['testing', 'unit testing', 'automation testing', 'selenium']
    }
    
    found_skills = []
    for skill, keywords in skill_keywords.items():
        if any(keyword in text for keyword in keywords):
            found_skills.append(skill)
    
    result["skills"] = found_skills
    
    # Enhanced location extraction with more Indian cities and international locations
    location_patterns = [
        r'in\s+([a-zA-Z\s,]+?)(?:\s|$|,)',
        r'at\s+([a-zA-Z\s,]+?)(?:\s|$|,)',
        r'based\s+in\s+([a-zA-Z\s,]+?)(?:\s|$|,)',
        r'location:?\s*([a-zA-Z\s,]+?)(?:\s|$|,)',
        r'from\s+([a-zA-Z\s,]+?)(?:\s|$|,)'
    ]
    
    # Comprehensive list of locations
    indian_cities = [
        'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'pune', 
        'ahmedabad', 'kolkata', 'surat', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 
        'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri', 'patna', 'vadodara', 
        'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 
        'kalyan', 'vasai', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad', 'amritsar',
        'navi mumbai', 'allahabad', 'prayagraj', 'ranchi', 'howrah', 'coimbatore',
        'jabalpur', 'gwalior', 'vijayawada', 'jodhpur', 'madurai', 'raipur', 'kota',
        'guwahati', 'chandigarh', 'solapur', 'hubballi', 'tiruchirappalli', 'bareilly',
        'mysore', 'tiruppur', 'gurgaon', 'gurugram', 'noida', 'greater noida',
        'thiruvananthapuram', 'kochi', 'kozhikode', 'thrissur', 'salem', 'erode'
    ]
    
    international_locations = [
        'usa', 'united states', 'america', 'us', 'uk', 'united kingdom', 'britain',
        'canada', 'australia', 'germany', 'singapore', 'dubai', 'uae', 'netherlands',
        'switzerland', 'sweden', 'norway', 'denmark', 'france', 'italy', 'spain',
        'japan', 'south korea', 'new zealand', 'ireland', 'israel', 'hong kong'
    ]
    
    all_locations = indian_cities + international_locations
    
    # Check for remote work first
    if any(word in text for word in ['remote', 'work from home', 'wfh', 'anywhere']):
        result["location"] = "remote"
        result["work_preference"] = "remote"
    else:
        # Try to find specific locations
        for location in all_locations:
            if location in text:
                result["location"] = location.title()
                break
        
        # If no specific location found, try patterns
        if not result["location"]:
            for pattern in location_patterns:
                match = re.search(pattern, text)
                if match:
                    location = match.group(1).strip().rstrip(',')
                    # Clean the location
                    location = re.sub(r'\b(the|a|an|with|for|and)\b', '', location).strip()
                    if len(location) > 2:
                        result["location"] = location.title()
                        break
    
    # Enhanced experience extraction
    years_match = re.search(r'(\d+)(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', text)
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
        exp_patterns = {
            "entry": r'\b(?:fresh|fresher|graduate|entry|junior|0-2|new|beginner)\b',
            "mid": r'\b(?:3-5|mid|intermediate|experienced|2-5)\b',
            "senior": r'\b(?:senior|6-10|7-8|experienced|lead)\b',
            "lead": r'\b(?:lead|manager|architect|principal|10\+|director)\b'
        }
        
        for exp_level, pattern in exp_patterns.items():
            if re.search(pattern, text):
                result["experience"] = exp_level
                break
    
    # Work preference
    if 'hybrid' in text:
        result["work_preference"] = "hybrid"
    elif any(word in text for word in ['onsite', 'on-site', 'office']):
        result["work_preference"] = "onsite"
    
    # Job type
    if any(word in text for word in ['part-time', 'part time', 'parttime']):
        result["job_type"] = "part-time"
    elif any(word in text for word in ['contract', 'freelance', 'contractor']):
        result["job_type"] = "contract"
    elif any(word in text for word in ['internship', 'intern']):
        result["job_type"] = "internship"
    elif any(word in text for word in ['full-time', 'full time', 'fulltime']):
        result["job_type"] = "full-time"
    
    # Salary extraction
    salary_patterns = [
        r'(\d+)\s*(?:lpa|lakhs?|l)\b',
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
    
    # Industry extraction
    industries = {
        'fintech': r'\b(?:fintech|financial|banking|finance|payment)\b',
        'healthcare': r'\b(?:healthcare|medical|health|pharma|hospital)\b', 
        'e-commerce': r'\b(?:ecommerce|e-commerce|retail|shopping|marketplace)\b',
        'gaming': r'\b(?:gaming|game|games|entertainment)\b',
        'education': r'\b(?:education|edtech|learning|training)\b',
        'travel': r'\b(?:travel|tourism|booking|hospitality)\b',
        'automotive': r'\b(?:automotive|automobile|car|vehicle)\b',
        'telecommunications': r'\b(?:telecom|telecommunications|mobile|network)\b',
        'media': r'\b(?:media|advertising|marketing|social media)\b'
    }
    
    for industry, pattern in industries.items():
        if re.search(pattern, text):
            result["industry"] = industry
            break
    
    # Platform extraction
    platforms = {
        'linkedin': r'\blinkedin\b',
        'naukri': r'\bnaukri\b',
        'indeed': r'\bindeed\b',
        'glassdoor': r'\bglassdoor\b',
        'monster': r'\bmonster\b',
        'shine': r'\bshine\b'
    }
    
    for platform, pattern in platforms.items():
        if re.search(pattern, text):
            result["platform"] = platform
            break
    
    # Auto-detect location if not found
    if not result["location"]:
        detected_location = detect_user_location()
        if detected_location:
            result["location"] = detected_location
    
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
        return location

def extract_experience_from_description(description):
    """Extract experience requirement from job description"""
    if not description:
        return None
    
    text = description.lower()
    
    # Pattern to find experience requirements
    experience_patterns = [
        r'(\d+)(?:\+|\s*to\s*\d+)?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
        r'minimum\s*(\d+)\s*(?:years?|yrs?)',
        r'at least\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)(?:\+)?\s*(?:years?|yrs?)\s*(?:in|of|with)',
        r'experience:?\s*(\d+)(?:\+)?\s*(?:years?|yrs?)',
        r'(\d+)(?:\+)?\s*(?:years?|yrs?)\s*relevant'
    ]
    
    for pattern in experience_patterns:
        match = re.search(pattern, text)
        if match:
            years = int(match.group(1))
            return f"{years}+ years"
    
    # Check for level-based requirements
    if any(word in text for word in ['senior', 'lead', 'principal']):
        return "Senior level"
    elif any(word in text for word in ['mid-level', 'intermediate', 'experienced']):
        return "Mid level"  
    elif any(word in text for word in ['junior', 'entry', 'fresher']):
        return "Entry level"
    
    return None

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
        st.warning("JSEARCH_API_KEY not found in environment variables")
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
    
    # Add location if specified
    if parsed.get('location') and parsed['location'].lower() != 'remote':
        query_parts.append(f"in {parsed['location']}")
    
    # Add experience level context
    if parsed.get('experience'):
        exp_mapping = {
            'entry': 'junior entry level',
            'mid': 'mid level experienced',
            'senior': 'senior experienced',
            'lead': 'lead manager principal'
        }
        query_parts.append(exp_mapping.get(parsed['experience'], ''))
    
    query = ' '.join(query_parts)
    
    params = {
        "query": query,
        "page": "1",
        "num_pages": "3",
        "date_posted": "month"
    }
    
    # Add location-specific parameters
    if parsed.get('location') and parsed['location'].lower() != 'remote':
        # Check if it's an Indian location
        indian_cities = ['mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad', 'chennai', 'pune', 'kolkata']
        if any(city in parsed['location'].lower() for city in indian_cities):
            params["country"] = "IN"
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            jobs = data.get('data', [])
            
            # Enhanced job processing
            processed_jobs = []
            for job in jobs:
                processed_job = {
                    'id': job.get('job_id', ''),
                    'title': job.get('job_title', ''),
                    'company': job.get('employer_name', ''),
                    'location': job.get('job_city', '') + ', ' + job.get('job_state', '') if job.get('job_city') else job.get('job_country', ''),
                    'description': job.get('job_description', ''),
                    'salary': extract_salary_from_job(job),
                    'experience_required': extract_experience_from_description(job.get('job_description', '')),
                    'skills_required': extract_skills_from_job(job.get('job_description', '')),
                    'job_type': job.get('job_employment_type', ''),
                    'posted_date': job.get('job_posted_at_datetime_utc', ''),
                    'apply_link': job.get('job_apply_link', ''),
                    'remote': job.get('job_is_remote', False),
                    'company_logo': job.get('employer_logo', ''),
                    'source': 'JSearch'
                }
                processed_jobs.append(processed_job)
            
            return processed_jobs[:20]  # Return top 20 jobs
        else:
            st.error(f"Job search API error: {response.status_code}")
            return []
            
    except requests.exceptions.Timeout:
        st.error("Job search request timed out. Please try again.")
        return []
    except Exception as e:
        st.error(f"Error fetching jobs: {str(e)}")
        return []

def extract_salary_from_job(job):
    """Extract salary information from job data"""
    salary_info = {
        'min': None,
        'max': None,
        'currency': 'USD',
        'period': 'annual'
    }
    
    # Check various salary fields
    if job.get('job_min_salary') and job.get('job_max_salary'):
        salary_info['min'] = job['job_min_salary']
        salary_info['max'] = job['job_max_salary']
        salary_info['currency'] = job.get('job_salary_currency', 'USD')
        salary_info['period'] = job.get('job_salary_period', 'annual')
    
    # Try to extract from description if not available
    if not salary_info['min'] and job.get('job_description'):
        salary_from_desc = extract_salary_from_description(job['job_description'])
        if salary_from_desc:
            salary_info.update(salary_from_desc)
    
    return salary_info

def extract_salary_from_description(description):
    """Extract salary from job description text"""
    if not description:
        return None
    
    text = description.lower()
    
    # Patterns for Indian salary (lakhs)
    lakh_patterns = [
        r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*lpa',
        r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*lakhs?',
        r'(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*lakhs?',
        r'salary:?\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*lpa'
    ]
    
    for pattern in lakh_patterns:
        match = re.search(pattern, text)
        if match:
            return {
                'min': float(match.group(1)),
                'max': float(match.group(2)),
                'currency': 'INR',
                'period': 'annual'
            }
    
    # Patterns for USD salary
    usd_patterns = [
        r'\$(\d+(?:,\d+)?)\s*-\s*\$(\d+(?:,\d+)?)',
        r'(\d+(?:,\d+)?)\s*-\s*(\d+(?:,\d+)?)\s*usd',
        r'salary:?\s*\$(\d+(?:,\d+)?)\s*-\s*\$(\d+(?:,\d+)?)'
    ]
    
    for pattern in usd_patterns:
        match = re.search(pattern, text)
        if match:
            min_sal = float(match.group(1).replace(',', ''))
            max_sal = float(match.group(2).replace(',', ''))
            return {
                'min': min_sal,
                'max': max_sal,
                'currency': 'USD',
                'period': 'annual'
            }
    
    return None

def extract_skills_from_job(description):
    """Extract required skills from job description"""
    if not description:
        return []
    
    text = description.lower()
    
    # Comprehensive skills database
    skills_database = {
        'Programming Languages': [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 
            'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl'
        ],
        'Web Technologies': [
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 
            'flask', 'spring', 'laravel', 'asp.net', 'next.js', 'nuxt.js', 'gatsby'
        ],
        'Databases': [
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 
            'cassandra', 'oracle', 'sqlite', 'dynamodb', 'neo4j'
        ],
        'Cloud & DevOps': [
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 
            'ansible', 'git', 'gitlab', 'github', 'ci/cd', 'linux', 'unix'
        ],
        'Data Science & ML': [
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 
            'numpy', 'scikit-learn', 'spark', 'hadoop', 'tableau', 'power bi'
        ],
        'Mobile Development': [
            'android', 'ios', 'react native', 'flutter', 'xamarin', 'cordova', 'ionic'
        ],
        'Others': [
            'agile', 'scrum', 'kanban', 'jira', 'confluence', 'microservices', 
            'rest api', 'graphql', 'oauth', 'jwt', 'websockets'
        ]
    }
    
    found_skills = []
    
    for category, skills in skills_database.items():
        for skill in skills:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text):
                found_skills.append(skill.title())
    
    # Remove duplicates and return
    return list(set(found_skills))

def calculate_job_match_score(job, parsed_data):
    """Calculate match score between job and user preferences"""
    score = 0
    total_weight = 0
    details = []
    
    # Role matching (weight: 30)
    if parsed_data.get('role') and job.get('title'):
        role_similarity = calculate_text_similarity(parsed_data['role'], job['title'])
        score += role_similarity * 30
        total_weight += 30
        details.append(f"Role match: {role_similarity:.0%}")
    
    # Skills matching (weight: 25)
    if parsed_data.get('skills') and job.get('skills_required'):
        user_skills = [s.lower() for s in parsed_data['skills']]
        job_skills = [s.lower() for s in job['skills_required']]
        
        if user_skills and job_skills:
            matching_skills = set(user_skills) & set(job_skills)
            skill_score = len(matching_skills) / len(user_skills) if user_skills else 0
            score += skill_score * 25
            total_weight += 25
            details.append(f"Skills match: {skill_score:.0%} ({len(matching_skills)}/{len(user_skills)})")
    
    # Location matching (weight: 20)
    if parsed_data.get('location') and job.get('location'):
        if parsed_data['location'].lower() == 'remote' and job.get('remote'):
            location_score = 1.0
        else:
            location_score = calculate_location_similarity(parsed_data['location'], job['location'])
        
        score += location_score * 20
        total_weight += 20
        details.append(f"Location match: {location_score:.0%}")
    
    # Experience matching (weight: 15)
    if parsed_data.get('experience') and job.get('experience_required'):
        exp_score = calculate_experience_match(parsed_data['experience'], job['experience_required'])
        score += exp_score * 15
        total_weight += 15
        details.append(f"Experience match: {exp_score:.0%}")
    
    # Salary matching (weight: 10)
    if parsed_data.get('salary') and job.get('salary'):
        salary_score = calculate_salary_match(parsed_data['salary'], job['salary'])
        score += salary_score * 10
        total_weight += 10
        details.append(f"Salary match: {salary_score:.0%}")
    
    # Calculate final score
    final_score = (score / total_weight * 100) if total_weight > 0 else 0
    
    return {
        'score': round(final_score, 1),
        'details': details,
        'category': get_match_category(final_score)
    }

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text strings using TF-IDF"""
    if not text1 or not text2:
        return 0.0
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform([text1.lower(), text2.lower()])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        # Fallback to simple word matching
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0

def calculate_location_similarity(user_location, job_location):
    """Calculate similarity between user and job locations"""
    if not user_location or not job_location:
        return 0.0
    
    user_loc = user_location.lower().strip()
    job_loc = job_location.lower().strip()
    
    # Exact match
    if user_loc == job_loc:
        return 1.0
    
    # Check if one location contains the other
    if user_loc in job_loc or job_loc in user_loc:
        return 0.8
    
    # Check for city matches (common Indian cities)
    user_parts = user_loc.replace(',', ' ').split()
    job_parts = job_loc.replace(',', ' ').split()
    
    common_parts = set(user_parts) & set(job_parts)
    if common_parts:
        return 0.6
    
    return 0.0

def calculate_experience_match(user_exp, job_exp_text):
    """Calculate experience level matching"""
    if not job_exp_text:
        return 0.5  # Neutral if no experience mentioned
    
    job_text = job_exp_text.lower()
    
    # Experience level mappings
    exp_weights = {
        'entry': 1,
        'mid': 3,
        'senior': 7,
        'lead': 12
    }
    
    user_level = exp_weights.get(user_exp, 1)
    
    # Extract years from job requirement
    years_match = re.search(r'(\d+)', job_text)
    if years_match:
        required_years = int(years_match.group(1))
        
        # Calculate compatibility
        if required_years <= user_level + 1:
            return 1.0
        elif required_years <= user_level + 3:
            return 0.7
        else:
            return 0.3
    
    # Check for level keywords
    if any(word in job_text for word in ['senior', 'lead', 'principal']) and user_exp in ['senior', 'lead']:
        return 1.0
    elif any(word in job_text for word in ['junior', 'entry', 'fresher']) and user_exp == 'entry':
        return 1.0
    elif 'mid' in job_text and user_exp == 'mid':
        return 1.0
    
    return 0.5

def calculate_salary_match(user_salary, job_salary):
    """Calculate salary expectation matching"""
    if not job_salary or not job_salary.get('min'):
        return 0.5  # Neutral if no salary info
    
    try:
        user_sal = float(user_salary)
        job_min = float(job_salary['min'])
        job_max = float(job_salary.get('max', job_min))
        
        # Convert currency if needed
        if job_salary.get('currency') == 'USD':
            # Convert USD to INR (approximate)
            job_min *= 83  # 1 USD = ~83 INR
            job_max *= 83
            job_min /= 100000  # Convert to lakhs
            job_max /= 100000
        
        # Check if user expectation is within range
        if job_min <= user_sal <= job_max:
            return 1.0
        elif user_sal < job_min:
            # User expectation is lower (good for employer)
            return 0.8
        else:
            # User expectation is higher
            gap = (user_sal - job_max) / job_max
            if gap <= 0.2:  # Within 20%
                return 0.6
            else:
                return 0.3
    except:
        return 0.5

def get_match_category(score):
    """Get match category based on score"""
    if score >= 90:
        return "Excellent match!"
    elif score >= 80:
        return "Very good match!"
    elif score >= 70:
        return "Good match!"
    elif score >= 60:
        return "Fair match"
    else:
        return "Low match"

def display_jobs(jobs, parsed_data):
    """Display jobs with enhanced formatting and match scores"""
    if not jobs:
        st.warning("No jobs found matching your criteria.")
        return
    
    # Calculate match scores for all jobs
    jobs_with_scores = []
    for job in jobs:
        match_info = calculate_job_match_score(job, parsed_data)
        job['match_score'] = match_info['score']
        job['match_details'] = match_info['details']
        job['match_category'] = match_info['category']
        jobs_with_scores.append(job)
    
    # Sort jobs by match score (highest first)
    jobs_with_scores.sort(key=lambda x: x['match_score'], reverse=True)
    
    st.subheader(f"Found {len(jobs_with_scores)} Jobs")
    
    for i, job in enumerate(jobs_with_scores, 1):
        with st.container():
            # Create columns for layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {i}. {job['title']}")
                st.markdown(f"**Company:** {job['company']}")
                st.markdown(f"**Location:** {job['location']}")
                
                # Display experience required if available
                if job.get('experience_required'):
                    st.markdown(f"**Experience Required:** {job['experience_required']}")
                
                # Display salary if available
                if job.get('salary') and job['salary'].get('min'):
                    salary = job['salary']
                    if salary['currency'] == 'INR':
                        st.markdown(f"**Salary:** {salary['min']}-{salary.get('max', salary['min'])} LPA")
                    else:
                        st.markdown(f"**Salary:** ${salary['min']:,}-${salary.get('max', salary['min']):,} {salary.get('period', 'annual')}")
                
                # Display skills if available
                if job.get('skills_required'):
                    skills_text = ', '.join(job['skills_required'][:8])  # Show first 8 skills
                    if len(job['skills_required']) > 8:
                        skills_text += f" +{len(job['skills_required']) - 8} more"
                    st.markdown(f"**Skills:** {skills_text}")
                
                # Job type and remote info
                job_details = []
                if job.get('job_type'):
                    job_details.append(job['job_type'])
                if job.get('remote'):
                    job_details.append("Remote")
                
                if job_details:
                    st.markdown(f"**Type:** {' | '.join(job_details)}")
            
            with col2:
                # Match score display
                score = job['match_score']
                st.markdown(f"### Match Score: {score}%")
                st.markdown(f"**{job['match_category']}**")
                
                # Progress bar for visual representation
                st.progress(score / 100)
                
                # Apply button
                if job.get('apply_link'):
                    st.markdown(f"[Apply Now]({job['apply_link']})")
            
            # Job description (truncated)
            if job.get('description'):
                description = job['description'][:300] + "..." if len(job['description']) > 300 else job['description']
                with st.expander("Job Description"):
                    st.text(description)
            
            # Match details
            if job.get('match_details'):
                with st.expander("Match Details"):
                    for detail in job['match_details']:
                        st.text(f"• {detail}")
            
            st.divider()

# Main application logic
def main():
    st.sidebar.header("Job Search Filters")
    
    # User input
    user_query = st.text_area(
        "Describe your job preferences:",
        placeholder="e.g., Looking for a senior Python developer role in Bangalore with 5+ years experience, remote work preferred, 15+ LPA salary"
    )
    
    if st.button("Search Jobs", type="primary"):
        if user_query:
            with st.spinner("Analyzing your requirements..."):
                # Parse user input
                parsed_data = enhanced_parse_prompt_gpt(user_query)
                
                # Display parsed information
                st.subheader("Understood Requirements:")
                col1, col2 = st.columns(2)
                
                with col1:
                    if parsed_data.get('role'):
                        st.write(f"**Role:** {parsed_data['role']}")
                    if parsed_data.get('skills'):
                        st.write(f"**Skills:** {', '.join(parsed_data['skills'])}")
                    if parsed_data.get('experience'):
                        st.write(f"**Experience:** {parsed_data['experience'].title()}")
                    if parsed_data.get('location'):
                        st.write(f"**Location:** {parsed_data['location']}")
                
                with col2:
                    if parsed_data.get('salary'):
                        st.write(f"**Salary:** {parsed_data['salary']} LPA")
                    if parsed_data.get('work_preference'):
                        st.write(f"**Work Mode:** {parsed_data['work_preference'].title()}")
                    if parsed_data.get('job_type'):
                        st.write(f"**Job Type:** {parsed_data['job_type'].title()}")
                    if parsed_data.get('industry'):
                        st.write(f"**Industry:** {parsed_data['industry'].title()}")
                
                # Salary prediction if not specified
                if not parsed_data.get('salary'):
                    predicted_salary = predict_salary_dynamically(parsed_data)
                    st.info(f"Predicted Salary Range: {predicted_salary['min']}-{predicted_salary['max']} LPA")
            
            with st.spinner("Searching for jobs..."):
                # Fetch jobs
                jobs = fetch_jobs(parsed_data)
                
                # Display jobs
                display_jobs(jobs, parsed_data)
        else:
            st.warning("Please enter your job preferences to search.")
    
    # Add sidebar information
    st.sidebar.subheader("Tips for Better Results:")
    st.sidebar.write("• Mention specific skills and technologies")
    st.sidebar.write("• Include your experience level")
    st.sidebar.write("• Specify location preferences")
    st.sidebar.write("• Add salary expectations")
    st.sidebar.write("• Mention work preferences (remote/onsite)")

if __name__ == "__main__":
    main()