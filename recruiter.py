import streamlit as st
import requests
import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
import time
from typing import Dict, List, Optional
import urllib.parse

# Load environment variables
load_dotenv()

# Azure OpenAI Setup - Enhanced for better model
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")  # Upgraded to GPT-4

# SERP API Key
serp_api_key = os.getenv("SERP_API_KEY")

# Initialize OpenAI client
try:
    client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
except Exception as e:
    st.error(f"Failed to initialize Azure OpenAI client: {e}")
    client = None

def normalize_location_with_gpt(location: str) -> str:
    """Use GPT-3.5 to normalize and standardize location names"""
    if not location or not client:
        return location.title() if location else ""
    
    try:
        system_prompt = """You are a location normalization expert. Your task is to standardize location names to their most common, recognizable form.

        RULES:
        1. Convert to standard city names (e.g., "Bombay" → "Mumbai", "Bengaluru" → "Bangalore")
        2. Handle abbreviations (e.g., "NYC" → "New York", "SF" → "San Francisco")
        3. Standardize country names (e.g., "USA" → "United States")
        4. Handle regional variations (e.g., "NCR" → "Delhi", "Silicon Valley" → "San Francisco")
        5. Return only the normalized location name, nothing else
        6. If unclear, return the original location with proper capitalization
        
        Examples:
        - "bombay" → "Mumbai"
        - "bengaluru" → "Bangalore"
        - "nyc" → "New York"
        - "sf" → "San Francisco"
        - "ncr" → "Delhi"
        - "silicon valley" → "San Francisco"
        """
        
        user_prompt = f"Normalize this location: {location}"
        
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        normalized = response.choices[0].message.content.strip()
        return normalized if normalized else location.title()
        
    except Exception as e:
        st.warning(f"Location normalization error: {e}")
        return location.title()

def normalize_location(location: str) -> str:
    """Normalize location for better matching using GPT-3.5"""
    if not location:
        return ""
    
    return normalize_location_with_gpt(location)

# --- Enhanced Function to parse recruiter query using GPT-4 ---
def parse_recruiter_query(query: str) -> Dict:
    """Enhanced query parsing with better prompting and validation"""
    if not client:
        return {"error": "Azure OpenAI client not available"}
    
    try:
        system_prompt = """You are an expert recruitment AI assistant with deep knowledge of job markets, skills, and hiring patterns across industries.

        Your task is to extract structured recruitment information from natural language queries and return a JSON object.

        EXTRACTION RULES:
        1. job_title: Extract the EXACT position title being hired for
           - Remove prefixes like "looking for", "hiring", "need"
           - Standardize common variations (e.g., "full stack developer" → "Full Stack Developer")
           - Include seniority levels (Junior, Senior, Lead, Principal)

        2. skills: Array of technical skills, tools, and technologies
           - Include programming languages, frameworks, tools, certifications
           - Separate related skills (e.g., ["React", "Node.js"] not ["React/Node.js"])
           - Include soft skills if specifically mentioned

        3. experience: Experience requirements in standardized format
           - Convert to consistent format: "2", "3-5", "5+", "10+"
           - Handle variations like "fresher" → "0", "experienced" → "3+"

        4. location: Normalized city/region name
           - Extract city, state/region, country
           - Standardize major city names (Bengaluru → Bangalore)
           - Handle multiple locations as primary location

        5. work_preference: Work arrangement preference
           - Options: "remote", "onsite", "hybrid", null
           - Infer from context (e.g., "WFH" → "remote")

        6. job_type: Employment type
           - Options: "full-time", "part-time", "contract", "internship", "freelance", null

        7. industry: Industry/domain if mentioned
           - Examples: "fintech", "healthcare", "e-commerce", "startup"

        8. company_size: Company size preference if mentioned
           - Options: "startup", "mid-size", "enterprise", "MNC"

        IMPORTANT: Return ONLY valid JSON without explanation."""
        
        user_prompt = f"""Extract recruitment information from: "{query}"

        Examples:
        Input: "Senior Python developer with Django, 5+ years experience, Mumbai, remote work"
        Output: {{"job_title": "Senior Python Developer", "skills": ["Python", "Django"], "experience": "5+", "location": "Mumbai", "work_preference": "remote", "job_type": null, "industry": null, "company_size": null}}

        Input: "Looking for React frontend engineer in Bangalore startup, 2-4 years, TypeScript, Redux"
        Output: {{"job_title": "React Frontend Engineer", "skills": ["React", "TypeScript", "Redux"], "experience": "2-4", "location": "Bangalore", "work_preference": null, "job_type": null, "industry": null, "company_size": "startup"}}

        Now extract from: "{query}"
        
        Return only the JSON object."""
        
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent extraction
            max_tokens=800,
            top_p=0.95
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean JSON response
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
        
        content = content.strip()
        
        # Parse and validate JSON
        parsed = json.loads(content)
        
        # Enhanced data cleaning and validation
        cleaned_result = {
            "job_title": parsed.get("job_title", "").strip() if parsed.get("job_title") else None,
            "skills": [skill.strip() for skill in parsed.get("skills", []) if skill.strip()],
            "experience": str(parsed.get("experience", "")).strip() if parsed.get("experience") else None,
            "location": normalize_location(parsed.get("location", "")),
            "work_preference": parsed.get("work_preference"),
            "job_type": parsed.get("job_type"),
            "industry": parsed.get("industry"),
            "company_size": parsed.get("company_size"),
            "parsing_status": "success"
        }
        
        return cleaned_result
        
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error: {e}")
        return {"error": f"JSON parsing error: {e}"}
    except Exception as e:
        st.warning(f"AI parsing error: {e}")
        return {"error": f"AI parsing error: {e}"}

# --- Enhanced location detection ---
def detect_user_location() -> Optional[str]:
    """Enhanced location detection with fallback options"""
    try:
        # Try primary IP geolocation service
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            ip_data = response.json()
            city = ip_data.get('city')
            if city:
                return normalize_location(city)
        
        # Fallback to alternative service
        response = requests.get('http://ip-api.com/json/', timeout=5)
        if response.status_code == 200:
            ip_data = response.json()
            city = ip_data.get('city')
            if city:
                return normalize_location(city)
                
        return None
    except Exception as e:
        st.warning(f"Could not detect location: {e}")
        return None

# --- Enhanced candidate fetching using SERP API ---
def fetch_candidates_serp(parsed_data: Dict) -> List[Dict]:
    """Fetch candidates using SERP API with multiple search strategies and strict location filtering"""
    if not serp_api_key:
        st.error("SERP API key not found. Please set SERP_API_KEY in your environment variables.")
        return []
    
    job_title = parsed_data.get("job_title", "")
    location = parsed_data.get("location", "")
    skills = parsed_data.get("skills", [])
    experience = parsed_data.get("experience")
    work_preference = parsed_data.get("work_preference")
    industry = parsed_data.get("industry")
    
    # Build comprehensive search queries
    search_queries = build_search_queries(parsed_data)
    
    all_candidates = []
    
    for i, query in enumerate(search_queries[:3]):  # Limit to 3 searches to avoid API limits
        st.info(f"🔍 Search {i+1}/3: {query}")
        candidates = perform_serp_search(query, location, work_preference)
        all_candidates.extend(candidates)
        time.sleep(1)  # Rate limiting
    
    # Remove duplicates and enhance candidate data
    unique_candidates = remove_duplicates(all_candidates)
    enhanced_candidates = enhance_candidate_data(unique_candidates, parsed_data)
    
    # Strict city-level location filtering (must match exactly)
    required_location = normalize_location(location).lower() if location else None
    if required_location:
        filtered_candidates = []
        for c in enhanced_candidates:
            candidate_loc = normalize_location(c.get('location', '')).lower()
            # Accept only if required_location == candidate_loc (city-level)
            if required_location == candidate_loc:
                filtered_candidates.append(c)
        return filtered_candidates
    else:
        return enhanced_candidates

def build_search_queries(parsed_data: Dict) -> List[str]:
    """Build multiple targeted search queries"""
    job_title = parsed_data.get("job_title", "")
    location = parsed_data.get("location", "")
    skills = parsed_data.get("skills", [])
    experience = parsed_data.get("experience")
    work_preference = parsed_data.get("work_preference")
    
    queries = []
    
    # Always include city and country if location is specified
    location_query = ''
    if location:
        location_query = f' "{location}" India'
    
    # Query 1: LinkedIn focused search
    linkedin_query = f'site:linkedin.com/in -site:in.linkedin.com/in "{job_title}"{location_query}'
    if skills:
        linkedin_query += f' {" ".join(skills[:2])}'
    queries.append(linkedin_query)
    
    # Query 2: General professional profile search
    general_query = f'"{job_title}" resume CV{location_query}'
    if experience:
        general_query += f' "{experience} years"'
    queries.append(general_query)
    
    # Query 3: Skills-focused search
    if skills:
        skills_query = f'{" ".join(skills[:3])} developer engineer{location_query}'
        if work_preference == "remote":
            skills_query += ' remote'
        queries.append(skills_query)
    
    # Query 4: Industry-specific search
    if parsed_data.get("industry"):
        industry_query = f'"{job_title}" {parsed_data["industry"]}{location_query}'
        queries.append(industry_query)
    
    return queries

def perform_serp_search(query: str, location: str, work_preference: str) -> List[Dict]:
    """Perform actual SERP API search"""
    try:
        params = {
            'api_key': serp_api_key,
            'engine': 'google',
            'q': query,
            'num': 20,  # Increased results
            'gl': get_country_code(location),  # Geographic location
            'hl': 'en'
        }
        
        # Add location-specific parameters
        if location:
            params['location'] = location
        
        response = requests.get('https://serpapi.com/search', params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            organic_results = data.get('organic_results', [])
            
            candidates = []
            for result in organic_results:
                candidate = process_serp_result(result, location, work_preference)
                if candidate:
                    candidates.append(candidate)
            
            return candidates
        else:
            st.warning(f"SERP API Error {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        st.error(f"Error fetching from SERP API: {e}")
        return []

def get_country_code(location: str) -> str:
    """Get country code for SERP API geographic targeting using GPT-3.5"""
    if not location or not client:
        return 'us'
    
    try:
        system_prompt = """You are a geographic expert. Given a location name, return the ISO 2-letter country code.

        Examples:
        - "Mumbai" → "in"
        - "New York" → "us"
        - "London" → "uk"
        - "Singapore" → "sg"
        - "Dubai" → "ae"
        - "Toronto" → "ca"
        - "Sydney" → "au"
        
        Return only the 2-letter country code, nothing else."""
        
        user_prompt = f"What is the country code for: {location}"
        
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        country_code = response.choices[0].message.content.strip().lower()
        
        # Validate country code format
        if len(country_code) == 2 and country_code.isalpha():
            return country_code
        
        return 'us'  # Default fallback
        
    except Exception as e:
        st.warning(f"Country code detection error: {e}")
        return 'us'

def process_serp_result(result: Dict, target_location: str, work_preference: str) -> Optional[Dict]:
    """Process individual SERP search result into candidate data"""
    try:
        title = result.get('title', '')
        link = result.get('link', '')
        snippet = result.get('snippet', '')
        
        # Skip irrelevant results
        if not is_relevant_profile(title, link, snippet):
            return None
        
        # Get thumbnail image from SERP API result
        thumbnail = result.get('thumbnail', {}).get('url', '')
        # If no thumbnail in SERP result, try to get from rich_snippet
        if not thumbnail and 'rich_snippet' in result:
            thumbnail = result.get('rich_snippet', {}).get('top', {}).get('detected_extensions', {}).get('image_url', '')
        
        # Extract candidate information
        candidate = {
            'name': extract_name_from_title(title),
            'title': title,
            'link': link,
            'snippet': snippet,
            'description': snippet[:300] + "..." if len(snippet) > 300 else snippet,
            'source': get_source_type(link),
            'location': extract_location_from_snippet(snippet, target_location),
            'skills': extract_skills_from_text(f"{title} {snippet}"),
            'experience': extract_experience_from_text(snippet),
            'linkedin_profile': link if 'linkedin.com' in link else "Not Available",
            'open_to_work': 'open to work' in snippet.lower() or 'looking for' in snippet.lower(),
            'remote_friendly': work_preference == 'remote' and ('remote' in snippet.lower() or 'work from home' in snippet.lower()),
            'image': thumbnail if thumbnail else get_default_image()
        }
        
        return candidate
        
    except Exception as e:
        st.warning(f"Error processing search result: {e}")
        return None

def is_relevant_profile(title: str, link: str, snippet: str) -> bool:
    """Check if search result is a relevant candidate profile"""
    # Check for professional profile indicators
    profile_indicators = [
        'linkedin.com/in',
        'resume',
        'cv',
        'profile',
        'developer',
        'engineer',
        'manager',
        'analyst',
        'consultant'
    ]
    
    combined_text = f"{title} {link} {snippet}".lower()
    
    # Must have at least one profile indicator
    if not any(indicator in combined_text for indicator in profile_indicators):
        return False
    
    # Exclude job postings and company pages
    exclude_patterns = [
        'jobs',
        'careers',
        'hiring',
        'vacancy',
        'openings',
        'company',
        'about us'
    ]
    
    if any(pattern in combined_text for pattern in exclude_patterns):
        return False
    
    return True

def extract_name_from_title(title: str) -> str:
    """Extract candidate name from search result title"""
    # LinkedIn profiles often have "Name | Title" format
    if ' | ' in title:
        potential_name = title.split(' | ')[0].strip()
        # Check if it looks like a name (contains spaces, reasonable length)
        if len(potential_name.split()) >= 2 and len(potential_name) <= 50:
            return potential_name
    
    # Look for name patterns in title
    import re
    name_pattern = r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    match = re.match(name_pattern, title)
    if match:
        return match.group(1)
    
    return "Name Not Available"

def extract_location_from_snippet(snippet: str, target_location: str) -> str:
    """Expert-level extraction of city from LinkedIn snippet using GPT and strict regex. Only return city if present."""
    if not client:
        return "Location Not Specified"
    try:
        # Use GPT to extract the most relevant city (not country or region)
        system_prompt = (
            "You are a location extraction expert. Extract ONLY the city name from the following text. "
            "If there is no city, or only a country/region is present, return 'Location Not Specified'. "
            "Examples: 'Mumbai, Maharashtra, India' -> 'Mumbai'; 'United States' -> 'Location Not Specified'; "
            "'St Paul, Minnesota, United States' -> 'St Paul'; 'Bangalore, India' -> 'Bangalore'; "
            "'India' -> 'Location Not Specified'"
        )
        user_prompt = f"Extract city from: {snippet}"
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        location = response.choices[0].message.content.strip()
        if location and location != "Location Not Specified":
            return normalize_location(location)
        # Fallback: strict regex for city extraction (city, state, country)
        import re
        match = re.match(r'([A-Za-z .\-]+),', snippet)
        if match:
            city = match.group(1).strip()
            # Filter out generic country/region names
            if city.lower() not in ["india", "united states", "usa", "uk", "singapore", "canada", "australia"]:
                return normalize_location(city)
        return "Location Not Specified"
    except Exception as e:
        st.warning(f"Location extraction error: {e}")
        return "Location Not Specified"

def extract_skills_from_text(text: str) -> List[str]:
    """Extract technical skills from text"""
    # Common technical skills
    skill_patterns = [
        r'\b(?:Python|Java|JavaScript|React|Node\.js|Angular|Vue|PHP|Ruby|Go|Rust|C\+\+|C#|Swift|Kotlin)\b',
        r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|MongoDB|PostgreSQL|MySQL|Redis)\b',
        r'\b(?:Machine Learning|ML|AI|Data Science|Deep Learning|TensorFlow|PyTorch)\b',
        r'\b(?:DevOps|CI/CD|Jenkins|Git|Linux|Unix|Bash)\b'
    ]
    
    skills = []
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        skills.extend(matches)
    
    return list(set(skills))  # Remove duplicates

def extract_experience_from_text(text: str) -> str:
    """Extract experience information from text"""
    exp_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'(\d+-\d+)\s*years?\s*(?:of\s*)?experience',
        r'over\s*(\d+)\s*years?',
        r'(\d+)\s*yrs?'
    ]
    
    for pattern in exp_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            return matches[0] + " years"
    
    return "Not Specified"

def get_source_type(link: str) -> str:
    """Determine the source type of the profile"""
    if 'linkedin.com' in link:
        return 'LinkedIn'
    elif 'github.com' in link:
        return 'GitHub'
    elif any(domain in link for domain in ['resume', 'cv', 'portfolio']):
        return 'Portfolio'
    else:
        return 'Web Profile'

def get_default_image() -> str:
    """Return default profile image URL"""
    return "https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_1280.png"

def remove_duplicates(candidates: List[Dict]) -> List[Dict]:
    """Remove duplicate candidates based on name and link"""
    seen = set()
    unique_candidates = []
    
    for candidate in candidates:
        identifier = (candidate.get('name', ''), candidate.get('link', ''))
        if identifier not in seen:
            seen.add(identifier)
            unique_candidates.append(candidate)
    
    return unique_candidates

def enhance_candidate_data(candidates: List[Dict], parsed_data: Dict) -> List[Dict]:
    """Enhance candidate data with additional processing"""
    enhanced = []
    
    for candidate in candidates:
        # Add email extraction if available
        candidate['email'] = extract_email_from_snippet(candidate.get('snippet', ''))
        
        # Add phone extraction if available
        candidate['phone'] = extract_phone_from_snippet(candidate.get('snippet', ''))
        
        # Enhance location matching
        if not candidate.get('location') or candidate['location'] == "Location Not Specified":
            candidate['location'] = detect_location_from_context(candidate, parsed_data.get('location'))
        
        enhanced.append(candidate)
    
    return enhanced

def extract_email_from_snippet(snippet: str) -> str:
    """Extract email from snippet if available"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, snippet)
    return matches[0] if matches else "Not Available"

def extract_phone_from_snippet(snippet: str) -> str:
    """Extract phone number from snippet if available"""
    phone_patterns = [
        r'\+\d{1,3}\s*\d{10}',
        r'\(\d{3}\)\s*\d{3}-\d{4}',
        r'\d{10}'
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, snippet)
        if matches:
            return matches[0]
    
    return "Not Available"

def detect_location_from_context(candidate: Dict, target_location: str) -> str:
    """Detect location from candidate context"""
    # Check if LinkedIn profile contains location
    if candidate.get('source') == 'LinkedIn' and target_location:
        return target_location
    
    return candidate.get('location', "Location Not Specified")

# --- Enhanced scoring system ---
def score_candidate(candidate: Dict, parsed_data: Dict) -> int:
    """Enhanced candidate scoring algorithm"""
    score = 0
    
    # Get candidate text for analysis
    candidate_text = f"{candidate.get('title', '')} {candidate.get('snippet', '')}".lower()
    
    # Job title match (weight: 8-12 points)
    job_title = parsed_data.get("job_title", "").lower()
    if job_title:
        if job_title in candidate_text:
            score += 12
        else:
            # Partial match for job title keywords
            title_words = job_title.split()
            matches = sum(1 for word in title_words if len(word) > 2 and word in candidate_text)
            score += matches * 3
    
    # Skills match (weight: 4 points per skill)
    skills = parsed_data.get("skills", [])
    matched_skills = 0
    for skill in skills:
        if skill.lower() in candidate_text:
            matched_skills += 1
            score += 4
    
    # Skills bonus for multiple matches
    if matched_skills >= 3:
        score += 5
    
    # Location match (weight: 8 points)
    target_location = parsed_data.get("location", "").lower()
    candidate_location = candidate.get("location", "").lower()
    if target_location and target_location in candidate_location:
        score += 8
    elif target_location and candidate_location and candidate_location in target_location:
        score += 6
    
    # Experience match (weight: 5 points)
    target_exp = parsed_data.get("experience")
    candidate_exp = candidate.get("experience", "")
    if target_exp and candidate_exp and target_exp in candidate_exp:
        score += 5
    
    # Work preference match (weight: 4 points)
    work_pref = parsed_data.get("work_preference")
    if work_pref == "remote" and candidate.get("remote_friendly"):
        score += 4
    
    # Source quality bonus
    source = candidate.get("source", "")
    if source == "LinkedIn":
        score += 6
    elif source == "GitHub":
        score += 4
    elif source == "Portfolio":
        score += 3
    
    # Open to work bonus
    if candidate.get("open_to_work"):
        score += 5
    
    # Contact information bonus
    if candidate.get("email") and candidate["email"] != "Not Available":
        score += 3
    if candidate.get("phone") and candidate["phone"] != "Not Available":
        score += 2
    
    return min(score, 50)  # Cap at 50

def get_match_category(score: int) -> str:
    """Enhanced match categorization"""
    if score >= 35:
        return "🔥 Excellent Match"
    elif score >= 25:
        return "✅ Good Match"
    elif score >= 15:
        return "⚡ Fair Match"
    elif score >= 8:
        return "📋 Basic Match"
    else:
        return "🔍 Potential Match"

# --- Enhanced candidate display ---
def display_candidate_card(candidate: Dict, index: int, score: int, match_category: str):
    """Enhanced candidate card display"""
    with st.container():
        # Header with score
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"### {index}. {candidate.get('name', 'Name Not Available')}")
            st.markdown(f"**{candidate.get('source', 'Unknown')} Profile**")
        
        with col2:
            st.markdown(f"**{match_category}**")
            st.progress(min(score / 50, 1.0))
        
        with col3:
            st.markdown(f"**Score: {score}/50**")
            if candidate.get('open_to_work'):
                st.success("🚀 Open to Work")
        
        # Profile link
        if candidate.get('link'):
            st.markdown(f"[🔗 View Profile]({candidate['link']})")
        
        # Description
        description = candidate.get('description', '')
        if description:
            st.markdown(f"**Description:** {description}")
        
        # Details in columns
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            location = candidate.get('location', 'Not Specified')
            st.markdown(f"**📍 Location:** {location}")
            
            experience = candidate.get('experience', 'Not Specified')
            st.markdown(f"**⏱️ Experience:** {experience}")
        
        with detail_col2:
            skills = candidate.get('skills', [])
            if skills:
                skills_display = ", ".join(skills[:4])
                if len(skills) > 4:
                    skills_display += f" +{len(skills)-4} more"
                st.markdown(f"**🛠️ Skills:** {skills_display}")
            
            if candidate.get('remote_friendly'):
                st.markdown("**💼 Remote Friendly:** ✅")
        
        with detail_col3:
            email = candidate.get('email', 'Not Available')
            if email != 'Not Available':
                st.markdown(f"**📧 Email:** {email}")
            
            phone = candidate.get('phone', 'Not Available')
            if phone != 'Not Available':
                st.markdown(f"**📱 Phone:** {phone}")
        
        st.markdown("---")

# --- Streamlit UI ---
st.set_page_config(page_title="AI Recruiter Pro", page_icon="🤝", layout="wide")

# Enhanced CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 20px;
}
.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 1.2em;
    margin-bottom: 30px;
}
.requirement-card {
    background: rgba(255, 255, 255, 0.05);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤝 AI Recruiter Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced candidate search powered by SERP API & GPT-4</p>', unsafe_allow_html=True)

# Enhanced sidebar
st.sidebar.header("💡 Advanced Search Guide")
st.sidebar.markdown("""
### 🎯 **Optimization Tips:**
- **Specific job titles** work best (e.g., "Senior React Developer")
- **Include 3-5 key skills** for better matching
- **Mention experience range** (2-4 years, 5+ years)
- **Be specific about location** (Mumbai, Singapore, Remote)
- **Add company preferences** (startup, MNC, fintech)

### 📝 **Enhanced Examples:**
- *"Senior Python developer with Django, FastAPI, 5+ years, Mumbai, fintech experience"*
- *"React Native mobile developer, 3-4 years, Bangalore, remote friendly"*
- *"DevOps engineer with AWS, Docker, Kubernetes, 6+ years, Singapore"*
- *"Data scientist with Python, ML, deep learning, 4+ years, startup experience"*

### 🚀 **New Features:**
- ✅ SERP API integration for comprehensive search
- ✅ GPT-4 powered query understanding
- ✅ Advanced location matching with GPT-3.5
- ✅ Multi-source candidate discovery
- ✅ Enhanced skill extraction
- ✅ Smart relevance scoring
- ✅ Open-to-work detection

### 🔧 **Configuration:**
- Requires SERP API key and Azure OpenAI credentials
- Set environment variables in .env file
- Supports multiple search strategies
""")

# Detect user location
user_location = detect_user_location()
if user_location:
    st.sidebar.success(f"📍 Detected location: {user_location}")

# Main search interface
st.markdown("## 🔍 Describe Your Ideal Candidate")

# Enhanced input section
user_query = st.text_area(
    "Enter your recruitment requirements:",
    placeholder="e.g., Senior Python developer with Django, 5+ years experience, Mumbai, remote work preferred",
    height=100,
    help="Be as specific as possible. Include job title, skills, experience, location, and any preferences."
)

# Set default search options (no UI for advanced options)
max_results = 10
search_depth = "Deep"
preferred_sources = ["LinkedIn"]

# Search button
if st.button("🚀 Find Candidates", type="primary", use_container_width=True):
    if not user_query.strip():
        st.error("⚠️ Please enter your recruitment requirements.")
    else:
        # Initialize session state for results
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None
        
        with st.spinner("🤖 Analyzing your requirements..."):
            # Parse the query
            parsed_data = parse_recruiter_query(user_query)
            
            if "error" in parsed_data:
                st.error(f"❌ Error parsing query: {parsed_data['error']}")
            else:
                # Display parsed requirements
                st.success("✅ Requirements understood!")
                
                # Show parsed data in an organized way
                st.markdown("### 📋 Extracted Requirements")
                
                req_col1, req_col2, req_col3 = st.columns(3)
                
                with req_col1:
                    if parsed_data.get("job_title"):
                        st.markdown(f'<div class="requirement-card"><strong>🎯 Job Title:</strong><br>{parsed_data["job_title"]}</div>', unsafe_allow_html=True)
                    
                    if parsed_data.get("experience"):
                        st.markdown(f'<div class="requirement-card"><strong>⏱️ Experience:</strong><br>{parsed_data["experience"]} years</div>', unsafe_allow_html=True)
                
                with req_col2:
                    if parsed_data.get("skills"):
                        skills_display = ", ".join(parsed_data["skills"][:5])
                        if len(parsed_data["skills"]) > 5:
                            skills_display += f" +{len(parsed_data['skills'])-5} more"
                        st.markdown(f'<div class="requirement-card"><strong>🛠️ Skills:</strong><br>{skills_display}</div>', unsafe_allow_html=True)
                    
                    if parsed_data.get("location"):
                        st.markdown(f'<div class="requirement-card"><strong>📍 Location:</strong><br>{parsed_data["location"]}</div>', unsafe_allow_html=True)
                
                with req_col3:
                    if parsed_data.get("work_preference"):
                        st.markdown(f'<div class="requirement-card"><strong>💼 Work Style:</strong><br>{parsed_data["work_preference"].title()}</div>', unsafe_allow_html=True)
                    
                    if parsed_data.get("industry"):
                        st.markdown(f'<div class="requirement-card"><strong>🏢 Industry:</strong><br>{parsed_data["industry"].title()}</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Fetch candidates
                with st.spinner("🔍 Searching for candidates across multiple sources..."):
                    candidates = fetch_candidates_serp(parsed_data)
                    
                    if not candidates:
                        st.warning("⚠️ No candidates found. Try adjusting your search criteria.")
                    else:
                        # Score and sort candidates
                        scored_candidates = []
                        for candidate in candidates:
                            score = score_candidate(candidate, parsed_data)
                            match_category = get_match_category(score)
                            scored_candidates.append((candidate, score, match_category))
                        
                        # Sort by score
                        scored_candidates.sort(key=lambda x: x[1], reverse=True)
                        
                        # Limit results
                        scored_candidates = scored_candidates[:max_results]
                        
                        st.session_state.search_results = (scored_candidates, parsed_data)

# Display results if available
if st.session_state.get('search_results'):
    scored_candidates, parsed_data = st.session_state.search_results
    
    # Results header
    st.markdown("## 🎯 Candidate Results")
    
    # Results summary
    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
    
    with result_col1:
        st.metric("Total Found", len(scored_candidates))
    
    with result_col2:
        excellent_matches = sum(1 for _, score, _ in scored_candidates if score >= 35)
        st.metric("Excellent Matches", excellent_matches)
    
    with result_col3:
        good_matches = sum(1 for _, score, _ in scored_candidates if 25 <= score < 35)
        st.metric("Good Matches", good_matches)
    
    with result_col4:
        avg_score = sum(score for _, score, _ in scored_candidates) / len(scored_candidates) if scored_candidates else 0
        st.metric("Average Score", f"{avg_score:.1f}/50")
    
    # Filter options
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        min_score_filter = st.slider("Minimum Score", 0, 50, 0)
    
    with filter_col2:
        source_filter = st.multiselect(
            "Filter by Source", 
            list(set(candidate.get('source', 'Unknown') for candidate, _, _ in scored_candidates)),
            default=[]
        )
    
    with filter_col3:
        location_filter = st.multiselect(
            "Filter by Location",
            list(set(candidate.get('location', 'Unknown') for candidate, _, _ in scored_candidates)),
            default=[]
        )
    
    # Apply filters
    filtered_candidates = []
    for candidate, score, match_category in scored_candidates:
        if score < min_score_filter:
            continue
        if source_filter and candidate.get('source', 'Unknown') not in source_filter:
            continue
        if location_filter and candidate.get('location', 'Unknown') not in location_filter:
            continue
        filtered_candidates.append((candidate, score, match_category))
    
    st.markdown(f"### Showing {len(filtered_candidates)} candidates")
    
    # Export options
    export_col1, export_col2 = st.columns([1, 4])
    
    with export_col1:
        if st.button("📊 Export to CSV"):
            # Create CSV data
            csv_data = []
            for candidate, score, match_category in filtered_candidates:
                csv_data.append({
                    'Name': candidate.get('name', ''),
                    'Score': score,
                    'Match Category': match_category.replace('🔥 ', '').replace('✅ ', '').replace('⚡ ', '').replace('📋 ', '').replace('🔍 ', ''),
                    'Source': candidate.get('source', ''),
                    'Location': candidate.get('location', ''),
                    'Experience': candidate.get('experience', ''),
                    'Skills': ', '.join(candidate.get('skills', [])),
                    'Email': candidate.get('email', ''),
                    'Phone': candidate.get('phone', ''),
                    'Profile Link': candidate.get('link', ''),
                    'Open to Work': candidate.get('open_to_work', False)
                })
            
            import pandas as pd
            df = pd.DataFrame(csv_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"candidates_{parsed_data.get('job_title', 'search').replace(' ', '_')}.csv",
                mime="text/csv"
            )
    
    # Display candidates
    for i, (candidate, score, match_category) in enumerate(filtered_candidates, 1):
        display_candidate_card(candidate, i, score, match_category)
    
    # Pagination for large results
    if len(filtered_candidates) > 20:
        st.markdown("### 📄 Need more results?")
        if st.button("Load More Candidates"):
            st.info("Implement pagination logic here for better UX")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🤝 <strong>AI Recruiter Pro</strong> | Powered by SERP API & GPT-4 | Built with ❤️ using Streamlit</p>
    <p><small>⚡ Fast • 🎯 Accurate • 🔒 Secure</small></p>
</div>
""", unsafe_allow_html=True)

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)