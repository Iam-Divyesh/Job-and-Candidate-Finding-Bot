import streamlit as st
import requests
import os
import json
import re
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Setup for GPT-3.5-turbo
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # Should be gpt-3.5-turbo

# Initialize OpenAI client
try:
    client = AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=endpoint)
except Exception as e:
    st.error(f"Failed to initialize Azure OpenAI client: {e}")
    client = None

# --- Function to parse recruiter query using GPT-3.5-turbo ---
def parse_recruiter_query(query):
    if not client:
        return {"error": "Azure OpenAI client not available"}
    
    try:
        system_prompt = """You are an expert recruitment assistant that extracts structured information from recruiter queries.
        
        Extract the following fields from the recruiter's input and return ONLY a valid JSON object:
        
        Fields to extract:
        - job_title: ONLY the exact position title they're hiring for (e.g., "Python Developer", "Data Scientist"). 
          DO NOT include phrases like "looking for", "need a", "hiring", etc.
        - skills: Array of required technical skills mentioned (e.g., ["Python", "Django", "SQL"])
        - experience: Required experience in years (numeric value or range)
        - location: ONLY the city, state, or country name (e.g., "Mumbai", "California", "India").
          DO NOT include additional phrases like "with X years experience" or skill descriptions.
        - work_preference: Work mode preference - one of: "remote", "onsite", "hybrid", null
        - job_type: Employment type - one of: "full-time", "part-time", "contract", "internship", null
        
        CRITICAL INSTRUCTIONS:
        1. For job_title, NEVER include phrases like "looking for", "need", "hiring", etc.
        2. For location, ONLY include the city/state/country name, nothing else.
        3. Return ONLY valid JSON without any explanation or additional text.
        4. Use your knowledge to recognize job titles across all industries and domains."""
        
        user_prompt = f"""Extract recruitment information from this query: "{query}"

        Examples of correct extraction:
        
        Input: "We are looking for a Python developer with 3 years experience from Mumbai"
        Output: {{"job_title": "Python Developer", "skills": ["Python"], "experience": "3", "location": "Mumbai", "work_preference": null, "job_type": null}}

        Input: "Need a senior React frontend developer with Redux, TypeScript, 5+ years"
        Output: {{"job_title": "React Frontend Developer", "skills": ["React", "Redux", "TypeScript"], "experience": "5+", "location": null, "work_preference": null, "job_type": null}}
        
        Input: "Hiring Java engineers for our Bangalore office, 2-4 years exp required"
        Output: {{"job_title": "Java Engineer", "skills": ["Java"], "experience": "2-4", "location": "Bangalore", "work_preference": "onsite", "job_type": null}}
        
        Input: "Looking for Python developer from Surat with 3 years of experience who have skills like Python Django Flask"
        Output: {{"job_title": "Python Developer", "skills": ["Python", "Django", "Flask"], "experience": "3", "location": "Surat", "work_preference": null, "job_type": null}}

        Input: "We are looking for a Data Scientist with 2 years of experience from Bangalore"
        Output: {{"job_title": "Data Scientist", "skills": [], "experience": "2", "location": "Bangalore", "work_preference": null, "job_type": null}}

        Input: "Need Python developer with Django, Flask experience, 3+ years, Mumbai"
        Output: {{"job_title": "Python Developer", "skills": ["Python", "Django", "Flask"], "experience": "3+", "location": "Mumbai", "work_preference": null, "job_type": null}}
        
        Input: "Hiring Java engineer for Pune office, 2-4 years experience"
        Output: {{"job_title": "Java Engineer", "skills": ["Java"], "experience": "2-4", "location": "Pune", "work_preference": "onsite", "job_type": null}}

        Input: "Remote React developer needed, 5 years experience, Redux, TypeScript"
        Output: {{"job_title": "React Developer", "skills": ["React", "Redux", "TypeScript"], "experience": "5", "location": null, "work_preference": "remote", "job_type": null}}

        Input: "We are looking for Marketing Manager in Delhi with MBA and 4 years experience"
        Output: {{"job_title": "Marketing Manager", "skills": ["MBA"], "experience": "4", "location": "Delhi", "work_preference": null, "job_type": null}}

        Input: "Need Graphic Designer with Photoshop, Illustrator skills, 2+ years"
        Output: {{"job_title": "Graphic Designer", "skills": ["Photoshop", "Illustrator"], "experience": "2+", "location": null, "work_preference": null, "job_type": null}}

        Now extract from the query: "{query}"
        
        Remember: 
        1. Extract ONLY the job title without any prefixes like "looking for", "need", etc.
        2. Extract ONLY the city/location name without additional text.
        3. Return ONLY valid JSON."""
        
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Set to 0 for more consistent results
            max_tokens=500
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up JSON if needed
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
        
        content = content.strip()
        
        # Parse JSON
        parsed = json.loads(content)
        
        # Minimal post-processing to ensure clean data
        cleaned_result = {
            "job_title": parsed.get("job_title", "").strip() if parsed.get("job_title") else None,
            "skills": [skill.strip() for skill in parsed.get("skills", []) if skill.strip()],
            "experience": str(parsed.get("experience", "")).strip() if parsed.get("experience") else None,
            "location": parsed.get("location", "").strip() if parsed.get("location") else None,
            "work_preference": parsed.get("work_preference"),
            "job_type": parsed.get("job_type"),
            "parsing_status": "success"
        }
        
        # If job title still has prefixes, try one more time with a direct request
        if cleaned_result["job_title"] and any(prefix in cleaned_result["job_title"].lower() for prefix in 
                                              ["looking for", "need", "hiring", "we are", "searching"]):
            
            fix_prompt = f"""The job title "{cleaned_result['job_title']}" still contains prefixes like "looking for", "need", etc.
            
            Extract ONLY the actual job position title without any prefixes.
            
            For example:
            - "Looking for Python Developer" → "Python Developer"
            - "We are hiring Data Scientist" → "Data Scientist"
            - "Need a Java Engineer" → "Java Engineer"
            
            Return ONLY the corrected job title, nothing else."""
            
            fix_response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": fix_prompt}],
                temperature=0.0,
                max_tokens=50
            )
            
            fixed_title = fix_response.choices[0].message.content.strip()
            if fixed_title:
                cleaned_result["job_title"] = fixed_title
        
        # If location has multiple words, try to extract just the city name
        if cleaned_result["location"] and len(cleaned_result["location"].split()) > 1:
            location_prompt = f"""The location "{cleaned_result['location']}" contains multiple words.
            
            Extract ONLY the city/state/country name without any additional text.
            
            For example:
            - "Mumbai with 3 years" → "Mumbai"
            - "Bangalore who knows Python" → "Bangalore"
            
            Return ONLY the corrected location name, nothing else."""
            
            location_response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": location_prompt}],
                temperature=0.0,
                max_tokens=50
            )
            
            fixed_location = location_response.choices[0].message.content.strip()
            if fixed_location:
                cleaned_result["location"] = fixed_location
        
        return cleaned_result
        
    except json.JSONDecodeError as e:
        st.warning(f"JSON parsing error: {e}")
        return {"error": f"JSON parsing error: {e}"}
    except Exception as e:
        st.warning(f"AI parsing error: {e}")
        return {"error": f"AI parsing error: {e}"}

# Add this function to detect user location
def detect_user_location():
    """Detect user's location using IP geolocation"""
    try:
        response = requests.get('https://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            ip_data = response.json()
            city = ip_data.get('city')
            if city:
                return city
        return None
    except Exception as e:
        st.warning(f"Could not detect location: {e}")
        return None

# --- IMPROVED Function to extract candidate info from LinkedIn profile ---
def extract_candidate_info(profile_data):
    """Extract candidate information from LinkedIn profile data with improved parsing"""
    title = profile_data.get("title", "")
    snippet = profile_data.get("snippet", "")
    link = profile_data.get("link", "")
    
    # Initialize candidate info
    candidate_info = {
        "name": "Not Available",
        "image": "https://via.placeholder.com/150x150?text=No+Image",
        "description": snippet if snippet else "No description available",
        "experience": "Not specified",
        "location": "Not specified",
        "email": "Not available"
    }
    
    # IMPROVED: Extract name from title
    if title:
        # Try multiple patterns to extract name
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[-|]',  # Name before dash or pipe
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),',         # Name before comma
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\(',     # Name before parenthesis
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+at\s+',  # Name before "at"
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*LinkedIn', # Name before LinkedIn
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, title)
            if match:
                name = match.group(1).strip()
                if len(name) > 2 and not any(word in name.lower() for word in ['linkedin', 'profile', 'the', 'and']):
                    candidate_info["name"] = name
                    break
        
        # If no pattern worked, try simple split
        if candidate_info["name"] == "Not Available":
            parts = title.split(" - ")[0].split(" | ")[0].strip()
            words = parts.split()
            if len(words) >= 2 and len(words) <= 4:  # Reasonable name length
                potential_name = " ".join(words[:3])  # Take first 3 words max
                if not any(word in potential_name.lower() for word in ['linkedin', 'profile', 'developer', 'engineer', 'manager']):
                    candidate_info["name"] = potential_name
    
    # IMPROVED: Extract experience with better patterns
    combined_text = f"{title} {snippet}".lower()
    
    exp_patterns = [
        r'(\d+)\+\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
        r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
        r'(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
        r'(\d+)\s*to\s*(\d+)\s*(?:years?|yrs?)',
        r'over\s*(\d+)\s*(?:years?|yrs?)',
        r'more\s*than\s*(\d+)\s*(?:years?|yrs?)',
        r'(\d+)\+\s*(?:years?|yrs?)',
        r'(\d+)\s*(?:years?|yrs?)',
    ]
    
    for pattern in exp_patterns:
        match = re.search(pattern, combined_text)
        if match:
            if len(match.groups()) == 2:  # Range pattern
                candidate_info["experience"] = f"{match.group(1)}-{match.group(2)} years"
            else:
                years = match.group(1)
                if '+' in pattern:
                    candidate_info["experience"] = f"{years}+ years"
                else:
                    candidate_info["experience"] = f"{years} years"
            break
    
    # IMPROVED: Extract location with better patterns and filtering
    # Define common Indian cities and international locations
    known_cities = [
        'mumbai', 'delhi', 'bangalore', 'bengaluru', 'pune', 'chennai', 'kolkata', 'hyderabad',
        'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore', 'thane',
        'bhopal', 'visakhapatnam', 'pimpri', 'patna', 'vadodara', 'ghaziabad', 'ludhiana',
        'agra', 'nashik', 'faridabad', 'meerut', 'rajkot', 'kalyan', 'vasai', 'varanasi',
        'srinagar', 'dhanbad', 'jodhpur', 'amritsar', 'raipur', 'allahabad', 'coimbatore',
        'jabalpur', 'gwalior', 'vijayawada', 'madurai', 'guwahati', 'chandigarh', 'hubli',
        'mysore', 'tiruchirappalli', 'bareilly', 'aligarh', 'tiruppur', 'gurgaon', 'salem',
        'mira', 'bhiwandi', 'saharanpur', 'gorakhpur', 'bikaner', 'amravati', 'noida',
        'jamshedpur', 'bhilai', 'cuttack', 'firozabad', 'kochi', 'nellore', 'bhavnagar',
        'dehradun', 'durgapur', 'asansol', 'rourkela', 'nanded', 'kolhapur', 'ajmer',
        'akola', 'gulbarga', 'jamnagar', 'ujjain', 'loni', 'siliguri', 'jhansi',
        'ulhasnagar', 'jammu', 'sangli', 'mangalore', 'erode', 'belgaum', 'ambattur',
        'tirunelveli', 'malegaon', 'gaya', 'jalgaon', 'udaipur', 'maheshtala',
        # International cities
        'london', 'new york', 'san francisco', 'toronto', 'vancouver', 'sydney', 'melbourne',
        'singapore', 'dubai', 'tokyo', 'berlin', 'paris', 'amsterdam', 'zurich', 'stockholm'
    ]
    
    # Skip these words that are often mistaken for locations
    skip_words = [
        'mongodb', 'mongo', 'mysql', 'postgresql', 'oracle', 'redis', 'cassandra',
        'python', 'java', 'javascript', 'react', 'angular', 'node', 'django', 'flask',
        'spring', 'hibernate', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
        'linkedin', 'profile', 'experience', 'years', 'developer', 'engineer', 'manager',
        'senior', 'junior', 'lead', 'team', 'software', 'web', 'mobile', 'full',
        'stack', 'front', 'back', 'end', 'data', 'machine', 'learning', 'artificial',
        'intelligence', 'devops', 'cloud', 'security', 'database', 'analyst', 'scientist'
    ]
    
    location_patterns = [
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*,\s*India\b',
        r'\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\bbased\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\blocation:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'·\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*·',
        r'\|+\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\|+',
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, combined_text, re.IGNORECASE)
        for match in matches:
            location = match.strip().lower()
            # Check if it's a known city and not a technology/skill
            if (location in known_cities or 
                any(city in location for city in known_cities)) and \
               location not in skip_words and \
               not any(skip in location for skip in skip_words):
                candidate_info["location"] = match.strip().title()
                break
        if candidate_info["location"] != "Not specified":
            break
    
    # IMPROVED: Try to extract profile image from LinkedIn
    # LinkedIn profile images are not directly accessible through search results
    # But we can try to construct a generic LinkedIn avatar or use a professional placeholder
    if 'linkedin.com' in link:
        # Use a professional avatar placeholder
        candidate_info["image"] = "https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_1280.png"
    
    # IMPROVED: Email extraction (though rare in public LinkedIn profiles)
    # Most LinkedIn profiles don't show emails publicly, but let's try
    email_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'email:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
        r'contact:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
    ]
    
    for pattern in email_patterns:
        match = re.search(pattern, combined_text, re.IGNORECASE)
        if match:
            email = match.group(1) if len(match.groups()) > 0 else match.group(0)
            # Validate email format
            if '@' in email and '.' in email.split('@')[1]:
                candidate_info["email"] = email
                break
    
    # If no email found, provide alternative contact method
    if candidate_info["email"] == "Not available" and 'linkedin.com' in link:
        candidate_info["email"] = "Contact via LinkedIn"
    
    return candidate_info

# --- Function to fetch LinkedIn profiles via SerpApi ---
def fetch_linkedin_profiles(parsed_data):
    job_title = parsed_data.get("job_title", "")
    location = parsed_data.get("location", "")
    skills = parsed_data.get("skills", [])
    experience = parsed_data.get("experience")
    work_preference = parsed_data.get("work_preference")
    
    # Build search query
    search_query = "site:linkedin.com/in "
    
    if job_title:
        search_query += f'"{job_title}" '
    
    # If work preference is remote and no location specified, use user's location
    if work_preference == "remote" and not location:
        user_location = detect_user_location()
        if user_location:
            location = user_location
            st.info(f"📍 Using your location ({location}) to find remote candidates nearby")
    
    if location:
        search_query += f'"{location}" '
    
    # Add top 3 skills to search query
    if skills and len(skills) > 0:
        for skill in skills[:3]:
            search_query += f'"{skill}" '
    
    # Add experience to search query if available
    if experience:
        if '+' in str(experience):
            exp_num = str(experience).replace('+', '')
            search_query += f'"{exp_num} years" OR "{exp_num}+ years" '
        elif '-' in str(experience):
            search_query += f'"{experience} years" '
        else:
            search_query += f'"{experience} years" '
    
    # Add work preference if specified
    if work_preference:
        search_query += f'"{work_preference}" '
    
    # SerpAPI parameters
    params = {
        "engine": "google",
        "q": search_query.strip(),
        "api_key": os.getenv("SERP_API_KEY"),
        "hl": "en",
        "num": 15
    }
    
    try:
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code == 200:
            results = response.json().get("organic_results", [])
            
            # Post-process results to fix LinkedIn URLs
            for result in results:
                if result.get("link") and "in.linkedin.com" in result["link"]:
                    result["link"] = result["link"].replace("in.linkedin.com", "linkedin.com")
                
                # Extract candidate information
                result["candidate_info"] = extract_candidate_info(result)
            
            return results
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching profiles: {e}")
        return []

# --- Function to score profiles by keyword relevance ---
def score_profile(profile, parsed_data):
    content = (profile.get("title", "") + " " + profile.get("snippet", "")).lower()
    score = 0
    
    # Score based on job title match
    if parsed_data.get("job_title"):
        job_title = parsed_data["job_title"].lower()
        if job_title in content:
            score += 6
        # Partial match for job title
        title_words = job_title.split()
        for word in title_words:
            if len(word) > 2 and word in content:
                score += 2
    
    # Score based on skills match
    if parsed_data.get("skills"):
        for skill in parsed_data["skills"]:
            if skill.lower() in content:
                score += 3
    
    # Score based on location match
    if parsed_data.get("location") and parsed_data["location"].lower() in content:
        score += 4
    
    # Score based on experience match
    if parsed_data.get("experience"):
        exp = str(parsed_data["experience"])
        exp_clean = exp.replace("+", "")
        exp_range = exp.replace("-", r"\s*-\s*")
        
        exp_patterns = [
            exp_clean + r'\s*(?:\+)?\s*(?:years?|yrs?)',
            exp_range + r'\s*(?:years?|yrs?)'
        ]
        
        for pattern in exp_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 3
                break
    
    # Score based on work preference match
    if parsed_data.get("work_preference") and parsed_data["work_preference"].lower() in content:
        score += 2
    
    # Score based on job type match
    if parsed_data.get("job_type") and parsed_data["job_type"].lower() in content:
        score += 2
    
    return score

# --- Function to get match category based on score ---
def get_match_category(score):
    if score >= 15:
        return "🔥 Excellent Match"
    elif score >= 10:
        return "✅ Good Match"
    elif score >= 6:
        return "⚡ Fair Match"
    else:
        return "📋 Basic Match"

# --- Function to display candidate card ---
def display_candidate_card(profile, index):
    """Display enhanced candidate card with profile information"""
    candidate = profile["candidate_info"]
    
    with st.container():
        # Create card-like appearance
        st.markdown("""
        <style>
        .candidate-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # Profile image
            st.image(candidate["image"], width=100)
        
        with col2:
            # Candidate details
            st.markdown(f"### {index}. {candidate['name']}")
            st.markdown(f"**📄 Description:** {candidate['description'][:200]}{'...' if len(candidate['description']) > 200 else ''}")
            
            # Create info columns
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                st.markdown(f"**⏱️ Experience:** {candidate['experience']}")
                st.markdown(f"**📍 Location:** {candidate['location']}")
            
            with info_col2:
                st.markdown(f"**📧 Email:** {candidate['email']}")
                st.markdown(f"**🔗 Profile:** [View LinkedIn]({profile.get('link', '#')})")
        
        with col3:
            # Match score and category
            score = profile["score"]
            st.markdown(f"**Score: {score}/20**")
            st.markdown(f"{profile['match_category']}")
            
            # Progress bar for visual representation
            progress_value = min(score / 20, 1.0)
            st.progress(progress_value)
        
        st.divider()

# --- Streamlit UI ---
st.set_page_config(page_title="AI Recruiter", page_icon="🤝", layout="wide")

st.title("🤝 AI-Powered Recruiter: Smart Candidate Finder")
st.markdown("*Find the perfect candidates using natural language queries*")

# Sidebar
st.sidebar.header("💡 Search Tips")
st.sidebar.markdown("""
**For Best Results:**
- Be specific about job titles
- Mention key technical skills
- Include experience requirements
- Specify location preferences
- Add work mode (remote/onsite/hybrid)

**Examples:**
- *"Looking for Python developer from Mumbai with 3 years Django experience"*
- *"Need remote React developer, 5+ years, TypeScript"*
- *"Data scientist in Bangalore, ML, Python, 2-4 years"*
- *"Marketing Manager in Delhi with MBA and 4 years experience"*
- *"Graphic Designer with Photoshop, Illustrator skills"*

**Remote Work Feature:**
- When searching for remote candidates, we'll use your location to find nearby talent
- This helps find candidates in your timezone and region
- Override by specifying a location in your search query
""")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    recruiter_query = st.text_area(
        "🎯 Describe your ideal candidate:",
        placeholder="e.g., Looking for Data Scientist with 2 years of experience from Bangalore who have knowledge of Python, Machine Learning, Statistics, SQL, AWS",
        height=100
    )

with col2:
    st.markdown("### 🔍 Quick Examples")
    if st.button("Python Developer Example", use_container_width=True):
        st.session_state.example_query = "Looking for Python developer from Mumbai with 3 years Django, Flask experience"
    if st.button("Data Scientist Example", use_container_width=True):
        st.session_state.example_query = "Need Data Scientist in Bangalore, 2+ years, Python, ML, Statistics"
    if st.button("Marketing Manager Example", use_container_width=True):
        st.session_state.example_query = "We are looking for Marketing Manager in Delhi with MBA and 4 years experience"

# Use example if selected
if 'example_query' in st.session_state:
    recruiter_query = st.session_state.example_query
    del st.session_state.example_query

# Search button
if st.button("🔍 Find Candidates", type="primary", use_container_width=True):
    if recruiter_query.strip():
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Parse query
        status_text.text("🤖 Understanding your requirements...")
        progress_bar.progress(25)
        
        parsed_data = parse_recruiter_query(recruiter_query)
        
        # Step 2: Display parsed information
        status_text.text("📋 Analyzing requirements...")
        progress_bar.progress(50)
        
        st.subheader("🎯 Understood Requirements:")
        
        # Create a nice display for parsed data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if parsed_data.get("job_title"):
                st.info(f"**👔 Job Title**\n{parsed_data['job_title']}")
            if parsed_data.get("experience"):
                st.info(f"**⏱️ Experience**\n{parsed_data['experience']} years")
        
        with col2:
            if parsed_data.get("skills"):
                skills_text = ", ".join(parsed_data['skills'][:5])  # Show max 5 skills
                if len(parsed_data['skills']) > 5:
                    skills_text += f" +{len(parsed_data['skills'])-5} more"
                st.info(f"**🛠️ Skills**\n{skills_text}")
            if parsed_data.get("location"):
                st.info(f"**📍 Location**\n{parsed_data['location']}")
        
        with col3:
            if parsed_data.get("work_preference"):
                st.info(f"**💼 Work Mode**\n{parsed_data['work_preference'].title()}")
            if parsed_data.get("job_type"):
                st.info(f"**📊 Job Type**\n{parsed_data['job_type'].title()}")
        
        # Show parsing status
        if parsed_data.get("error"):
            st.warning(f"⚠️ {parsed_data['error']}")
        else:
            st.success("✅ Successfully parsed using AI")
        
        # Step 3: Search for candidates
        status_text.text("🔍 Searching LinkedIn for candidates...")
        progress_bar.progress(75)
        
        results = fetch_linkedin_profiles(parsed_data)
        
        # Step 4: Process and display results
        status_text.text("📊 Analyzing candidate matches...")
        progress_bar.progress(100)
        
        if results:
            # Score and sort profiles
            for result in results:
                score = score_profile(result, parsed_data)
                result["score"] = score
                result["match_category"] = get_match_category(score)
            
            # Sort by score
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.success(f"🎉 Found {len(sorted_results)} candidate profiles!")
            
            # Show match distribution
            match_counts = {}
            for result in sorted_results:
                category = result["match_category"]
                match_counts[category] = match_counts.get(category, 0) + 1
            
            cols = st.columns(len(match_counts))
            for i, (category, count) in enumerate(match_counts.items()):
                with cols[i]:
                    st.metric(category, count)
            
            st.divider()
            
            # Display individual candidate cards
            st.subheader("👥 Candidate Profiles")
            for i, profile in enumerate(sorted_results, 1):
                display_candidate_card(profile, i)
                
                
        else:
            progress_bar.empty()
            status_text.empty()
            st.error("❌ No candidate profiles found. Try refining your search terms or check your SERP API key.")
            
            # Suggestions for better search
            st.markdown("""
            ### 💡 Try these tips:
            - Use more common job titles
            - Include alternative skill names
            - Try broader location terms
            - Check if your SERP API key is valid
            """)
    else:
        st.warning("⚠️ Please describe the candidate you're looking for.")