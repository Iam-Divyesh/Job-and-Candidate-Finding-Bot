# nlp_parser.py

import os
import json
import re
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize client with error handling
try:
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
except Exception as e:
    print(f"Failed to initialize Azure OpenAI client: {e}")
    client = None

def validate_json_response(content):
    """Clean and validate JSON response from GPT"""
    # Remove markdown code blocks
    if content.startswith("```json"):
        content = content[7:-3].strip()
    elif content.startswith("```"):
        content = content[3:-3].strip()
    
    # Remove any text before the first {
    json_start = content.find('{')
    if json_start > 0:
        content = content[json_start:]
    
    # Remove any text after the last }
    json_end = content.rfind('}')
    if json_end > 0:
        content = content[:json_end + 1]
    
    return content

def parse_prompt_gpt(prompt: str) -> dict:
    """Parse job/candidate requirements using GPT with improved error handling"""
    
    if not client:
        print("Azure OpenAI client not available, using fallback")
        return get_fallback_response()
    
    system_msg = """You are an expert job assistant AI. Extract structured data from user queries about job search or hiring.

IMPORTANT: Always return ONLY a valid JSON object with these exact keys:
- query_type: "job" or "candidate" 
- role: specific job title (e.g. "Software Engineer", "Data Analyst", "Python Developer")
- location: city/country or "Remote" (e.g. "Bangalore", "Mumbai", "Remote")
- skills: array of technical skills (e.g. ["Python", "React", "SQL"])
- work_preference: "Remote", "On-site", "Hybrid", or null
- experience: "entry", "mid", or "senior"

Examples:
Input: "I want a Python developer job in Bangalore with 3 years experience"
Output: {"query_type": "job", "role": "Python Developer", "location": "Bangalore", "skills": ["Python"], "work_preference": null, "experience": "mid"}

Input: "Looking for senior React developer for remote position"
Output: {"query_type": "job", "role": "React Developer", "location": "Remote", "skills": ["React"], "work_preference": "Remote", "experience": "senior"}"""

    user_msg = f"""Extract information from this job-related query and return ONLY valid JSON:

Query: "{prompt}"

Return only the JSON object, no other text."""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.2,
            max_tokens=500
        )

        content = response.choices[0].message.content.strip()
        print(f"Raw GPT response: {content}")  # Debug log
        
        # Clean and validate JSON
        content = validate_json_response(content)
        print(f"Cleaned JSON: {content}")  # Debug log
        
        parsed = json.loads(content)
        
        # Validate required fields
        required_keys = ["query_type", "role", "location", "skills", "work_preference", "experience"]
        for key in required_keys:
            if key not in parsed:
                parsed[key] = None
        
        # Clean up skills array
        if parsed.get('skills') and isinstance(parsed['skills'], list):
            parsed['skills'] = [skill.strip() for skill in parsed['skills'] if skill.strip()]
            parsed['skills'] = list(set(parsed['skills']))  # Remove duplicates
        elif not parsed.get('skills'):
            parsed['skills'] = []
        
        # Validate experience level
        if parsed.get('experience') not in ['entry', 'mid', 'senior']:
            parsed['experience'] = 'entry'
        
        print(f"Final parsed result: {parsed}")  # Debug log
        return parsed

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Content that failed to parse: {content}")
        return get_fallback_response()
    except Exception as e:
        print(f"GPT parsing failed: {e}")
        return get_fallback_response()

def get_fallback_response():
    """Return default response structure when parsing fails"""
    return {
        "query_type": "job",
        "role": None,
        "location": None,
        "skills": [],
        "work_preference": None,
        "experience": "entry"
    }

def extract_with_regex(prompt: str) -> dict:
    """Regex-based fallback parser for when GPT fails"""
    
    parsed = get_fallback_response()
    prompt_lower = prompt.lower()
    
    # Job roles patterns
    job_patterns = {
        'python developer': r'python\s*dev|python\s*engineer|python\s*programmer',
        'java developer': r'java\s*dev|java\s*engineer|java\s*programmer',
        'react developer': r'react\s*dev|react\s*engineer|reactjs',
        'full stack developer': r'full[\s-]*stack|fullstack',
        'data analyst': r'data\s*analyst|data\s*analysis',
        'data scientist': r'data\s*scientist|ml\s*engineer',
        'software engineer': r'software\s*engineer|swe',
        'frontend developer': r'frontend|front[\s-]*end|ui\s*dev',
        'backend developer': r'backend|back[\s-]*end|api\s*dev',
        'devops engineer': r'devops|dev[\s-]*ops|infrastructure',
        'product manager': r'product\s*manager|pm\s*role',
        'ux designer': r'ux\s*design|user\s*experience|ui[\s/]*ux'
    }
    
    # Extract role
    for role, pattern in job_patterns.items():
        if re.search(pattern, prompt_lower):
            parsed['role'] = role.title()
            break
    
    # Common Indian cities
    cities = ['bangalore', 'mumbai', 'delhi', 'hyderabad', 'pune', 'chennai', 
              'kolkata', 'ahmedabad', 'surat', 'jaipur', 'gurgaon', 'noida']
    
    # Extract location
    if 'remote' in prompt_lower:
        parsed['location'] = 'Remote'
        parsed['work_preference'] = 'Remote'
    else:
        for city in cities:
            if city in prompt_lower:
                parsed['location'] = city.title()
                break
    
    # Extract skills
    skill_patterns = {
        'python': r'\bpython\b',
        'java': r'\bjava\b',
        'javascript': r'\bjavascript\b|\bjs\b',
        'react': r'\breact\b|\breactjs\b',
        'angular': r'\bangular\b',
        'vue': r'\bvue\b|\bvuejs\b',
        'node.js': r'\bnode\b|\bnodejs\b',
        'sql': r'\bsql\b|\bmysql\b|\bpostgres\b',
        'mongodb': r'\bmongo\b|\bmongodb\b',
        'aws': r'\baws\b|\bamazon\s*web\s*services\b',
        'docker': r'\bdocker\b',
        'kubernetes': r'\bkubernetes\b|\bk8s\b',
        'git': r'\bgit\b|\bgithub\b',
        'html': r'\bhtml\b',
        'css': r'\bcss\b',
        'tensorflow': r'\btensorflow\b|\btf\b',
        'pytorch': r'\bpytorch\b',
        'machine learning': r'\bml\b|\bmachine\s*learning\b',
        'artificial intelligence': r'\bai\b|\bartificial\s*intelligence\b'
    }
    
    found_skills = []
    for skill, pattern in skill_patterns.items():
        if re.search(pattern, prompt_lower):
            found_skills.append(skill.title())
    
    parsed['skills'] = found_skills
    
    # Extract experience
    experience_patterns = {
        'senior': r'\bsenior\b|\blead\b|\b5\+?\s*years?\b|\bexperienced\b|\b6\+?\s*years?\b',
        'mid': r'\bmid\b|\bintermediate\b|\b2-4\s*years?\b|\b3\s*years?\b|\b4\s*years?\b',
        'entry': r'\bentry\b|\bfresher?\b|\bgraduate\b|\bjunior\b|\b0-2\s*years?\b|\bnew\s*grad\b'
    }
    
    for level, pattern in experience_patterns.items():
        if re.search(pattern, prompt_lower):
            parsed['experience'] = level
            break
    
    # Work preference
    if 'hybrid' in prompt_lower:
        parsed['work_preference'] = 'Hybrid'
    elif 'onsite' in prompt_lower or 'on-site' in prompt_lower:
        parsed['work_preference'] = 'On-site'
    elif 'remote' in prompt_lower:
        parsed['work_preference'] = 'Remote'
    
    # Determine query type
    if any(word in prompt_lower for word in ['hire', 'recruit', 'candidate', 'looking for', 'need']):
        parsed['query_type'] = 'candidate'
    else:
        parsed['query_type'] = 'job'
    
    return parsed

# Test function
def test_parser():
    """Test the parser with sample inputs"""
    test_cases = [
        "I want a Python developer job in Bangalore with 3 years experience",
        "Looking for senior React developer for remote position",
        "We need a data scientist with ML experience in Mumbai",
        "Fresh graduate seeking entry level Java developer role in Pune"
    ]
    
    for test in test_cases:
        print(f"\nInput: {test}")
        result = parse_prompt_gpt(test)
        print(f"Output: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    test_parser()