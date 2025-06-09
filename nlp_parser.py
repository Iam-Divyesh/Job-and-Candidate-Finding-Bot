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
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # Should be gpt-4.1-mini
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize client with error handling
try:
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint
    )
    print("✅ Azure OpenAI client initialized successfully with GPT-4.1-mini")
except Exception as e:
    print(f"❌ Failed to initialize Azure OpenAI client: {e}")
    client = None

def validate_and_clean_json(content):
    """Advanced JSON cleaning and validation with better error handling"""
    try:
        # Remove markdown code blocks and extra content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Find JSON object boundaries more precisely
        brace_count = 0
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(content):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    end_idx = i
                    break
        
        if start_idx != -1 and end_idx != -1:
            content = content[start_idx:end_idx + 1]
        
        # Clean up common JSON issues
        content = re.sub(r'//.*?\n', '', content)  # Remove single-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # Remove multi-line comments
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas before }
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas before ]
        
        # Try to parse
        parsed = json.loads(content)
        
        # Validate that it's a dictionary
        if not isinstance(parsed, dict):
            return None, "Response is not a JSON object"
            
        return parsed, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"
    except Exception as e:
        return None, f"Content cleaning error: {e}"

def get_fallback_response(query_type="jobseeker"):
    """Return a structured fallback response when AI parsing fails"""
    base_response = {
        "role": None,
        "skills": [],
        "experience": "entry",
        "location": None,
        "work_preference": None,
        "parsing_status": "fallback_used"
    }
    
    if query_type == "jobseeker":
        return {
            **base_response,
            "salary": None,
            "job_type": None,
            "industry": None,
            "platform": None
        }
    elif query_type == "recruiter":
        return base_response
    else:  # salary prediction
        return base_response

def normalize_parsed_data(data, query_type="jobseeker"):
    """Normalize and validate parsed data to ensure all required fields are present"""
    
    # Base structure for all types
    normalized = {
        "role": data.get("role"),
        "skills": data.get("skills", []) if isinstance(data.get("skills"), list) else [],
        "experience": data.get("experience", "entry"),
        "location": data.get("location"),
        "work_preference": data.get("work_preference"),
        "parsing_status": "success"
    }
    
    # Add jobseeker-specific fields
    if query_type == "jobseeker":
        normalized.update({
            "salary": data.get("salary"),
            "job_type": data.get("job_type"),
            "industry": data.get("industry"),
            "platform": data.get("platform")
        })
    
    # Normalize experience levels
    exp_map = {
        "0": "entry", "1": "entry", "2": "entry",
        "3": "mid", "4": "mid", "5": "mid",
        "6": "senior", "7": "senior", "8": "senior",
        "9": "lead", "10": "lead", "fresher": "entry",
        "beginner": "entry", "junior": "entry", "intermediate": "mid",
        "experienced": "senior", "expert": "lead", "architect": "lead"
    }
    
    exp_str = str(normalized.get("experience", "")).lower()
    for key, value in exp_map.items():
        if key in exp_str:
            normalized["experience"] = value
            break
    
    # Ensure experience is valid
    if normalized["experience"] not in ["entry", "mid", "senior", "lead"]:
        normalized["experience"] = "entry"
    
    return normalized

def parse_prompt_gpt(prompt: str, query_type: str = "jobseeker") -> dict:
    """Enhanced GPT-4.1-mini parser with improved prompts and validation"""
    
    if not client:
        print("❌ Azure OpenAI client not available")
        return get_fallback_response(query_type)
    
    # Enhanced system messages with more specific instructions
    if query_type == "jobseeker":
        system_msg = """You are an expert job search analyzer. Extract structured information from job seeker queries.

CRITICAL REQUIREMENTS:
1. Respond ONLY with valid JSON - no text before or after
2. Include ALL fields even if null
3. Be thorough in extraction
4. Infer reasonable defaults when appropriate

EXTRACT THESE FIELDS:
- role: Job title/position (extract even partial matches)
- skills: ALL technical skills mentioned (programming languages, frameworks, tools)
- experience: Map to "entry" (0-2 years), "mid" (3-5 years), "senior" (6+ years), "lead" (10+ years)
- location: City/state or "remote"
- salary: Any salary/pay mention (keep original format)
- work_preference: "remote", "onsite", "hybrid" or null
- job_type: "full-time", "part-time", "contract", "internship" or null  
- industry: Business sector if mentioned
- platform: Job site preference if mentioned

REQUIRED JSON STRUCTURE:
{
    "role": "string or null",
    "skills": ["array", "of", "strings"],
    "experience": "entry|mid|senior|lead",
    "location": "string or null",
    "salary": "string or null",
    "work_preference": "string or null",
    "job_type": "string or null",
    "industry": "string or null",
    "platform": "string or null"
}

EXAMPLES:
Input: "I want Python developer job in Mumbai with 4 years experience"
Output: {
    "role": "Python Developer",
    "skills": ["Python"],
    "experience": "mid",
    "location": "Mumbai",
    "salary": null,
    "work_preference": null,
    "job_type": null,
    "industry": null,
    "platform": null
}

Input: "Looking for remote React frontend role with Redux, TypeScript, 6+ years, fintech, full-time, 15-20 LPA"
Output: {
    "role": "Frontend Developer",
    "skills": ["React", "Redux", "TypeScript"],
    "experience": "senior",
    "location": "remote",
    "salary": "15-20 LPA",
    "work_preference": "remote",
    "job_type": "full-time",
    "industry": "fintech",
    "platform": null
}"""

    elif query_type == "recruiter":
        system_msg = """You are an expert recruitment analyzer. Extract candidate requirements from recruiter queries.

CRITICAL REQUIREMENTS:
1. Respond ONLY with valid JSON - no text before or after
2. Include ALL fields even if null
3. Focus on what recruiters are looking for

EXTRACT THESE FIELDS:
- role: Position they're hiring for
- experience: Required experience level
- location: Candidate location requirement
- skills: Required technical skills

REQUIRED JSON STRUCTURE:
{
    "role": "string or null",
    "skills": ["array", "of", "strings"],
    "experience": "entry|mid|senior|lead or null",
    "location": "string or null"
}

EXAMPLES:
Input: "We need senior React developer with 6+ years experience in Bangalore"
Output: {
    "role": "React Developer",
    "skills": ["React"],
    "experience": "senior",
    "location": "Bangalore"
}"""

    else:  # salary prediction
        system_msg = """You are an expert salary analyzer. Extract information needed for salary prediction.

CRITICAL REQUIREMENTS:
1. Respond ONLY with valid JSON - no text before or after
2. Include ALL fields even if null
3. Focus on salary-relevant information

EXTRACT THESE FIELDS:
- role: Job title/position
- experience: Experience level for salary calculation  
- location: Work location affecting salary
- skills: Technical skills that impact salary

REQUIRED JSON STRUCTURE:
{
    "role": "string or null",
    "skills": ["array", "of", "strings"],
    "experience": "entry|mid|senior|lead",
    "location": "string or null"
}"""

    try:
        print(f"🤖 Sending query to GPT-4.1-mini...")
        print(f"Query: {prompt}")
        
        # Make API call with more specific parameters
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Extract structured information from this query: {prompt}"}
            ],
            temperature=0.1,  # Low temperature for consistent output
            max_tokens=1500,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract response content
        content = response.choices[0].message.content.strip()
        print(f"🔍 Raw AI Response: {content}")
        
        # Validate and clean the JSON response
        parsed_data, error = validate_and_clean_json(content)
        
        if error:
            print(f"❌ JSON validation error: {error}")
            print(f"❌ Problematic content: {content}")
            return get_fallback_response(query_type)
        
        # Normalize the parsed data
        normalized_data = normalize_parsed_data(parsed_data, query_type)
        
        print(f"✅ Successfully parsed query as {query_type}")
        print(f"✅ Normalized result: {json.dumps(normalized_data, indent=2)}")
        
        return normalized_data
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return get_fallback_response(query_type)

def determine_query_type(prompt: str) -> str:
    """Determine the type of query based on content analysis"""
    prompt_lower = prompt.lower()
    
    # Keywords for different query types
    recruiter_keywords = [
        "we need", "we are looking", "hiring", "recruit", "candidate", 
        "we want", "our company", "we require", "position available",
        "job opening", "vacancy", "looking for candidates", "need someone"
    ]
    
    salary_keywords = [
        "salary", "pay", "compensation", "how much", "expected salary",
        "salary range", "pay scale", "earnings", "income", "what should i expect",
        "market rate", "typical salary"
    ]
    
    # Check for recruiter queries
    if any(keyword in prompt_lower for keyword in recruiter_keywords):
        return "recruiter"
    
    # Check for salary prediction queries
    if any(keyword in prompt_lower for keyword in salary_keywords):
        return "salary"
    
    # Default to jobseeker
    return "jobseeker"

def process_query(prompt: str) -> dict:
    """Main function to process any type of query with enhanced error handling"""
    try:
        query_type = determine_query_type(prompt)
        print(f"🎯 Detected query type: {query_type}")
        
        result = parse_prompt_gpt(prompt, query_type)
        result["query_type"] = query_type
        
        return result
        
    except Exception as e:
        print(f"❌ Error in process_query: {e}")
        return {
            **get_fallback_response("jobseeker"),
            "query_type": "jobseeker",
            "error": str(e)
        }

def extract_skills_fallback(prompt: str) -> list:
    """Fallback skill extraction using regex patterns"""
    skills = []
    
    # Common programming languages and technologies
    tech_patterns = [
        r'\b(python|java|javascript|js|react|angular|vue|node\.?js|express)\b',
        r'\b(html|css|sql|mongodb|mysql|postgresql|redis|docker|kubernetes)\b',
        r'\b(aws|azure|gcp|git|django|flask|spring|laravel|php)\b',
        r'\b(machine learning|ml|ai|data science|tensorflow|pytorch)\b',
        r'\b(devops|ci/cd|jenkins|ansible|terraform)\b'
    ]
    
    prompt_lower = prompt.lower()
    for pattern in tech_patterns:
        matches = re.findall(pattern, prompt_lower)
        skills.extend([match.title() for match in matches])
    
    return list(set(skills))  # Remove duplicates

# Example usage and testing
if __name__ == "__main__":
    # Comprehensive test cases
    test_queries = [
        "I want a Python developer job in Bangalore with 3 years experience, Django and React skills, full-time, 8-12 LPA",
        "Looking for remote React developer position with Redux, TypeScript skills, hybrid work, fintech industry",
        "We need a senior full-stack developer with MERN stack experience in Mumbai, 6+ years",
        "What's the salary for a data scientist with Python, ML, and 5 years experience in Pune?",
        "Entry level Java developer position in Hyderabad, part-time, learning Spring Boot",
        "Senior DevOps engineer with AWS, Docker, Kubernetes skills in Delhi, startup environment",
        "Frontend developer role with Angular, JavaScript, CSS in Chennai, 2-4 years experience",
        "Looking for machine learning engineer position with TensorFlow, PyTorch, remote work possible"
    ]
    
    print("🚀 Testing Enhanced NLP Parser...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")
        print("-" * 40)
        
        result = process_query(query)
        print(f"Result:")
        print(json.dumps(result, indent=2))
        print("-" * 40)