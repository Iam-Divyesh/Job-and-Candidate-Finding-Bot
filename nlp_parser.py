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
    """Advanced JSON cleaning and validation"""
    try:
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Find JSON object boundaries
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx + 1]
        
        # Remove comments and extra whitespace
        content = re.sub(r'//.*?\n', '', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'\s+', ' ', content)
        
        # Try to parse
        parsed = json.loads(content)
        return parsed, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"
    except Exception as e:
        return None, f"Content cleaning error: {e}"

def get_fallback_response(query_type="jobseeker"):
    """Return a fallback response when AI parsing fails"""
    base_response = {
        "role": None,
        "skills": [],
        "experience": "entry",
        "location": None,
        "work_preference": None,
        "error": "AI parsing unavailable - using fallback response"
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

def parse_prompt_gpt(prompt: str, query_type: str = "jobseeker") -> dict:
    """Enhanced GPT-4.1-mini parser with dynamic requirement recognition"""
    
    if not client:
        print("❌ Azure OpenAI client not available")
        return get_fallback_response(query_type)
    
    # Dynamic system message based on query type
    if query_type == "jobseeker":
        system_msg = """You are an expert AI assistant specializing in job search analysis. Your task is to extract structured information from job seeker queries with high accuracy.

CRITICAL INSTRUCTIONS:
1. Always respond with valid JSON only - no explanations or additional text
2. Extract all relevant job-related information from the user's query
3. Use null for missing information, don't make assumptions
4. Infer missing details intelligently from context

JOBSEEKER REQUIREMENTS TO EXTRACT:
- role: Job title/position (string)
- skills: Technical skills mentioned (array of strings)
- experience: Experience level - "entry", "mid", "senior", or "lead" (string)
- location: Preferred work location or "remote" (string)
- salary: Expected salary range or amount (string)
- work_preference: "remote", "onsite", "hybrid", or null (string)
- job_type: "full-time", "part-time", "contract", "internship", or null (string)
- industry: Target industry if mentioned (string)
- platform: Job search platform preference if mentioned (string)

RESPONSE FORMAT (JSON):
{
    "role": "Software Developer",
    "skills": ["Python", "Django", "React"],
    "experience": "mid",
    "location": "Bangalore",
    "salary": "8-12 LPA",
    "work_preference": "hybrid",
    "job_type": "full-time",
    "industry": "fintech",
    "platform": "naukri"
}

EXAMPLES:
Query: "I want a Python developer job in Bangalore with 3 years experience, Django skills, full-time, 8-12 LPA salary"
Response: {
    "role": "Python Developer",
    "skills": ["Python", "Django"],
    "experience": "mid",
    "location": "Bangalore",
    "salary": "8-12 LPA",
    "work_preference": null,
    "job_type": "full-time",
    "industry": null,
    "platform": null
}

Query: "Looking for remote React developer position with Redux skills, hybrid work, startup environment"
Response: {
    "role": "React Developer",
    "skills": ["React", "Redux"],
    "experience": "entry",
    "location": "remote",
    "salary": null,
    "work_preference": "hybrid",
    "job_type": null,
    "industry": "startup",
    "platform": null
}"""

    elif query_type == "recruiter":
        system_msg = """You are an expert AI assistant specializing in recruitment analysis. Your task is to extract structured information from recruiter queries with high accuracy.

CRITICAL INSTRUCTIONS:
1. Always respond with valid JSON only - no explanations or additional text
2. Extract all relevant candidate requirements from the recruiter's query
3. Use null for missing information, don't make assumptions

RECRUITER REQUIREMENTS TO EXTRACT:
- role: Target job title/position they're hiring for (string)
- experience: Required experience level - "entry", "mid", "senior", or "lead" (string)
- location: Preferred candidate location (string)
- skills: Required technical skills (array of strings)

RESPONSE FORMAT (JSON):
{
    "role": "React Developer",
    "experience": "senior",
    "location": "Mumbai",
    "skills": ["React", "Redux", "JavaScript"]
}

EXAMPLES:
Query: "We need a senior React developer with Redux experience in Mumbai"
Response: {
    "role": "React Developer",
    "experience": "senior",
    "location": "Mumbai",
    "skills": ["React", "Redux"]
}

Query: "Looking for Python data scientist with ML and TensorFlow experience, remote OK"
Response: {
    "role": "Data Scientist",
    "experience": null,
    "location": "remote",
    "skills": ["Python", "Machine Learning", "TensorFlow"]
}"""

    else:  # salary prediction
        system_msg = """You are an expert AI assistant specializing in salary analysis. Your task is to extract structured information for salary prediction with high accuracy.

CRITICAL INSTRUCTIONS:
1. Always respond with valid JSON only - no explanations or additional text
2. Extract role, experience, location, and skills for accurate salary prediction
3. Use null for missing information, don't make assumptions

SALARY PREDICTION REQUIREMENTS TO EXTRACT:
- role: Job title/position (string)
- experience: Experience level - "entry", "mid", "senior", or "lead" (string)
- location: Work location (string)
- skills: Technical skills mentioned (array of strings)

RESPONSE FORMAT (JSON):
{
    "role": "Software Engineer",
    "experience": "mid",
    "location": "Bangalore",
    "skills": ["Python", "React", "Node.js"]
}

EXAMPLES:
Query: "What's the salary for a senior Python developer with Django skills in Mumbai?"
Response: {
    "role": "Python Developer",
    "experience": "senior",
    "location": "Mumbai",
    "skills": ["Python", "Django"]
}

Query: "Expected salary for ML engineer with 2 years experience in Pune"
Response: {
    "role": "ML Engineer",
    "experience": "mid",
    "location": "Pune",
    "skills": ["Machine Learning"]
}"""

    try:
        # Make API call to Azure OpenAI
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            top_p=0.95
        )
        
        # Extract response content
        content = response.choices[0].message.content.strip()
        print(f"🔍 Raw AI Response: {content}")
        
        # Validate and clean the JSON response
        parsed_data, error = validate_and_clean_json(content)
        
        if error:
            print(f"❌ JSON validation error: {error}")
            return get_fallback_response(query_type)
        
        print(f"✅ Successfully parsed query as {query_type}")
        return parsed_data
        
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
        "job opening", "vacancy"
    ]
    
    salary_keywords = [
        "salary", "pay", "compensation", "how much", "expected salary",
        "salary range", "pay scale", "earnings", "income"
    ]
    
    # Check for recruiter queries
    if any(keyword in prompt_lower for keyword in recruiter_keywords):
        return "recruiter"
    
    # Check for salary prediction queries
    if any(keyword in prompt_lower for keyword in salary_keywords):
        return "salary_prediction"
    
    # Default to jobseeker
    return "jobseeker"

def process_query(prompt: str) -> dict:
    """Main function to process any type of query"""
    query_type = determine_query_type(prompt)
    print(f"🎯 Detected query type: {query_type}")
    
    result = parse_prompt_gpt(prompt, query_type)
    result["query_type"] = query_type
    
    return result

# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_queries = [
        "I want a Python developer job in Bangalore with 3 years experience",
        "We need a senior React developer with Redux experience in Mumbai",
        "What's the salary for a data scientist with 5 years experience in Pune?",
        "Looking for remote full-stack developer position with MERN stack skills"
    ]
    
    print("🚀 Testing NLP Parser...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")
        result = process_query(query)
        print(f"Result: {json.dumps(result, indent=2)}")