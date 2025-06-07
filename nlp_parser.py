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
    print("✅ Azure OpenAI client initialized successfully")
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
        content = re.sub(r'//.*?\n', '', content)  # Remove single-line comments
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)  # Remove multi-line comments
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        
        # Try to parse
        parsed = json.loads(content)
        return parsed, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"
    except Exception as e:
        return None, f"Content cleaning error: {e}"

def get_fallback_response():
    """Return a fallback response when AI parsing fails"""
    return {
        "job_titles": [],
        "locations": [],
        "companies": [],
        "skills": [],
        "experience_level": "not_specified",
        "salary_range": {
            "min": None,
            "max": None,
            "currency": "USD"
        },
        "job_type": "not_specified",
        "industry": "not_specified",
        "education_level": "not_specified",
        "query_intent": "job_search",
        "urgency": "normal",
        "remote_preference": "not_specified",
        "error": "AI parsing unavailable - using fallback response"
    }

def parse_prompt_gpt(prompt: str) -> dict:
    """Enhanced GPT-only parser with better prompting and error handling"""
    
    if not client:
        print("❌ Azure OpenAI client not available")
        return get_fallback_response()
    
    # Enhanced system message with more examples and clearer instructions
    system_msg = """You are an expert AI assistant specializing in job market analysis. Your task is to extract structured information from job-related queries with high accuracy.

CRITICAL INSTRUCTIONS:
1. Always respond with valid JSON only - no explanations or additional text
2. Extract all relevant job-related information from the user's query
3. Use null for missing information, don't make assumptions
4. Standardize job titles, locations, and skills to common industry terms
5. Infer query intent based on context (job_search, career_advice, salary_inquiry, etc.)

RESPONSE FORMAT (JSON):
{
    "job_titles": ["Software Engineer", "Data Scientist"],
    "locations": ["San Francisco", "Remote"],
    "companies": ["Google", "Microsoft"],
    "skills": ["Python", "Machine Learning", "SQL"],
    "experience_level": "mid_level", // entry_level, mid_level, senior_level, executive, not_specified
    "salary_range": {
        "min": 80000,
        "max": 120000,
        "currency": "USD"
    },
    "job_type": "full_time", // full_time, part_time, contract, internship, freelance, not_specified
    "industry": "technology", // technology, healthcare, finance, education, retail, etc.
    "education_level": "bachelors", // high_school, associates, bachelors, masters, phd, not_specified
    "query_intent": "job_search", // job_search, career_advice, salary_inquiry, skill_development, etc.
    "urgency": "normal", // low, normal, high, immediate
    "remote_preference": "hybrid" // remote, onsite, hybrid, not_specified
}

EXAMPLES:
Query: "Looking for Python developer jobs in NYC with 5+ years experience"
Response: {
    "job_titles": ["Python Developer", "Software Engineer"],
    "locations": ["New York City"],
    "companies": [],
    "skills": ["Python"],
    "experience_level": "senior_level",
    "salary_range": {"min": null, "max": null, "currency": "USD"},
    "job_type": "not_specified",
    "industry": "technology",
    "education_level": "not_specified",
    "query_intent": "job_search",
    "urgency": "normal",
    "remote_preference": "not_specified"
}

Query: "What's the average salary for data scientists at Google?"
Response: {
    "job_titles": ["Data Scientist"],
    "locations": [],
    "companies": ["Google"],
    "skills": ["Data Science", "Analytics"],
    "experience_level": "not_specified",
    "salary_range": {"min": null, "max": null, "currency": "USD"},
    "job_type": "not_specified",
    "industry": "technology",
    "education_level": "not_specified",
    "query_intent": "salary_inquiry",
    "urgency": "normal",
    "remote_preference": "not_specified"
}"""

    user_msg = f"""Parse this job-related query and extract structured information:

"{prompt}"

Return only valid JSON with the extracted information."""

    try:
        # Make API call with enhanced parameters
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,  # Low temperature for consistent parsing
            max_tokens=1000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract and validate response
        content = response.choices[0].message.content.strip()
        
        # Clean and validate JSON
        parsed_data, error = validate_and_clean_json(content)
        
        if error:
            print(f"⚠️ JSON validation error: {error}")
            print(f"Raw response: {content}")
            return get_fallback_response()
        
        # Validate required fields and structure
        validated_data = validate_response_structure(parsed_data)
        
        print("✅ Successfully parsed job query")
        return validated_data
        
    except Exception as e:
        print(f"❌ Error in GPT parsing: {e}")
        return get_fallback_response()

def validate_response_structure(data):
    """Validate and ensure proper structure of parsed response"""
    
    # Default structure
    default_response = {
        "job_titles": [],
        "locations": [],
        "companies": [],
        "skills": [],
        "experience_level": "not_specified",
        "salary_range": {
            "min": None,
            "max": None,
            "currency": "USD"
        },
        "job_type": "not_specified",
        "industry": "not_specified",
        "education_level": "not_specified",
        "query_intent": "job_search",
        "urgency": "normal",
        "remote_preference": "not_specified"
    }
    
    # Merge with parsed data, keeping defaults for missing fields
    for key, default_value in default_response.items():
        if key not in data:
            data[key] = default_value
        elif key == "salary_range" and isinstance(data[key], dict):
            # Ensure salary_range has required fields
            for salary_key, salary_default in default_value.items():
                if salary_key not in data[key]:
                    data[key][salary_key] = salary_default
    
    # Validate enum values
    valid_experience_levels = ["entry_level", "mid_level", "senior_level", "executive", "not_specified"]
    if data["experience_level"] not in valid_experience_levels:
        data["experience_level"] = "not_specified"
    
    valid_job_types = ["full_time", "part_time", "contract", "internship", "freelance", "not_specified"]
    if data["job_type"] not in valid_job_types:
        data["job_type"] = "not_specified"
    
    valid_education_levels = ["high_school", "associates", "bachelors", "masters", "phd", "not_specified"]
    if data["education_level"] not in valid_education_levels:
        data["education_level"] = "not_specified"
    
    valid_urgency_levels = ["low", "normal", "high", "immediate"]
    if data["urgency"] not in valid_urgency_levels:
        data["urgency"] = "normal"
    
    valid_remote_preferences = ["remote", "onsite", "hybrid", "not_specified"]
    if data["remote_preference"] not in valid_remote_preferences:
        data["remote_preference"] = "not_specified"
    
    # Ensure lists are actually lists
    list_fields = ["job_titles", "locations", "companies", "skills"]
    for field in list_fields:
        if not isinstance(data[field], list):
            data[field] = []
    
    return data

def parse_job_query(query: str, use_fallback: bool = False) -> dict:
    """Main function to parse job-related queries"""
    
    if not query or not query.strip():
        return {
            **get_fallback_response(),
            "error": "Empty query provided"
        }
    
    if use_fallback or not client:
        print("ℹ️ Using fallback parsing (regex-based)")
        return parse_with_regex(query)
    
    print(f"🔍 Parsing query: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    # Try GPT parsing first
    result = parse_prompt_gpt(query)
    
    # If GPT parsing fails, fall back to regex
    if "error" in result:
        print("⚠️ GPT parsing failed, falling back to regex parsing")
        return parse_with_regex(query)
    
    return result

def parse_with_regex(query: str) -> dict:
    """Fallback regex-based parsing for basic information extraction"""
    
    result = get_fallback_response()
    query_lower = query.lower()
    
    # Extract common job titles
    job_titles = []
    title_patterns = [
        r'\b(software engineer|developer|programmer|data scientist|analyst|manager|director|designer|architect)\b',
        r'\b(python|java|javascript|react|angular|node\.?js) (developer|engineer)\b',
        r'\b(frontend|backend|full.?stack|devops|qa|test) (engineer|developer)\b',
        r'\b(product|project|technical) (manager|lead)\b'
    ]
    
    for pattern in title_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            if isinstance(match, tuple):
                title = ' '.join(match).title()
            else:
                title = match.title()
            if title not in job_titles:
                job_titles.append(title)
    
    # Extract locations
    locations = []
    location_patterns = [
        r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\b(New York|San Francisco|Los Angeles|Chicago|Boston|Seattle|Austin|Denver|Atlanta|Miami)\b',
        r'\b(NYC|SF|LA|DC)\b',
        r'\b(remote|work from home|wfh)\b'
    ]
    
    for pattern in location_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            if match.lower() in ['remote', 'work from home', 'wfh']:
                result["remote_preference"] = "remote"
            else:
                locations.append(match.strip())
    
    # Extract experience level
    if re.search(r'\b(entry.?level|junior|fresh|graduate|0.?2 years?)\b', query_lower):
        result["experience_level"] = "entry_level"
    elif re.search(r'\b(senior|lead|principal|5\+|5 or more|5-10|6\+)\b', query_lower):
        result["experience_level"] = "senior_level"
    elif re.search(r'\b(mid.?level|intermediate|2-5|3-7)\b', query_lower):
        result["experience_level"] = "mid_level"
    elif re.search(r'\b(executive|director|vp|c-level)\b', query_lower):
        result["experience_level"] = "executive"
    
    # Extract salary information
    salary_matches = re.findall(r'\$?(\d{2,3})[k,]?\s*-?\s*\$?(\d{2,3})[k,]?', query)
    if salary_matches:
        min_sal, max_sal = salary_matches[0]
        result["salary_range"]["min"] = int(min_sal) * 1000
        result["salary_range"]["max"] = int(max_sal) * 1000
    
    # Determine query intent
    if re.search(r'\b(salary|pay|compensation|wage)\b', query_lower):
        result["query_intent"] = "salary_inquiry"
    elif re.search(r'\b(advice|guidance|help|should|career)\b', query_lower):
        result["query_intent"] = "career_advice"
    elif re.search(r'\b(learn|skill|training|course)\b', query_lower):
        result["query_intent"] = "skill_development"
    else:
        result["query_intent"] = "job_search"
    
    # Determine urgency
    if re.search(r'\b(urgent|asap|immediately|soon|quickly)\b', query_lower):
        result["urgency"] = "high"
    elif re.search(r'\b(whenever|no rush|flexible)\b', query_lower):
        result["urgency"] = "low"
    
    result["job_titles"] = job_titles
    result["locations"] = list(set(locations))  # Remove duplicates
    result["error"] = "Parsed using regex fallback"
    
    return result

# Test function
def test_parser():
    """Test the parser with sample queries"""
    
    test_queries = [
        "Looking for Python developer jobs in San Francisco with 5+ years experience",
        "What's the average salary for data scientists at Google?",
        "Remote React developer positions",
        "Entry level software engineer jobs in NYC",
        "Senior DevOps engineer roles with $120k-150k salary",
        "Career advice for transitioning to data science"
    ]
    
    print("🧪 Testing NLP Parser\n" + "="*50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        result = parse_job_query(query)
        print(f"📊 Result: {json.dumps(result, indent=2)}")
        print("-" * 50)

if __name__ == "__main__":
    # Run tests if executed directly
    test_parser()