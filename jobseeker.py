import streamlit as st
import requests
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from dotenv import load_dotenv
import os

load_dotenv()

@st.cache_resource
def load_nlp_models():
    classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    model = AutoModelForTokenClassification.from_pretrained("jjzha/dajobbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("jjzha/dajobbert-base-uncased")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return classifier, ner_pipeline

classifier, ner_pipeline = load_nlp_models()

def predict_salary(role, experience, location):
    base_salaries = {
        "developer": {"entry": 4, "mid": 8, "senior": 15},
        "data analyst": {"entry": 5, "mid": 10, "senior": 18},
        "full stack": {"entry": 6, "mid": 12, "senior": 20},
        "default": {"entry": 3, "mid": 7, "senior": 12}
    }
    location_factors = {
        "bangalore": 1.3, "mumbai": 1.25, "delhi": 1.2, "hyderabad": 1.15, "pune": 1.1, "default": 1.0
    }
    role_key = role.lower() if role.lower() in base_salaries else "default"
    location_key = location.lower() if location.lower() in location_factors else "default"
    base = base_salaries[role_key][experience]
    multiplier = location_factors[location_key]
    variation = 1 + (np.random.rand() * 0.3 - 0.15)
    return round(base * multiplier * variation, 1)

def enhanced_relevance_score(job, data):
    text = f"{job.get('job_title', '')} {job.get('job_description', '')}".lower()
    score = 0
    if data['role'].lower() in text:
        score += 5
    score += sum(2 for skill in data['skills'] if skill.lower() in text)
    if data['location'].lower() in text:
        score += 3
    if data['work_preference'].lower() in text:
        score += 2
    if data['job_type'].lower() in text:
        score += 2
    if data['industry'] and data['industry'].lower() in text:
        score += 2
    return score

def is_job_from_platform(job, platform_keyword):
    for field in ['job_apply_link', 'employer_website', 'job_google_link']:
        link = job.get(field, "")
        if link and platform_keyword in link.lower():
            return True
    return False

def detect_platform_label(url):
    if "linkedin.com" in url:
        return "LinkedIn"
    elif "indeed.com" in url:
        return "Indeed"
    elif "internshala.com" in url:
        return "Internshala"
    elif "upwork.com" in url:
        return "Upwork"
    elif "glassdoor.com" in url:
        return "Glassdoor"
    elif "naukri.com" in url:
        return "Naukri"
    elif "monster.com" in url:
        return "Monster"
    elif "angel.co" in url:
        return "AngelList"
    elif "foundit.in" in url:
        return "Foundit"
    elif "shine.com" in url:
        return "Shine"
    elif "timesjobs.com" in url:
        return "TimesJobs"
    elif "fiverr.com" in url:
        return "Fiverr"
    elif "freelancer.com" in url:
        return "Freelancer"
    elif "google_jobs_apply" in url:
        return "Company Site (via Google Jobs)"
    else:
        return "Other"

def ai_job_search(data):
    url = "https://jsearch.p.rapidapi.com/search"
    query = f"{data['role']} in {data['location']}" if data['location'] else data['role']
    params = {
        "query": query,
        "num_pages": "5",
        "page": "1",
        "employment_types": data["job_type"].upper(),
        "remote_jobs_only": "true" if data['work_preference'].lower() == "remote" else "false"
    }
    headers = {
        "X-RapidAPI-Key": os.getenv("JSEARCH_API"),
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        st.error(f"API Error {response.status_code}: {response.text}")
        return [], [], []

    all_jobs = response.json().get("data", [])

    platform = data.get("platform", "all").lower()
    platform_map = {
        "linkedin": "linkedin.com", "indeed": "indeed.com", "glassdoor": "glassdoor.com",
        "monster": "monster.com", "naukri": "naukri.com", "angelist": "angel.co",
        "foundit": "foundit.in", "shine": "shine.com", "timesjobs": "timesjobs.com"
    }

    unsupported_platforms = ["upwork", "internshala", "fiverr", "freelancer"]

    if platform != "all":
        if platform in unsupported_platforms:
            st.warning(f"⚠️ {platform.capitalize()} is not currently supported due to API limitations. Showing jobs from other platforms instead.")
            filtered_jobs = all_jobs
        else:
            keyword = platform_map.get(platform, "")
            filtered_jobs = [job for job in all_jobs if is_job_from_platform(job, keyword)]
    else:
        filtered_jobs = all_jobs

    scored_jobs = [(job, enhanced_relevance_score(job, data)) for job in filtered_jobs]
    scored_jobs.sort(key=lambda x: x[1], reverse=True)

    return all_jobs, [job for job, _ in scored_jobs[:10]], [score for _, score in scored_jobs[:10]]

# --- Streamlit UI ---
st.title("📝 Job Seeker Form – AI Finder Pro")

with st.form("job_form"):
    role = st.text_input("Job Role / Domain", placeholder="e.g. Data Analyst")
    location = st.text_input("Preferred Location", placeholder="e.g. Bangalore")
    min_exp = st.slider("Minimum Experience (years)", 0, 10, 0)
    max_salary = st.number_input("Max Expected Salary (Monthly INR)", min_value=5000, value=50000, step=1000)
    knowledge = st.selectbox("Knowledge Level", ["entry", "mid", "senior"])
    work_preference = st.selectbox("Work Preference", ["Remote", "Onsite", "Hybrid"])
    job_type = st.selectbox("Job Type", ["FULLTIME", "PARTTIME", "INTERNSHIP", "CONTRACT"])
    platform = st.selectbox("Preferred Platform to Search", [
        "All", "LinkedIn", "Indeed", "Glassdoor", "Monster", "Naukri", "AngelList", "Foundit",
        "Shine", "TimesJobs", "Upwork", "Internshala", "Fiverr", "Freelancer"
    ])
    industry = st.text_input("Preferred Industry (optional)", placeholder="e.g. Fintech, Healthcare")
    skills = st.text_input("Key Skills (comma-separated)", placeholder="e.g. Python, SQL, Excel")
    submit = st.form_submit_button("🔍 Find Job")

# --- After form submission ---
if submit:
    st.subheader("🔍 Finding Your Best Fit Jobs...")

    user_data = {
        "role": role,
        "location": location,
        "salary": str(max_salary),
        "experience": knowledge,
        "work_preference": work_preference,
        "job_type": job_type,
        "skills": [s.strip() for s in skills.split(',') if s.strip()],
        "industry": industry,
        "platform": platform
    }

    with st.spinner("Please wait while we fetch and score job listings..."):
        predicted = predict_salary(role, knowledge, location)
        st.info(f"🧮 AI Predicted Salary Range: ₹{predicted - 20000} - ₹{predicted + 20000}/month")

        all_jobs, top_jobs, scores = ai_job_search(user_data)

        # ✅ Save in session_state to access outside
        st.session_state['job_results'] = {
            "user_data": user_data,
            "predicted_salary": predicted,
            "all_jobs": all_jobs,
            "top_jobs": top_jobs,
            "top_scores": scores
        }

# --- Independent UI Block: Show Results ---
if 'job_results' in st.session_state:
    data = st.session_state['job_results']
    user_data = data['user_data']

    st.write(f"✅ From **{len(data['all_jobs'])} jobs**, here are your **Top {len(data['top_jobs'])} most relevant jobs**:")

    # ✅ Checkbox outside form for dynamic display
    show_all = st.checkbox("📋 Show All Jobs", value=False)

    if show_all:
        jobs_to_show = data['all_jobs']
        scores_to_show = [enhanced_relevance_score(job, user_data) for job in jobs_to_show]
    else:
        jobs_to_show = data['top_jobs']
        scores_to_show = data['top_scores']

    # === New sorting dropdown by relevance score ===
    sort_by = st.selectbox("🔃 Sort Jobs By", ["Relevance (High to Low)", "Relevance (Low to High)"], index=0)

    # Zip and sort jobs and scores together
    jobs_and_scores = list(zip(jobs_to_show, scores_to_show))

    if sort_by == "Relevance (High to Low)":
        jobs_and_scores.sort(key=lambda x: x[1], reverse=True)
    else:
        jobs_and_scores.sort(key=lambda x: x[1])

    # Unpack sorted lists safely
    if jobs_and_scores:
        jobs_to_show, scores_to_show = zip(*jobs_and_scores)
    else:
        jobs_to_show, scores_to_show = [], []

    # Display jobs
    for idx, (job, score) in enumerate(zip(jobs_to_show, scores_to_show), 1):
        with st.expander(f"{idx}. {job.get('job_title', 'N/A')} [Relevance Score: {score}]"):
            st.markdown(f"**Company:** {job.get('employer_name', 'N/A')}")
            st.markdown(f"**Location:** {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}")
            st.markdown(f"**Type:** {job.get('job_employment_type', 'N/A')}")
            if job.get("job_min_salary") and job.get("job_max_salary"):
                st.markdown(f"**Salary:** ₹{job['job_min_salary'] / 1000}k – ₹{job['job_max_salary'] / 1000}k")
            else:
                st.markdown("**Salary:** Not Specified")

            apply_link = job.get('job_apply_link', '#')
            source_label = detect_platform_label(apply_link)
            st.markdown(f"**Source:** {source_label}")
            st.markdown(f"[🔗 Apply Here]({apply_link})")
