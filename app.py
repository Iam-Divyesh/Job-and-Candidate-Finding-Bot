import streamlit as st
import requests
import re
import apis  # Your API keys here

# Put logo centered and bigger (no extra div, just st.image)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("headsin.png", width=250)  # Adjust width as needed, centered by default

# Your original title (it will appear below the logo)
st.title("Job Search & Candidate Finder App")
st.markdown("<h6 style='text-align: center;'>Beta Testing Prototype</h6>", unsafe_allow_html=True)
# --- Rest of your existing code ---

# Extract location for Job Seeker
def extract_location(text):
    match = re.search(r'at ([a-zA-Z\s]+)', text.lower())
    return match.group(1).strip() if match else None

# Job Seeker Search using JSearch API
def search_jobs_natural_query(query):
    url = "https://jsearch.p.rapidapi.com/search"
    location = extract_location(query) or "India"
    params = {
        "query": query,
        "num_pages": "1",
        "page": "1",
        "location": location
    }
    headers = {
        "X-RapidAPI-Key": apis.JSEARCH_API,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        jobs = data.get("data", [])[:5]
        if not jobs:
            st.info("No jobs found.")
            return

        st.write(f"### 🔍 Top 5 Job Results for: '{query}'\n")
        for idx, job in enumerate(jobs, start=1):
            st.markdown(f"**{idx}. {job.get('job_title', 'N/A')}**")
            st.write(f"Company: {job.get('employer_name', 'N/A')}")
            st.write(f"Location: {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}")
            apply_link = job.get('job_apply_link', '#')
            st.markdown(f"[🔗 Apply here]({apply_link})")
            if apply_link != '#':
                ref = re.findall(r'https?://([^/]+)/', apply_link)
                reference = ref[0] if ref else "Unknown"
            else:
                reference = "Unknown"
            st.write(f"Reference: {reference}\n")
            st.markdown("---")
    else:
        st.error(f"Error {response.status_code}: {response.text}")

# Extract role and location for Recruiter
def extract_role_and_location(query):
    location_match = re.search(r'from ([a-zA-Z\s]+)', query.lower())
    location = location_match.group(1).strip() if location_match else None
    role = re.sub(r'from [a-zA-Z\s]+', '', query, flags=re.IGNORECASE).strip()
    return role, location

# Recruiter Search with SerpAPI (fixed to fetch candidate profiles)
def search_candidates_serpapi(query):
    serpapi_key = apis.SERP_API
    role, location = extract_role_and_location(query)

    base_sites = "(site:linkedin.com/in OR site:upwork.com/freelancers OR site:freelancer.com/users)"
    search_query = f"{role} {base_sites}"
    if location:
        search_query += f" {location}"

    url = "https://serpapi.com/search.json"
    params = {
        "q": search_query,
        "hl": "en",
        "api_key": serpapi_key
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        results = data.get("organic_results", [])[:5]
        return results
    else:
        st.error(f"SerpAPI Error {response.status_code}: {response.text}")
        return None

# Display candidate profiles
def display_profiles_serpapi(results):
    if not results:
        st.info("No candidate profiles found.")
        return

    st.write("### 🔍 Top Candidate Profiles Found:")
    for idx, result in enumerate(results, start=1):
        title = result.get("title", "N/A")
        snippet = result.get("snippet", "")
        link = result.get("link", "#")
        if link != '#':
            ref = re.findall(r'https?://([^/]+)/', link)
            reference = ref[0] if ref else "Unknown"
        else:
            reference = "Unknown"

        st.markdown(f"**{idx}. {title}**")
        st.write(f"Skill Keywords: {snippet}")
        st.markdown(f"[🔗 View Profile]({link})")
        st.write(f"Reference: {reference}\n")
        st.markdown("---")

# Streamlit UI
tab1, tab2 = st.tabs(["Job Seeker", "Recruiter"])

with tab1:
    st.header("Job Seeker")
    job_query = st.text_input("Enter your job search query:", placeholder="I am looking for graphic design job at Surat")
    if st.button("Search Jobs", key="job_seeker"):
        if not job_query.strip():
            st.warning("Please enter your job search query.")
        else:
            search_jobs_natural_query(job_query)

with tab2:
    st.header("Recruiter")
    recruiter_query = st.text_input("Enter your candidate search query:", placeholder="I am looking for graphic designer with 2 years experience from Surat")
    if st.button("Search Candidates", key="recruiter"):
        if not recruiter_query.strip():
            st.warning("Please enter your candidate search query.")
        else:
            results = search_candidates_serpapi(recruiter_query)
            display_profiles_serpapi(results)
