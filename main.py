import streamlit as st
from nlp_parser import parse_job_prompt
import apis
import requests

st.title("🧠 NLP-Powered Job & Candidate Finder")

tabs = st.tabs(["🔍 Job Seeker", "📌 Recruiter"])

job_data_cache = []

def get_top_relevant_jobs(jobs, role, skills, top_n=10):
    if not jobs:
        return []

    def relevance_score(job):
        score = 0
        title = job.get('job_title', '').lower()
        description = job.get('job_description', '').lower() if 'job_description' in job else ''
        combined_text = title + " " + description

        if role and role.lower() in combined_text:
            score += 5

        if skills:
            for skill in skills:
                if skill.lower() in combined_text:
                    score += 1

        return score

    ranked_jobs = sorted(jobs, key=relevance_score, reverse=True)
    return ranked_jobs[:top_n]

with tabs[0]:
    st.subheader("Job Seeker Mode")
    prompt = st.text_area("Enter your job search prompt:", placeholder="e.g. Find remote Full Stack Developer jobs in Surat...")

    parsed = {}
    filters_visible = False

    if st.button("Search Jobs", key="job_search"):
        if not prompt.strip():
            st.warning("Please enter a job search prompt.")
        else:
            parsed = parse_job_prompt(prompt)
            st.subheader("🔎 Parsed Job Details")
            st.json(parsed)

            role = parsed.get('role', '')
            location = parsed.get('location', 'India')
            skills = parsed.get('skills', [])

            query = f"{role} in {location}"

            url = "https://jsearch.p.rapidapi.com/search"
            params = {
                "query": query,
                "num_pages": "1",
                "page": "1",
                "location": location,
                "country": "IN"
            }

            headers = {
                "X-RapidAPI-Key": apis.JSEARCH_API,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            }

            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                jobs = response.json().get("data", [])
                job_data_cache = jobs

                st.session_state['job_parsed'] = parsed
                st.session_state['job_data'] = job_data_cache
                st.session_state['api_params'] = params
                st.session_state['api_url'] = url
                st.session_state['api_headers'] = headers

                top_jobs = get_top_relevant_jobs(job_data_cache, role, skills)
                st.subheader("📄 Top Matching Jobs")
                for idx, job in enumerate(top_jobs, start=1):
                    st.markdown(f"**{idx}. {job.get('job_title', 'N/A')}**")
                    st.write(f"Company: {job.get('employer_name', 'N/A')}")
                    st.write(f"Location: {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}")
                    st.markdown(f"[🔗 Apply here]({job.get('job_apply_link', '#')})")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

    if 'job_data' in st.session_state and 'job_parsed' in st.session_state:
        filters_visible = True
        parsed = st.session_state['job_parsed']
        job_data_cache = st.session_state['job_data']

    if filters_visible:
        with st.form("job_filters"):
            st.markdown("### 🎨 Filter Results")
            location = st.text_input("Location", value=parsed.get('location', ''))
            experience = st.selectbox("Experience Level", ["Any", "Entry", "Mid", "Senior"])
            job_type = st.selectbox("Job Type", ["Any", "Full-time", "Part-time", "Contract", "Internship"])
            work_pref = st.selectbox("Work Preference", ["Any", "Remote", "Hybrid", "On-site"])
            salary = st.slider("Salary Range (LPA)", 1, 50, (1, 50))
            remote_only = st.checkbox("Remote Only")
            date_posted = st.selectbox("Date Posted", ["Any", "Last 24 hours", "Last 7 days", "Last 30 days"])

            submit_filters = st.form_submit_button("Save and Search")

            if submit_filters:
                role = parsed.get('role')
                skills = parsed.get('skills')

                if location != st.session_state['job_parsed'].get('location', ''):
                    st.info("Location changed. Making a new API call...")
                    new_query = f"{role} in {location}"
                    params = st.session_state['api_params']
                    url = st.session_state['api_url']
                    headers = st.session_state['api_headers']

                    params["query"] = new_query
                    params["location"] = location

                    response = requests.get(url, headers=headers, params=params)
                    if response.status_code == 200:
                        jobs = response.json().get("data", [])
                        job_data_cache = jobs
                        st.session_state['job_data'] = job_data_cache
                        st.session_state['job_parsed']['location'] = location
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")

                job_data_cache = st.session_state['job_data']
                top_jobs = get_top_relevant_jobs(job_data_cache, role, skills)
                st.subheader("📄 Filtered Top Matching Jobs")
                for idx, job in enumerate(top_jobs, start=1):
                    st.markdown(f"**{idx}. {job.get('job_title', 'N/A')}**")
                    st.write(f"Company: {job.get('employer_name', 'N/A')}")
                    st.write(f"Location: {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}")
                    st.markdown(f"[🔗 Apply here]({job.get('job_apply_link', '#')})")

with tabs[1]:
    st.subheader("Recruiter Mode")
    prompt = st.text_area("Enter your candidate search prompt:", placeholder="e.g. Find experienced Data Scientist profiles in Bangalore skilled in Python and ML...")

    if st.button("Search Candidates", key="candidate_search"):
        if not prompt.strip():
            st.warning("Please enter a candidate search prompt.")
        else:
            parsed = parse_job_prompt(prompt)
            st.subheader("🔎 Parsed Candidate Criteria")
            st.json(parsed)

            role = parsed.get('role', '')
            location = parsed.get('location', '')

            query = f"{role} site:linkedin.com/in OR site:upwork.com/freelancers OR site:freelancer.com/users"
            if location:
                query += f" {location}"

            serp_url = "https://serpapi.com/search.json"
            serp_params = {
                "q": query,
                "hl": "en",
                "api_key": apis.SERP_API
            }

            response = requests.get(serp_url, params=serp_params)
            if response.status_code == 200:
                results = response.json().get("organic_results", [])[:5]
                st.subheader("👤 Top Candidate Profiles")
                for idx, result in enumerate(results, start=1):
                    st.markdown(f"**{idx}. {result.get('title', 'N/A')}**")
                    st.write(result.get("snippet", ""))
                    st.markdown(f"[🔗 View Profile]({result.get('link', '#')})")
            else:
                st.error(f"SerpAPI Error {response.status_code}: {response.text}")
