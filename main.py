try:
    import streamlit as st
    from nlp_parser import parse_job_prompt
    import apis
    import requests

    st.title("🧠 NLP-Powered Job & Candidate Finder")

    tabs = st.tabs(["🔍 Job Seeker", "📌 Recruiter"])

    with tabs[0]:
        st.subheader("Job Seeker Mode")
        prompt = st.text_area("Enter your job search prompt:",
                              placeholder="e.g. Find remote Senior Python Developer roles in European fintech startups...")

        if st.button("Search Jobs", key="job_search"):
            if not prompt.strip():
                st.warning("Please enter a job search prompt.")
            else:
                parsed = parse_job_prompt(prompt)
                st.subheader("🔎 Parsed Job Details")
                st.json(parsed)

                role = parsed.get('role', '')
                location = parsed.get('location', 'India')

                url = "https://jsearch.p.rapidapi.com/search"
                params = {
                    "query": f"{role} in {location}",
                    "num_pages": "1",
                    "page": "1",
                    "location": location,
                    "country": "IN"  # Force country to India if supported
                }

                headers = {
                    "X-RapidAPI-Key": apis.JSEARCH_API,
                    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
                }
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 200:
                    jobs = response.json().get("data", [])[:5]
                    st.subheader("📄 Top Matching Jobs")
                    for idx, job in enumerate(jobs, start=1):
                        st.markdown(f"**{idx}. {job.get('job_title', 'N/A')}**")
                        st.write(f"Company: {job.get('employer_name', 'N/A')}")
                        st.write(f"Location: {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}")
                        apply_link = job.get('job_apply_link', '#')
                        st.markdown(f"[🔗 Apply here]({apply_link})")
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

    with tabs[1]:
        st.subheader("Recruiter Mode")
        prompt = st.text_area("Enter your candidate search prompt:",
                              placeholder="e.g. Find experienced Data Scientist profiles in Bangalore skilled in Python and ML...")

        if st.button("Search Candidates", key="candidate_search"):
            if not prompt.strip():
                st.warning("Please enter a candidate search prompt.")
            else:
                parsed = parse_job_prompt(prompt)
                st.subheader("🔎 Parsed Candidate Criteria")
                st.json(parsed)

                role = parsed.get('role', '')
                location = parsed.get('location', '')

                query = role + " site:linkedin.com/in OR site:upwork.com/freelancers OR site:freelancer.com/users"
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
                        link = result.get("link", "#")
                        st.markdown(f"[🔗 View Profile]({link})")
                else:
                    st.error(f"SerpAPI Error {response.status_code}: {response.text}")

except RuntimeError as e:
    import streamlit as st
    st.error(f"Runtime Error: {str(e)}. If using Codespaces, try restarting the kernel.")
