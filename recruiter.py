import streamlit as st
import requests
import apis

# --- Function to fetch LinkedIn profiles via SerpApi ---
def fetch_linkedin_profiles(job_title, location, keywords, num_results=10):
    search_query = f"site:linkedin.com/in {job_title} {location} " + " ".join(keywords)
    params = {
        "engine": "google",
        "q": search_query,
        "api_key": apis.SERP_API,
        "hl": "en",
        "gl": "in",  # Set to India; change to "us" or others if needed
        "num": num_results
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code == 200:
        return response.json().get("organic_results", [])
    else:
        st.error(f"API Error {response.status_code}: {response.text}")
        return []

# --- Function to score profiles by keyword relevance ---
def score_profile(profile, keywords):
    content = (profile.get("title", "") + " " + profile.get("snippet", "")).lower()
    return sum(1 for kw in keywords if kw.lower() in content)

# --- Streamlit UI ---
st.title("🤝 Recruiter: LinkedIn Candidate Finder")

with st.form("linkedin_form"):
    job_title = st.text_input("Job Title", placeholder="e.g. Data Analyst")
    location = st.text_input("Preferred Location", placeholder="e.g. Bangalore")
    keyword_input = st.text_input("Required Skills (comma-separated)", placeholder="e.g. Python, Excel, SQL")
    num_results = st.slider("Number of Profiles to Fetch", 5, 30, 10)
    submit = st.form_submit_button("🔍 Search LinkedIn Profiles")

# --- Handle Search ---
if submit:
    keywords = [kw.strip() for kw in keyword_input.split(",") if kw.strip()]
    with st.spinner("Fetching LinkedIn profiles..."):
        results = fetch_linkedin_profiles(job_title, location, keywords, num_results)

    if results:
        # Score and sort profiles
        for result in results:
            result["score"] = score_profile(result, keywords)
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        st.success(f"✅ Found {len(sorted_results)} LinkedIn profiles:")
        for i, profile in enumerate(sorted_results, 1):
            st.markdown(f"### {i}. [{profile.get('title', 'LinkedIn Profile')}]({profile.get('link', '#')})")
            st.write(f"**Relevance Score:** {profile['score']}")
            st.write(profile.get("snippet", "No description available"))
            st.markdown("---")
    else:
        st.warning("No profiles found. Try different keywords or location.")
