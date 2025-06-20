import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 🔐 Apollo API keys from .env
SEARCH_API_KEY = os.getenv("APOLLO_API_KEY")
MATCH_API_KEY = os.getenv("APOLLO_PEOPLE_MATCH_API_KEY")

# 🖥️ Streamlit page setup
st.set_page_config(page_title="Apollo Candidate Finder", layout="wide")
st.title("🔍 Apollo Candidate Finder")

# 🧾 Candidate search form
with st.form("search_form"):
    job_title = st.text_input("Job Title")
    experience = st.number_input("Minimum Experience (Years)", min_value=0, max_value=50, step=1)
    location = st.text_input("Location")
    skills = st.text_input("Skills (comma-separated)")
    submitted = st.form_submit_button("Search Candidates")

# 🔎 Enrichment function using people/match
def enrich_person(apollo_id):
    match_url = "https://api.apollo.io/api/v1/people/match"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": MATCH_API_KEY
    }
    payload = {
        "id": apollo_id,
        "reveal_personal_emails": True,
        "reveal_work_emails": True,
        "reveal_phone_number": True
    }
    res = requests.post(match_url, json=payload, headers=headers)

    # Show debug if it fails
    if res.status_code != 200:
        st.warning(f"Match API error for ID {apollo_id} — Status {res.status_code}")
        st.code(res.text)

    return res.json().get("person", {}) if res.status_code == 200 else {}

# 🔍 Main Search + Display Logic
if submitted:
    with st.spinner("Searching candidates..."):
        search_url = "https://api.apollo.io/api/v1/mixed_people/search"
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "X-Api-Key": SEARCH_API_KEY
        }

        payload = {
            "person_titles": [job_title] if job_title else [],
            "person_locations": [location] if location else [],
            "person_skills": skills.split(",") if skills else [],
            "person_min_years_experience": int(experience),
            "page": 1,
            "per_page": 10
        }

        response = requests.post(search_url, json=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            people = data.get("people", [])

            if not people:
                st.warning("No candidates found.")
            else:
                for person in people:
                    apollo_id = person.get("id")
                    enriched = enrich_person(apollo_id)

                    # Show raw enriched API output for debug
                    with st.expander(f"🔍 Raw JSON for {person.get('name', 'Candidate')}"):
                        st.json(enriched)

                    # Candidate details
                    name = person.get("name", "N/A")
                    title = person.get("title", "N/A")
                    company = person.get("organization", {}).get("name", "N/A")
                    city = person.get("city", "")
                    state = person.get("state", "")
                    country = person.get("country", "")
                    loc = ", ".join(part for part in [city, state, country] if part)

                    # Contact info from enriched
                    email = person.get("email") or "N/A"
                    phone = person.get("phone_number") or "N/A"

                    linkedin = person.get("linkedin_url", "")
                    image = person.get("photo_url") or "https://via.placeholder.com/100x100.png?text=No+Image"

                    # 💳 Candidate Card UI
                    with st.container():
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            st.image(image, width=100)
                        with col2:
                            st.markdown(f"### {name}")
                            st.markdown(f"**{title}** at **{company}**")
                            st.markdown(f"📍 {loc}")
                            st.markdown(f"📧 {email} | 📞 {phone}")
                            if linkedin:
                                st.markdown(f"[🔗 LinkedIn]({linkedin})")
                        st.markdown("---")
        else:
            st.error(f"❌ Failed to fetch candidates — Status Code: {response.status_code}")
            st.code(response.text)
