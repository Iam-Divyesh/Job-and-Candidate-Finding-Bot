import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

st.set_page_config(page_title="Indian Talent Finder", page_icon="🇮🇳")
st.title("🇮🇳 Indian Tech Talent Finder")
st.markdown("Search tech candidates in **Indian cities** using flexible filters.")

# API Key
PDL_API_KEY = os.getenv("PDL_API_KEY")

# --- Form Input ---
with st.form("search_form"):
    st.subheader("🔍 Search Filters")

    job_title = st.text_input("Job Title", placeholder="e.g., Python Developer")
    experience = st.slider("Minimum Experience (in years)", 0, 30, 2)
    city = st.text_input("City (Indian Location)", placeholder="e.g., Surat, Bangalore, Mumbai")
    skills_input = st.text_input("Skills (comma-separated)", placeholder="e.g., Python, Django, REST")
    relax_filters = st.checkbox("🔁 Relax filters if no candidates found", value=True)

    submitted = st.form_submit_button("Search Candidates")

# --- Search Execution ---
if submitted:
    if not PDL_API_KEY:
        st.error("🚫 Missing PDL_API_KEY in environment variables.")
        st.stop()

    # Build filters
    must_filters = [
        {"match": {"location_country": "India"}},
        {"range": {"experience": {"gte": experience}}}
    ]
    should_filters = []

    if job_title:
        should_filters.append({"match": {"job_title": job_title}})
    if city:
        should_filters.append({"match": {"location_locality": city}})
    if skills_input:
        skills_list = [s.strip() for s in skills_input.split(",") if s.strip()]
        if skills_list:
            should_filters.append({"terms": {"skills": skills_list}})

    query = {
        "bool": {
            "must": must_filters,
            "should": should_filters,
            "minimum_should_match": 1 if should_filters else 0
        }
    }

    params = {
        "query": query,
        "size": 20
    }

    with st.spinner(f"🔎 Searching candidates in {city or 'India'}..."):
        try:
            response = requests.post(
                "https://api.peopledatalabs.com/v5/person/search",
                headers={"X-API-Key": PDL_API_KEY},
                json=params
            )
            data = response.json()

            st.markdown("#### 🛠️ Query Sent to API")
            st.code(query, language="json")

            # Found candidates
            if response.status_code == 200 and data.get("data"):
                st.success(f"✅ Found {len(data['data'])} candidates:")
                for person in data["data"]:
                    name = person.get("full_name", "Unknown")
                    location = person.get("location", {})
                    st.expander(f"👤 {name}").markdown(f"""
**📧 Email:** `{person.get('work_email', 'Not Available')}`  
**📞 Phone:** `{person.get('mobile_phone', 'Not Available')}`  
**🏢 Company:** {person.get('job_company_name', 'N/A')}  
**💼 Job Title:** {person.get('job_title', 'N/A')}  
**📍 Location:** {location.get('locality', '')}, {location.get('region', '')}, {location.get('country', '')}  
**🛠️ Skills:** {', '.join(person.get('skills', []))}  
**📅 Experience:** {person.get('experience', [{}])[0].get('years', 'N/A')} years
                    """)

            # No candidates found
            else:
                st.warning("⚠️ No matching candidates found.")
                if relax_filters and len(should_filters) > 0:
                    st.info("🔁 Trying again with relaxed filters...")

                    # Retry with only must filters
                    relaxed_query = {
                        "bool": {
                            "must": must_filters
                        }
                    }
                    relaxed_params = {
                        "query": relaxed_query,
                        "size": 20
                    }
                    relaxed_response = requests.post(
                        "https://api.peopledatalabs.com/v5/person/search",
                        headers={"X-API-Key": PDL_API_KEY},
                        json=relaxed_params
                    )
                    relaxed_data = relaxed_response.json()

                    if relaxed_response.status_code == 200 and relaxed_data.get("data"):
                        st.success(f"✅ Found {len(relaxed_data['data'])} candidates (relaxed filters):")
                        for person in relaxed_data["data"]:
                            name = person.get("full_name", "Unknown")
                            location = person.get("location", {})
                            st.expander(f"👤 {name}").markdown(f"""
**📧 Email:** `{person.get('work_email', 'Not Available')}`  
**📞 Phone:** `{person.get('mobile_phone', 'Not Available')}`  
**🏢 Company:** {person.get('job_company_name', 'N/A')}  
**💼 Job Title:** {person.get('job_title', 'N/A')}  
**📍 Location:** {location.get('locality', '')}, {location.get('region', '')}, {location.get('country', '')}  
**🛠️ Skills:** {', '.join(person.get('skills', []))}  
**📅 Experience:** {person.get('experience', [{}])[0].get('years', 'N/A')} years
                            """)
                    else:
                        st.warning("❌ Still no results after relaxing filters.")
                        st.json(relaxed_data.get("error", relaxed_data))
                else:
                    st.markdown("#### 🔍 API Response:")
                    st.json(data.get("error", data))

        except Exception as e:
            st.error(f"❌ API Request Error: {str(e)}")
