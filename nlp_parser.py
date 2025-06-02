# nlp_parser.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from difflib import SequenceMatcher

MODEL_NER = "Jean-Baptiste/roberta-large-ner-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NER)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NER)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
classifier_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = [
    "job search", "hiring", "developer", "data analyst", "full stack", "React", "Python", "Surat", "remote"
]

known_skills = ["python", "react", "full stack", "machine learning", "sql", "javascript", "html", "css"]
known_locations = ["surat", "mumbai", "bangalore", "delhi", "remote", "pune"]

def parse_job_prompt(prompt):
    ner_results = ner_pipeline(prompt)
    classification = classifier_pipeline(prompt, candidate_labels=labels)

    role = None
    location = None
    skills = []
    work_preference = None

    for entity in ner_results:
        word = entity.get("word", "").strip().lower()
        label = entity.get("entity_group", "")
        if label == "LOC" and not location:
            location = word.title()
        elif label in ["MISC", "ORG", "PER"]:
            if word in known_skills:
                skills.append(word.title())

    prompt_lower = prompt.lower()
    for loc in known_locations:
        if loc in prompt_lower:
            location = loc.title()
    for skill in known_skills:
        if skill in prompt_lower and skill.title() not in skills:
            skills.append(skill.title())
    for pref in ["remote", "hybrid", "on-site"]:
        if pref in prompt_lower:
            work_preference = pref.title()

    for label in classification["labels"]:
        if label in ["full stack", "developer", "data analyst"]:
            role = label
            break

    query_type = "job" if "job search" in classification["labels"][:2] else "candidate"

    return {
        "query_type": query_type,
        "role": role,
        "location": location,
        "skills": list(set(skills)),
        "work_preference": work_preference
    }

def score_job_relevance(job, role, skills):
    job_title = job.get("job_title", "").lower()
    job_desc = job.get("job_description", "").lower()

    score = 0
    if role and role.lower() in job_title:
        score += 3
    elif role and role.lower() in job_desc:
        score += 2

    for skill in skills:
        if skill.lower() in job_title:
            score += 2
        elif skill.lower() in job_desc:
            score += 1

    score += SequenceMatcher(None, job_title, role or "").ratio() * 2
    return score

def filter_top_jobs(jobs, role, skills, top_n=10):
    job_scores = [(score_job_relevance(job, role, skills), job) for job in jobs]
    sorted_jobs = sorted(job_scores, key=lambda x: x[0], reverse=True)
    return [job for score, job in sorted_jobs[:top_n]]
