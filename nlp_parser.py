from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load pretrained model and tokenizer locally for NER
MODEL_NER = "dslim/bert-base-NER"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NER, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NER, local_files_only=True)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Zero-shot classification (will use remote if not cached)
classifier_pipeline = pipeline("zero-shot-classification")

labels = ["job search", "hiring", "developer", "data analyst", "machine learning", "React", "Python", "Bangalore",
          "remote", "senior", "junior"]


def parse_job_prompt(prompt):
    ner_results = ner_pipeline(prompt)
    classification = classifier_pipeline(prompt, candidate_labels=labels)

    role = None
    location = None
    skills = []

    # Custom lists to help extract more accurate info
    known_skills = ["python", "react", "machine learning", "data analysis", "sql", "javascript", "java"]
    known_locations = ["surat", "mumbai", "bangalore", "delhi", "remote", "pune", "ahmedabad", "hyderabad"]

    # NER-based detection
    for entity in ner_results:
        word = entity.get("word", "").strip().lower()
        label = entity.get("entity_group", "")
        if label == "LOC" and not location:
            location = word.title()
        elif label in ["MISC", "ORG", "PER"]:
            if word.lower() in known_skills:
                skills.append(word.title())

    # Fallback: Keyword matching if NER fails
    prompt_lower = prompt.lower()
    for loc in known_locations:
        if loc in prompt_lower:
            location = loc.title()
    for skill in known_skills:
        if skill in prompt_lower and skill.title() not in skills:
            skills.append(skill.title())

    # Classification-based role detection
    for label in classification["labels"]:
        if label in ["developer", "data analyst", "machine learning"]:
            role = label
            break

    query_type = "job" if "job search" in classification["labels"][:2] else "candidate"

    return {
        "query_type": query_type,
        "role": role,
        "location": location,
        "skills": list(set(skills))
    }
