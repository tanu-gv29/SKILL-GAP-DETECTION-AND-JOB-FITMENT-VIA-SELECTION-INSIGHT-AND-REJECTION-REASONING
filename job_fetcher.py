import requests
import spacy
import re
from docx import Document
from database import create_table, insert_or_update_job, role_exists

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DOC_PATH = os.path.join(BASE_DIR, "data", "Skills dataset.docx")



# Load NLP model
nlp = spacy.load("en_core_web_sm")

API_KEY = "932ac69105msh06dd5d23a658a8ep1edea1jsn38e990ffced2"


# -----------------------------
# LOAD SKILLS FROM DOCX
# -----------------------------
def load_skills_from_docx(file_path):
    doc = Document(file_path)

    text = ""
    for para in doc.paragraphs:
        text += para.text + " "

    raw_skills = text.replace(',', '\n').split('\n')

    skills = set()

    for skill in raw_skills:
        skill = skill.lower().strip()
        if len(skill) > 2:
            skills.add(skill)

    return skills

SKILLS_SET = load_skills_from_docx(SKILL_DOC_PATH)

# SKILLS_SET = load_skills_from_docx("data/Skill dataset.docx")


# -----------------------------
# MINIMAL FILTER
# -----------------------------
GENERIC_WORDS = {'code', 'design', 'process', 'team', 'work'}


# -----------------------------
# NORMALIZATION
# -----------------------------
def normalize_skill(skill):
    mapping = {
        "js": "javascript",
        "nodejs": "node.js",
        "reactjs": "react",
        "py": "python"
    }
    return mapping.get(skill, skill)

def is_valid_new_skill(skill, description):
    skill = skill.lower().strip()

    # Reject very short
    if len(skill) < 3:
        return False

    # Reject if contains stopwords inside phrase
    words = skill.split()
    for w in words:
        if w in nlp.Defaults.stop_words:
            return False

    # Must look technical
    if not (
        '.' in skill or
        '-' in skill or
        '+' in skill or
        '#' in skill or
        len(words) > 1
    ):
        return False

    return True

# def extract_final_skills(skills_list, description):
#     final_skills = set()

#     for skill in skills_list:
#         skill = normalize_skill(skill.lower().strip())

#         # 🔥 STEP 1: HARD FILTER
#         if not is_clean_skill(skill):
#             continue

#         # 🔥 STEP 2: SHAPE VALIDATION (NEW)
#         if not is_valid_skill_shape(skill):
#             continue

#         # 🔥 STEP 3: DATASET MATCH
#         if skill in SKILLS_SET:
#             final_skills.add(skill)
#             continue

#         # 🔥 STEP 4: SMART LOGIC
#         if is_valid_new_skill(skill, description):
#             final_skills.add(skill)

#     return list(final_skills)

def extract_final_skills_with_priority(skills_list, description):
    docx_skills = []
    smart_skills = []

    for skill in skills_list:
        skill = normalize_skill(skill.lower().strip())

        if not is_clean_skill(skill):
            continue

        if not is_valid_skill_shape(skill):
            continue

        if skill in SKILLS_SET:
            docx_skills.append(skill)
            continue

        if is_valid_new_skill(skill, description):
            smart_skills.append(skill)

    final_skills = list(dict.fromkeys(docx_skills + smart_skills))

    return final_skills[:20]

def clean_duplicates(skills):
    return list(set(skills))

def is_clean_skill(skill):
    skill = skill.lower().strip()

    # ❌ Remove URLs / emails
    if "http" in skill or "www" in skill or "@" in skill:
        return False

    # ❌ Remove numbers
    if any(char.isdigit() for char in skill):
        return False

    # ❌ Reject location-like words
    LOCATION_WORDS = {
        'bangalore', 'mumbai', 'delhi', 'india', 'new york',
        'london', 'dallas', 'pune'
    }

    if skill in LOCATION_WORDS:
        return False

    # ❌ Reject company/common words
    BAD_WORDS = {
        'team', 'work', 'role', 'company', 'environment',
        'members', 'job', 'ability', 'opportunity',
        'industry', 'business', 'organization',
        'people', 'things', 'way', 'process',
        'support', 'growth', 'project managers'
    }

    words = skill.split()

    for w in words:
        if w in BAD_WORDS or w in LOCATION_WORDS:
            return False

    # ❌ Reject phrases with verbs/adjectives (not pure skills)
    doc = nlp(skill)
    for token in doc:
        if token.pos_ in ["VERB", "ADV"]:
            return False

    return True

def is_valid_skill_shape(skill):
    words = skill.split()

    # ❌ Too long phrases (likely sentences)
    if len(words) > 3:
        return False

    # ❌ Contains bad connectors
    BAD_CONNECTORS = {"like", "and", "or", "with", "for", "the"}
    for w in words:
        if w in BAD_CONNECTORS:
            return False

    # ❌ Reject if all words are too generic
    GENERIC = {
        "system", "process", "management", "development",
        "environment", "understanding", "knowledge"
    }

    if all(w in GENERIC for w in words):
        return False

    return True


def extract_skills_with_context(text):
    doc = nlp(text)

    CONTEXT_WORDS = {"in", "with", "using"}
    skills = []

    for i, token in enumerate(doc):
        if token.text.lower() in CONTEXT_WORDS:

            phrase = []
            for j in range(1, 4):
                if i + j < len(doc):
                    next_token = doc[i + j]

                    if (
                        next_token.is_alpha and
                        not next_token.is_stop and
                        len(next_token.text) > 2
                    ):
                        phrase.append(next_token.text.lower())

            if len(phrase) >= 1:
                skills.append(" ".join(phrase))

    return list(set(skills))


# -----------------------------
# NLP EXTRACTION
# -----------------------------
def extract_skills(text):
    doc = nlp(text)
    skills = []

    for token in doc:
        word = token.text.lower()

        if (
            token.pos_ in ["NOUN", "PROPN"] and
            len(word) > 2 and
            not token.is_stop
        ):
            skills.append(word)

    return list(set(skills))

def extract_final_skills_balanced(skills_list):
    docx_skills = []
    smart_skills = []

    for skill in skills_list:
        skill = normalize_skill(skill.lower().strip())

        if not is_clean_skill(skill):
            continue

        if not is_valid_skill_shape(skill):
            continue

        # 🔥 PARTIAL MATCH (VERY IMPORTANT)
        if any(docx in skill or skill in docx for docx in SKILLS_SET):
            docx_skills.append(skill)
        else:
            if is_valid_new_skill(skill, ""):
                smart_skills.append(skill)

    # 🔥 REMOVE DUPLICATES
    docx_skills = list(dict.fromkeys(docx_skills))
    smart_skills = list(dict.fromkeys(smart_skills))

    # 🔥 TAKE 10 + 10
    final = docx_skills[:10] + smart_skills[:10]

    return final

# -----------------------------
# FETCH JOBS
# -----------------------------
def fetch_jobs(query="software developer"):
    url = "https://jsearch.p.rapidapi.com/search"

    querystring = {
        "query": query,
        "page": "1",
        "num_pages": "1",
        "country": "in"
    }

    headers = {
        "X-RapidAPI-Key": API_KEY,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    return response.json()

def fetch_and_store_jobs(query):

    query = query.lower().strip()

    # 🔥 STEP 1: CHECK IF ROLE ALREADY EXISTS
    if role_exists(query):
        print(f"⏩ Skipping API (already exists): {query}")
        return

    print(f"🚀 Fetching jobs for: {query}")

    data = fetch_jobs(query)

    if "data" not in data:
        print("❌ API failed or no data received")
        return
    
    create_table()

    all_skills = []

    # 🔹 Collect skills from ALL jobs
    for job in data["data"]:
        description = job.get("job_description", "")

        if not description:
           continue

        raw_skills = extract_skills(description)
        context_skills = extract_skills_with_context(description)

        combined = list(set(raw_skills + context_skills))
        all_skills.extend(combined)

    # 🔹 Remove duplicates globally
    all_skills = list(set(all_skills))

    # 🔹 Final prioritized split (10 DOCX + 10 SMART)
    final_skills = extract_final_skills_balanced(all_skills)

    # 🔹 Store under USER QUERY ROLE ONLY
    insert_or_update_job(query, final_skills)

    print(f"✅ Stored data for: {query}")
