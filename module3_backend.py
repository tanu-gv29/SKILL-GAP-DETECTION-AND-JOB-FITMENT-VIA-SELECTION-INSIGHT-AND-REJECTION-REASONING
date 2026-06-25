
# ============================================================
# 1) IMPORTS
# ============================================================
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pdfplumber
import docx2txt
import shap
import spacy

from typing import List, Dict, Tuple


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 2) FILE TEXT EXTRACTION
# ============================================================
def extract_text_from_path(filepath: str) -> str:
    filepath_l = filepath.lower()

    if filepath_l.endswith(".pdf"):
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)

    elif filepath_l.endswith((".docx", ".doc")):
        return docx2txt.process(filepath)

    elif filepath_l.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {filepath}")


# ============================================================
# 3) DYNAMIC KEYWORDS FROM JD (NO STATIC EXCEL)
# ============================================================
def extract_dynamic_keywords(jd_text: str, max_keywords: int = 70) -> List[str]:
    jd_text = jd_text.lower().strip()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_keywords
    )
    vectorizer.fit([jd_text])

    keywords = vectorizer.get_feature_names_out().tolist()

    keywords = [
        k for k in keywords
        if len(k) > 2
        and not k.isdigit()
        and "year" not in k
        and "experience" not in k
    ]
    return sorted(list(set(keywords)))

nlp = spacy.load("en_core_web_sm")

def intelligent_keyword_filter(keywords):
    clean_keywords = []

    for phrase in keywords:
        phrase = phrase.strip().lower()
        doc = nlp(phrase)

        # 1️⃣ Must contain at least one NOUN or PROPN
        if not any(token.pos_ in ["NOUN", "PROPN"] for token in doc):
            continue

        # 2️⃣ Must NOT start with a verb
        if doc[0].pos_ == "VERB":
            continue

        # 3️⃣ Remove phrases that are all stopwords
        if all(token.is_stop for token in doc):
            continue

        # 4️⃣ Remove very short phrases
        if len(phrase) < 3:
            continue

        clean_keywords.append(phrase)

    return list(set(clean_keywords))
# ============================================================
# 4) FEATURES
# ============================================================
SECTION_PATTERNS = {
    "has_summary":        r"(summary|objective|profile|about me)",
    "has_experience":     r"(experience|work history|employment|career)",
    "has_education":      r"(education|academic|qualification|degree)",
    "has_skills":         r"(skill|competenc|technolog|expertise|proficien)",
    "has_projects":       r"(project|portfolio|work sample)",
    "has_certifications": r"(certif|award|honour|honor|publication|patent)",
    "has_contact":        r"(email|phone|linkedin|github|contact|@)",
    "has_quantified":     r"(\d+\s*%|\$\s*\d+|\d+x\s|\d+\+?\s*(user|client|team|million|k\b))",
}

ACTION_VERBS_PATTERN = (
    r"\b(led|built|designed|developed|managed|created|improved|"
    r"optimized|launched|delivered|achieved|reduced|increased|"
    r"automated|implemented|deployed|integrated|architected|mentored|"
    r"streamlined|spearheaded|orchestrated)\b"
)

def compute_tfidf_similarity(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )
    try:
        matrix = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except Exception:
        return 0.0

def compute_keyword_overlap(resume_text: str, jd_keywords: List[str]) -> Tuple[List[str], List[str], float]:
    resume_lower = resume_text.lower()
    matched = [kw for kw in jd_keywords if re.search(r"\b" + re.escape(kw) + r"\b", resume_lower)]
    missing = [kw for kw in jd_keywords if kw not in matched]
    ratio = len(matched) / max(len(jd_keywords), 1)
    return matched, missing, ratio

def extract_features(text: str, jd_text: str, jd_keywords: List[str]) -> Dict[str, float]:
    text_lower = text.lower()

    words = re.findall(r"\b\w+\b", text_lower)
    word_count = len(words)

    sentences = re.split(r"[.!?]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    features = {}

    # Length + structure
    features["word_count"] = word_count
    features["char_count"] = len(text)
    features["sentence_count"] = len(sentences)
    features["avg_sentence_length"] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    features["unique_word_ratio"] = len(set(words)) / max(word_count, 1)

    # Section presence
    for feat, pattern in SECTION_PATTERNS.items():
        features[feat] = float(bool(re.search(pattern, text_lower, re.IGNORECASE)))

    # Contact signals
    features["has_email"]    = float(bool(re.search(r"[\w.+-]+@[\w-]+\.\w+", text)))
    features["has_url"]      = float(bool(re.search(r"https?://|www\.", text_lower)))
    features["has_linkedin"] = float(bool(re.search(r"linkedin", text_lower)))
    features["has_github"]   = float(bool(re.search(r"github", text_lower)))

    # Quantified achievements
    features["has_numbers"] = float(bool(re.search(r"\b\d+\.?\d*\b", text)))
    numbers = re.findall(r"\b\d+\.?\d*\b", text)
    features["number_count"] = len(numbers)
    features["number_density"] = len(numbers) / max(word_count, 1)

    # Bullets
    bullets = re.findall(r"^\s*[•\-\*►▸▶◆]\s", text, re.MULTILINE)
    features["bullet_count"] = len(bullets)
    features["bullet_density"] = len(bullets) / max(len(text.split("\n")), 1)

    # Action verbs
    action_verbs = re.findall(ACTION_VERBS_PATTERN, text_lower)
    features["action_verb_count"] = len(action_verbs)
    features["action_verb_variety"] = len(set(action_verbs))

    # Estimated years of experience
    year_exp = re.findall(r"(\d+)\+?\s*(?:year|yr)s?\s*(?:of)?\s*(?:experience|exp)", text_lower)
    features["years_experience"] = max([int(y) for y in year_exp], default=0)

    # Education score
    edu_score = 0
    if re.search(r"\bphd\b|\bdoctoral\b|\bd\.phil\b", text_lower): edu_score = 4
    elif re.search(r"\bm\.tech\b|\bm\.e\b|\bmasters?\b|\bmba\b|\bm\.s\b", text_lower): edu_score = 3
    elif re.search(r"\bb\.tech\b|\bb\.e\b|\bbachelor\b|\bb\.s\b|\bb\.sc\b", text_lower): edu_score = 2
    elif re.search(r"\bdiploma\b|\bassociate\b", text_lower): edu_score = 1
    features["education_level"] = edu_score

    # JD similarity
    features["jd_similarity"] = compute_tfidf_similarity(text, jd_text)

    # JD keyword overlap
    matched_kw, _, overlap = compute_keyword_overlap(text, jd_keywords)
    features["kw_overlap"] = overlap
    features["jd_kw_matched"] = len(matched_kw)

    return features


# ============================================================
# 5) MODEL
# ============================================================
def build_feature_matrix(rejected_text: str, selected_texts: List[str], jd_text: str):
    jd_keywords = extract_dynamic_keywords(jd_text)

    all_texts = [rejected_text] + selected_texts
    labels = np.array([0] + [1] * len(selected_texts))

    rows = []
    for t in all_texts:
        rows.append(extract_features(t, jd_text, jd_keywords))

    X_df = pd.DataFrame(rows).fillna(0)
    return X_df, labels, jd_keywords

def train_model(X_df: pd.DataFrame, y: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_scaled, y)
    return model, scaler


# ============================================================
# 6) SHAP FIXED
# ============================================================
def compute_shap(model, scaler, X_df, sample_idx=0):
    X_scaled = scaler.transform(X_df)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_scaled)

    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals

    if len(sv.shape) == 3:
        sv = sv[:, :, 1]

    rejected_shap = sv[sample_idx]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    return rejected_shap, float(base_value)


# ============================================================
# 7) PLOTS
# ============================================================
def label_feat(f: str) -> str:
    return f.replace("_", " ").title()

def plot_shap_bar(shap_vals: np.ndarray, feature_names: List[str], top_n: int = 8) -> go.Figure:
    pairs = list(zip(feature_names, shap_vals))

    positive_pairs = [(f, float(v)) for f, v in pairs if float(v) > 0]
    negative_pairs = [(f, float(v)) for f, v in pairs if float(v) < 0]

    # strongest contributors
    positive_pairs = sorted(positive_pairs, key=lambda x: x[1], reverse=True)[:top_n]
    negative_pairs = sorted(negative_pairs, key=lambda x: x[1])[:top_n]

    pos_feats = [label_feat(f) for f, _ in positive_pairs]
    pos_vals = [v for _, v in positive_pairs]

    neg_feats = [label_feat(f) for f, _ in negative_pairs]
    neg_vals = [v for _, v in negative_pairs]

    fig = go.Figure()

    if negative_pairs:
        fig.add_trace(go.Bar(
            x=neg_vals,
            y=neg_feats,
            orientation="h",
            name="Hurts Selection",
            marker_color="#fc8181",
            text=[round(v, 3) for v in neg_vals],
            textposition="outside"
        ))

    if positive_pairs:
        fig.add_trace(go.Bar(
            x=pos_vals,
            y=pos_feats,
            orientation="h",
            name="Helps Selection",
            marker_color="#68d391",
            text=[round(v, 3) for v in pos_vals],
            textposition="outside"
        ))

    fig.update_layout(
        title="SHAP Contribution Analysis",
        template="plotly_white",
        height=560,
        margin=dict(l=220, r=40, t=70, b=40),
        xaxis_title="SHAP value",
        barmode="relative",
        legend=dict(orientation="h", y=1.08, x=0.02)
    )

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    # if not positive_pairs:
    #     fig.add_annotation(
    #     text="No strong positive feature contributions found for this resume.",
    #     xref="x domain",
    #     yref="paper",
    #     x=0.5,
    #     y=1.03,
    #     xanchor="center",
    #     yanchor="bottom",
    #     align="center",
    #     showarrow=False,
    #     font=dict(size=12, color="#4a5568")
    # )
    return fig


def plot_feature_importance(model, feature_names: List[str], top_n: int = 18) -> go.Figure:
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    pairs = pairs[::-1]

    feats = [label_feat(f) for f, _ in pairs]
    vals = [float(v) for _, v in pairs]

    fig = go.Figure(go.Bar(
        x=vals, y=feats,
        orientation="h"
    ))
    fig.update_layout(
        title="Random Forest Feature Importance",
        template="plotly_white",
        height=520,
        margin=dict(l=220, r=30, t=60, b=30),
        xaxis_title="importance"
    )
    return fig

def plot_radar(rejected_features: Dict[str, float], selected_avg: Dict[str, float]) -> go.Figure:
    cats = [
        "jd_similarity",
        "kw_overlap",
        "years_experience",
        "education_level",
        "action_verb_variety",
        "bullet_count",
        "number_density",
        "word_count"
    ]

    labels = [label_feat(c) for c in cats]

    rej = []
    sel = []
    for c in cats:
        r = rejected_features.get(c, 0)
        s = selected_avg.get(c, 1)
        if s == 0:
            s = 1
        rej.append(min((r / s) * 100, 100))
        sel.append(100)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sel + [sel[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="Selected Avg"
    ))
    fig.add_trace(go.Scatterpolar(
        r=rej + [rej[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name="Rejected Resume"
    ))
    fig.update_layout(
        title="Resume Profile (100 = matches selected avg)",
        template="plotly_white",
        height=520,
        polar=dict(radialaxis=dict(visible=True, range=[0, 110])),
    )
    return fig


# ============================================================
# 8) REASONS + ROADMAP
# ============================================================
def generate_reasons(
    rejected_shap: np.ndarray,
    feature_names: List[str],
    rejected_features: Dict[str, float],
    selected_avg_features: Dict[str, float],
    matched_kw: List[str],
    missing_kw: List[str],
    jd_sim: float,
    kw_overlap: float,
) -> List[Dict]:

    shap_pairs = sorted(zip(feature_names, rejected_shap), key=lambda x: float(x[1]))
    top_negative = [(f, v) for f, v in shap_pairs if float(v) < 0][:10]

    reasons = []

    if jd_sim < 0.25:
        reasons.append({
            "severity": "HIGH",
            "category": "JD Alignment",
            "reason": f"Very low alignment with JD (similarity {jd_sim:.0%}).",
            "fix": "Rewrite summary + experience bullets using JD wording."
        })
    elif jd_sim < 0.45:
        reasons.append({
            "severity": "MEDIUM",
            "category": "JD Alignment",
            "reason": f"Moderate alignment with JD (similarity {jd_sim:.0%}).",
            "fix": "Add a tailored summary and highlight the most relevant projects."
        })

    if kw_overlap < 0.35:
        reasons.append({
            "severity": "HIGH",
            "category": "Keyword Coverage",
            "reason": f"Low keyword coverage ({kw_overlap:.0%}). Missing important JD terms.",
            "fix": f"Add these keywords naturally: {', '.join(missing_kw[:12])}"
        })
    elif kw_overlap < 0.60:
        reasons.append({
            "severity": "MEDIUM",
            "category": "Keyword Coverage",
            "reason": f"Medium keyword coverage ({kw_overlap:.0%}).",
            "fix": f"Add these keywords naturally: {', '.join(missing_kw[:8])}"
        })

    for feat, shap_val in top_negative:
        rej_val = rejected_features.get(feat, 0)
        sel_val = selected_avg_features.get(feat, 0)

        if feat == "has_projects" and rej_val == 0 and sel_val > 0.5:
            reasons.append({
                "severity": "MEDIUM",
                "category": "Projects Missing",
                "reason": "Selected resumes include projects but yours does not.",
                "fix": "Add 2–3 projects with tech stack + outcome + metrics."
            })

        if feat == "has_quantified" and rej_val == 0:
            reasons.append({
                "severity": "HIGH",
                "category": "No Metrics",
                "reason": "Your resume lacks measurable impact (%, users, time saved).",
                "fix": "Add numbers to bullets: improved by X%, reduced time by Y%."
            })

        if feat == "word_count" and rej_val < 200:
            reasons.append({
                "severity": "HIGH",
                "category": "Resume Too Short",
                "reason": f"Your resume is very short ({rej_val:.0f} words).",
                "fix": "Expand experience/project bullets to show depth."
            })

        if feat == "has_linkedin" and rej_val == 0:
            reasons.append({
                "severity": "LOW",
                "category": "Missing LinkedIn",
                "reason": "Many selected resumes include LinkedIn.",
                "fix": "Add LinkedIn URL in header."
            })

        if feat == "has_github" and rej_val == 0:
            reasons.append({
                "severity": "LOW",
                "category": "Missing GitHub",
                "reason": "Many selected resumes include GitHub.",
                "fix": "Add GitHub link for technical roles."
            })

    # Deduplicate by category
    seen = set()
    uniq = []
    for r in reasons:
        if r["category"] not in seen:
            seen.add(r["category"])
            uniq.append(r)

    return uniq

def generate_roadmap(reasons: List[Dict], missing_kw: List[str]) -> pd.DataFrame:
    priority_map = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
    reasons_sorted = sorted(reasons, key=lambda r: priority_map.get(r["severity"], 3))

    rows = []
    for i, r in enumerate(reasons_sorted[:8], 1):
        rows.append({
            "Step": i,
            "Priority": r["severity"],
            "Area": r["category"],
            "Action": r["fix"]
        })

    if missing_kw:
        rows.append({
            "Step": len(rows) + 1,
            "Priority": "MEDIUM",
            "Area": "Keyword Integration",
            "Action": "Integrate these JD keywords naturally: " + ", ".join(missing_kw[:12])
        })

    return pd.DataFrame(rows)


# ============================================================
# 9) MAIN RUNNER
# ============================================================
# 1️⃣ Utility functions first

def extract_feature_impact(shap_values, feature_names):
    impacts = list(zip(feature_names, shap_values))
    impacts_sorted = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)
    return impacts_sorted[:5]


def generate_intelligent_insight(feature, value, shap_value):
    feature_name = feature.replace("_", " ").title()

    if shap_value > 0:
        return f"{feature_name} helped improve the selection probability (impact score: {round(shap_value, 3)})."
    else:
        return f"{feature_name} reduced the selection probability (impact score: {round(shap_value, 3)})."

def run_analysis(rejected_path: str, selected_paths: List[str], jd_text: str) -> Dict:
    rejected_text = extract_text_from_path(rejected_path)
    selected_texts = [extract_text_from_path(p) for p in selected_paths]

    X_df, y, jd_keywords = build_feature_matrix(rejected_text, selected_texts, jd_text)
    model, scaler = train_model(X_df, y)

    rejected_shap, base_value = compute_shap(model, scaler, X_df, sample_idx=0)

    # 🔥 Intelligent SHAP Insight Generation
    top_impacts = extract_feature_impact(
    rejected_shap if len(rejected_shap.shape) == 1 else rejected_shap[0],
    X_df.columns.tolist()
    )


    intelligent_insights = []
    for feature, shap_val in top_impacts:
        feature_value = X_df.iloc[0][feature]
        insight = generate_intelligent_insight(feature, feature_value, shap_val)
        intelligent_insights.append(insight)

    prob_selected = model.predict_proba(scaler.transform(X_df))[0][1]

    matched_kw, missing_kw, kw_overlap = compute_keyword_overlap(rejected_text, jd_keywords)
    jd_sim = compute_tfidf_similarity(rejected_text, jd_text)

    # Intelligent filtering (dynamic, no hardcoding)
    matched_kw = intelligent_keyword_filter(matched_kw)
    missing_kw = intelligent_keyword_filter(missing_kw)

# Recalculate coverage after cleaning
    kw_overlap = len(matched_kw) / max(len(jd_keywords), 1)

    rejected_features = X_df.iloc[0].to_dict()
    selected_avg = X_df.iloc[1:].mean().to_dict()

    reasons = generate_reasons(
        rejected_shap,
        X_df.columns.tolist(),
        rejected_features,
        selected_avg,
        matched_kw,
        missing_kw,
        jd_sim,
        kw_overlap
    )

    roadmap_df = generate_roadmap(reasons, missing_kw)

    return {
        "X_df": X_df,
        "model": model,
        "scaler": scaler,
        "prob_selected": prob_selected,
        "matched_kw": matched_kw,
        "missing_kw": missing_kw,
        "kw_overlap": kw_overlap,
        "jd_similarity": jd_sim,
        "rejected_shap": rejected_shap,
        "base_value": base_value,
        "feature_names": X_df.columns.tolist(),
        "jd_keywords": jd_keywords,
        "rejected_features": rejected_features,
        "selected_avg": selected_avg,
        "reasons": reasons,
        "roadmap_df": roadmap_df,
        "rejected_text": rejected_text,
        "jd_text": jd_text,
        "intelligent_insights": intelligent_insights,
        "has_positive_shap": bool(np.any(rejected_shap > 0)),
    }



