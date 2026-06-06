# %% [markdown]
# # ⚙️ Notebook 02 — Feature Engineering Pipeline
# ### REDRO AI Hackathon | 100,000 Candidates → Feature Table
# 
# **Goal:** Convert raw candidate JSON into a clean, normalized feature table.
# One row per candidate. Ready for ranking in Notebook 03.
# 
# | Input | `candidates.jsonl` (100k) |
# |---|---|
# | Output | `features_df` (100k × 53 features + candidate_id) |
# | Saved | `outputs/features_df.pkl` + `outputs/features_df.csv` |
# 
# Every feature in this notebook is traceable to a specific EDA-01 observation.
# No charts. No conclusions. Just features.
# 
# ---

# %% [markdown]
# ## ⚙️ Phase 0 — Setup & Dataset Load

# %%
import json, warnings, os, math
from datetime import date
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.4f}".format)

os.makedirs("outputs", exist_ok=True)
REFERENCE_DATE = date(2026, 6, 5)
print("Imports OK")

# %%
DATASET_PATH = "raw_dataset/candidates.jsonl"

candidates = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            candidates.append(json.loads(line))

N = len(candidates)
print(f"Loaded: {N:,} candidates")

# %%
# ── JD text (technical core only — tuned for semantic matching in Phase 10) ──
JD_TEXT = """
Senior AI Engineer — Retrieval, Search, Ranking and Recommendation Systems

Production experience with embeddings-based retrieval systems.
Vector databases: FAISS, Pinecone, Milvus, Weaviate, Qdrant, Elasticsearch, OpenSearch.
Hybrid search: dense retrieval, BM25, sparse-dense fusion, semantic search.
Information retrieval, search relevance, retrieval quality, ranking systems.
Evaluation: NDCG, MRR, MAP, offline evaluation, A/B testing, online evaluation.
Learning to rank: XGBoost-based ranking, neural ranking, feature engineering.
Recommendation systems, candidate matching, similarity search.
LLM: RAG, retrieval augmented generation, LangChain, Prompt Engineering.
Fine-tuning: LoRA, QLoRA, PEFT, model adaptation.
Sentence transformers, embedding models, embedding drift, index refresh.
NLP, natural language processing, text ranking, search quality.
Python, production ML, MLOps, MLflow, feature engineering.
Machine learning, deep learning, PyTorch, TensorFlow, scikit-learn.
5-9 years at product companies. Pune, Noida, Hyderabad, Mumbai, Delhi NCR.
"""

print(f"JD text: {len(JD_TEXT.split())} words")

# %%
# ── Constants derived from EDA-01 ────────────────────────────────────────────

# Improvement 2: Recommendation Systems REMOVED from retrieval — separate bucket
# JD focuses on retrieval/search/ranking, not generic recommendations
RETRIEVAL_SKILLS = {
    "Embeddings","FAISS","Milvus","Elasticsearch","BM25",
    "Information Retrieval","Vector Search","Pinecone",
    "Weaviate","Qdrant","OpenSearch","Dense Retrieval",
    "Hybrid Search","Semantic Search",
}

# Improvement 2: Recommendation as separate, lower-weight bucket
RECOMMENDATION_SKILLS = {
    "Recommendation Systems",
    "Collaborative Filtering",
    "Matrix Factorization",
    "Content-Based Filtering",
    "Personalization",
}
LLM_SKILLS = {
    "LangChain","Prompt Engineering","Fine-tuning LLMs","RAG",
    "LoRA","QLoRA","PEFT","Retrieval Augmented Generation",
}
ML_SKILLS = {
    "Machine Learning","Deep Learning","PyTorch","TensorFlow",
    "MLflow","Hugging Face Transformers","Sentence Transformers",
    "scikit-learn","MLOps","Feature Engineering",
    "Learning to Rank","XGBoost","LightGBM",
}
HIDDEN_SIGNALS = {
    "NDCG","MRR","MAP","A/B Testing","Offline Evaluation",
    "Online Evaluation","Search Relevance","Retrieval Quality",
    "Ranking Systems","Search Quality",
}
# AI skill categories only — HIDDEN_SIGNALS are text evidence, not skills
AI_ALL_SKILLS = RETRIEVAL_SKILLS | LLM_SKILLS | ML_SKILLS

PROF_WEIGHT = {"beginner":0.5,"intermediate":1.0,"advanced":2.0,"expert":3.0}

AI_CERTS = {
    "Deep Learning Specialization","NLP Specialization",
    "AWS Certified Machine Learning Specialty",
    "Google Cloud Professional ML Engineer",
    "LangChain for LLM Application Development",
    "TensorFlow Developer Certificate",
    "Hugging Face NLP Certificate",
    "MLOps Professional Certificate",
    "Machine Learning Engineering for Production",
    "Natural Language Processing Specialization",
    "Recommendation Systems Specialization",
}

CONSULTING_FIRMS = {
    "tcs","infosys","wipro","accenture","cognizant",
    "capgemini","hcl","tech mahindra","mphasis","ltimindtree",
}

# Improvement 1: Evaluation-signal terms (highest-value missing feature)
# These appear ONLY in career descriptions of people who built+measured ranking systems
# JD emphasizes these MORE than LangChain — they separate real IR engineers from practitioners
EVALUATION_TERMS = [
    "ndcg", "mrr", "map", "a/b test", "a/b testing", "ab test",
    "offline evaluation", "online evaluation", "search relevance",
    "ranking quality", "mean average precision", "mean reciprocal rank",
    "precision@", "recall@", "hit rate", "evaluation framework",
    "relevance judgment", "query evaluation", "ranking evaluation",
]

# Improvement 3: Production-signal terms — evidence of real deployment
# JD explicitly says "production experience" is the #1 requirement
# Pure researchers and LangChain practitioners will NOT have these in descriptions
PRODUCTION_TERMS = [
    "production", "deployed", "real users", "live traffic",
    "shipped", "launched", "serving", "at scale",
    "production system", "online serving", "production environment",
    "millions of", "billions of", "production traffic",
    "latency", "throughput", "production pipeline",
]

# Improvement 4: Career keyword density — direct domain vocabulary evidence
# These are the words that describe the work this role actually does
CAREER_KEYWORDS = [
    "ranking", "retrieval", "search", "embedding", "relevance",
    "evaluation", "recommendation", "matching", "indexing",
    "reranking", "re-ranking", "semantic", "vector", "dense retrieval",
    "hybrid", "query", "recall", "precision", "candidate retrieval",
    "feature engineering", "ranking model", "similarity",
]

# Location scoring (EDA Obs 15.3 + JD)
PREFERRED_CITIES  = {"pune","noida"}
ACCEPTABLE_CITIES = {
    "hyderabad","mumbai","bangalore","bengaluru",
    "delhi","new delhi","gurgaon","gurugram","navi mumbai",
}

print("Constants defined:")
print(f"  Retrieval={len(RETRIEVAL_SKILLS)} | LLM={len(LLM_SKILLS)} | ML={len(ML_SKILLS)} | Recommendation={len(RECOMMENDATION_SKILLS)}")
print(f"  Hidden signals : {len(HIDDEN_SIGNALS)}")
print(f"  Evaluation terms : {len(EVALUATION_TERMS)}")
print(f"  Production terms : {len(PRODUCTION_TERMS)}")
print(f"  Career keywords  : {len(CAREER_KEYWORDS)}")

# %% [markdown]
# **Design Note 0.1 — Two Critical Inputs**
# This notebook requires exactly two things: `candidates.jsonl` (raw data) and `JD_TEXT`
# (the role specification). Every feature computed below is relative to one or both of these inputs.
# 
# The JD text is stripped to its technical core — company culture and logistics text
# removed — to give the semantic model in Phase 10 a clean signal to match against.
# 
# **Design Note 0.2 — Constant Definitions Are EDA Artefacts**
# Every constant above (`RETRIEVAL_SKILLS`, `PROF_WEIGHT`, `CONSULTING_FIRMS`, etc.) was
# derived from specific EDA-01 observations. They are hypotheses, not ground truth.
# Notebook 03 will tune the weights; this notebook only produces the raw feature signals.

# %% [markdown]
# ## 📝 Phase 1 — Candidate Text Construction

# %%
def build_candidate_text(c):
    """
    Combine all textual fields into one searchable string per candidate.
    Order matters: role-defining fields first, supporting evidence after.
    """
    parts = []

    # 1. Current title + headline (role context)
    parts.append(c["profile"].get("current_title", ""))
    parts.append(c["profile"].get("headline", ""))

    # 2. Professional summary
    parts.append(c["profile"].get("summary", ""))

    # 3. Skill names (explicit declarations)
    skill_names = [s["name"] for s in c.get("skills", [])]
    parts.append(" ".join(skill_names))

    # 4. Career titles + descriptions (richest source — EDA Obs 13.11)
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))

    return " ".join(p for p in parts if p).strip()

candidate_texts = [build_candidate_text(c) for c in candidates]

print(f"Built {len(candidate_texts):,} candidate texts")
print(f"\nSample text (first 300 chars):")
print(candidate_texts[0][:300])

# %%
# Basic text stats
lengths = [len(t.split()) for t in candidate_texts]
print("Candidate Text Length (words)")
print(f"  Mean   : {np.mean(lengths):.0f}")
print(f"  Median : {np.median(lengths):.0f}")
print(f"  Min    : {min(lengths)}")
print(f"  Max    : {max(lengths)}")
print(f"  Empty  : {sum(1 for l in lengths if l == 0)}")

# Lookup dict — avoids rebuilding candidate texts inside Phase 2 loop (Fix 2)
candidate_text_lookup = {
    c["candidate_id"]: t
    for c, t in zip(candidates, candidate_texts)
}
print(f"Lookup dict built: {len(candidate_text_lookup):,} entries")

# %% [markdown]
# **Design Note 1.1 — Why This Text Order**
# Order reflects signal strength from EDA:
# 1. `current_title` + `headline` → role context for the model
# 2. `summary` → self-description, often contains seniority and domain signals
# 3. `skills` → explicit keyword declarations
# 4. `career_history.description` → **the richest source** (EDA Obs 13.11)
# 
# Career descriptions are last but most important for semantic matching. They contain
# keywords that never appear in the structured skills field:
# `NDCG`, `MRR`, `A/B testing`, `offline evaluation`, `search relevance`, `ranking quality`.
# 
# A candidate with `Recommendation Systems Engineer` title who built an NDCG-evaluated
# ranking system will score high on semantic similarity even if their skills section
# lists nothing AI-related. That is the correct behaviour.
# 
# **Design Note 1.2 — No Truncation Applied**
# Text is left at full length. The sentence-transformer model in Phase 10 will handle
# tokenization and truncation internally. Max token limits are handled by the model.

# %% [markdown]
# ## 🎯 Phase 2 — Skill Category Features

# %%
skill_rows = []

for c in candidates:
    cid    = c["candidate_id"]
    skills = {s["name"] for s in c.get("skills", [])}

    ret = len(skills & RETRIEVAL_SKILLS)
    llm = len(skills & LLM_SKILLS)
    ml  = len(skills & ML_SKILLS)
    # Improvement 2: recommendation now separate, NOT folded into retrieval
    rec = len(skills & RECOMMENDATION_SKILLS)

    # Weighted scores: retrieval > llm > recommendation > ml
    # (JD hierarchy: retrieval/search/ranking first, reco secondary, generic ML last)
    ret_score = ret * 3
    llm_score = llm * 2
    rec_score = rec * 2   # Improvement 2: separate weight — it's related but not primary
    ml_score  = ml  * 1
    ai_total  = ret_score + llm_score + ml_score   # keep existing total without reco
    ai_total_with_rec = ret_score + llm_score + rec_score + ml_score

    # Full candidate text for all text-based signals (pre-built in Phase 1)
    full_text = candidate_text_lookup[cid].lower()

    # Existing: hidden signal count
    hidden_count = sum(1 for sig in HIDDEN_SIGNALS if sig.lower() in full_text)
    hidden_signal_bonus = min(hidden_count * 0.03, 0.15)  # Fix #3

    # Improvement 1: Evaluation-signal score — HIGHEST VALUE MISSING FEATURE
    # NDCG/MRR/MAP in career desc = candidate has measured ranking systems in production
    # This separates real IR engineers from LangChain practitioners completely
    eval_hits  = sum(1 for term in EVALUATION_TERMS if term in full_text)
    evaluation_signal_score = round(min(eval_hits * 0.20, 1.0), 4)

    # Improvement 3: Production-signal score — evidence of real deployment
    # JD says "production experience" is the #1 absolute requirement
    # Researchers and practitioners will score near-zero here
    prod_hits  = sum(1 for term in PRODUCTION_TERMS if term in full_text)
    production_signal_score = round(min(prod_hits * 0.12, 1.0), 4)

    # Improvement 4: Career keyword density — direct domain vocabulary presence
    # Fraction of core retrieval/ranking/search keywords found in full text
    kw_hits  = sum(1 for kw in CAREER_KEYWORDS if kw in full_text)
    career_keyword_score = round(kw_hits / len(CAREER_KEYWORDS), 4)

    skill_rows.append({
        "candidate_id"           : cid,
        "retrieval_count"        : ret,
        "llm_count"              : llm,
        "ml_count"               : ml,
        "recommendation_count"   : rec,              # Improvement 2
        "retrieval_score"        : ret_score,
        "llm_score"              : llm_score,
        "ml_score"               : ml_score,
        "recommendation_score"   : rec_score,        # Improvement 2
        "ai_skill_total"         : ai_total,
        "ai_total_with_rec"      : ai_total_with_rec, # Improvement 2
        "hidden_signal_count"    : hidden_count,
        "hidden_signal_bonus"    : hidden_signal_bonus,
        "evaluation_signal_score": evaluation_signal_score,  # Improvement 1 ★
        "production_signal_score": production_signal_score,  # Improvement 3 ★
        "career_keyword_score"   : career_keyword_score,     # Improvement 4 ★
    })

skill_df = pd.DataFrame(skill_rows)
print("Skill features shape:", skill_df.shape)
print("\n--- Skill category scores ---")
print(skill_df[["retrieval_score","recommendation_score","llm_score","ml_score","ai_skill_total"]].describe().round(2))
print("\n--- New text-based signals ---")
print(skill_df[["evaluation_signal_score","production_signal_score","career_keyword_score","hidden_signal_bonus"]].describe().round(4))

# %% [markdown]
# **Design Note 2.1 — Category Weights Are JD-Derived Hypotheses**
# Weights: `retrieval=×3` > `llm=×2` = `recommendation=×2` > `ml=×1`.
# Retrieval/search is the JD's core mandate. Recommendation is related but secondary — kept at ×2 but as a SEPARATE bucket so Notebook 03 can weight them independently.
# 
# **Design Note 2.2 (Improvement 1) — evaluation_signal_score Is the Highest-Value New Feature**
# The JD emphasizes NDCG, MRR, MAP, A/B testing, and offline/online evaluation *more than LangChain*.
# A candidate who writes "improved NDCG@10 by 12%" has definitively built and measured a ranking system in production.
# This single feature cleanly separates real IR engineers from LangChain practitioners.
# 
# **Design Note 2.3 (Improvement 3) — production_signal_score Validates Real Deployment**
# The JD's #1 absolute requirement is production experience — not research, not demos, not tutorials.
# Terms like "deployed", "live traffic", "at scale", "production system" in career descriptions
# are the strongest possible evidence. Researchers and practitioners score near-zero here.
# 
# **Design Note 2.4 (Improvement 4) — career_keyword_score Measures Domain Vocabulary Density**
# Fraction of core retrieval/ranking/search keywords found in the full candidate text.
# This is a direct, interpretable feature for Notebook 03 — no normalization needed (already 0–1).
# 
# **Design Note 2.2 — Hidden Signal Count Is a Separate Feature**
# `NDCG`, `MRR`, `MAP`, `A/B Testing`, `search relevance` appear in
# `career_history.description`, not in the skills field.
# A candidate who writes "improved NDCG@10 by 12%" in their job description has
# production evaluation experience. This is a strong signal that can't be captured by
# skill-category scoring alone. `hidden_signal_count` explicitly surfaces it.
# 
# **Design Note 2.3 — Skill Count ≠ Skill Quality**
# This phase captures breadth (how many AI-category skills a candidate claims).
# Phase 3 captures depth (how credible each skill is). Both are needed.

# %% [markdown]
# ## 🔬 Phase 3 — Skill Quality Features

# %%
quality_rows = []

for c in candidates:
    cid = c["candidate_id"]
    ai_skills = [
        s for s in c.get("skills", [])
        if s["name"] in AI_ALL_SKILLS
    ]

    if not ai_skills:
        quality_rows.append({
            "candidate_id"       : cid,
            "avg_ai_duration"    : 0,
            "advanced_ai_skills" : 0,
            "expert_ai_skills"   : 0,
            "quality_score"      : 0.0,
            "quality_score_log"  : 0.0,   # Fix #4: log1p normalized — use this in Notebook 03
            "max_endorsements_ai": 0,
        })
        continue

    # Per-skill quality score = proficiency_weight × log1p(duration) / log1p(12)
    # Normalised so that intermediate skill at 12 months = 1.0
    # (EDA Obs 14.2: duration tracks proficiency correctly)
    scores = []
    for s in ai_skills:
        pw  = PROF_WEIGHT.get(s.get("proficiency","beginner"), 0.5)
        dur = s.get("duration_months", 0)
        score = pw * math.log1p(dur) / math.log1p(12)
        scores.append(score)

    qs = sum(scores)   # Fix #4: capture before appending so log1p can use same value
    quality_rows.append({
        "candidate_id"       : cid,
        "avg_ai_duration"    : np.mean([s.get("duration_months",0) for s in ai_skills]),
        "advanced_ai_skills" : sum(1 for s in ai_skills if s.get("proficiency") in ("advanced","expert")),
        "expert_ai_skills"   : sum(1 for s in ai_skills if s.get("proficiency") == "expert"),
        "quality_score"      : qs,
        "quality_score_log"  : math.log1p(qs),  # Fix #4: log1p normalizes skewed dist (max=55, mean=1.84)
        "max_endorsements_ai": max(s.get("endorsements",0) for s in ai_skills),
    })

quality_df = pd.DataFrame(quality_rows)
print("Quality features shape:", quality_df.shape)
print(quality_df[["avg_ai_duration","advanced_ai_skills","expert_ai_skills","quality_score","quality_score_log"]].describe().round(2))

# %% [markdown]
# **Design Note 3.1 — The Quality Score Formula**
# ```
# quality_score += PROF_WEIGHT[proficiency] × log1p(duration_months) / log1p(12)
# ```
# - `PROF_WEIGHT`: expert=3.0 > advanced=2.0 > intermediate=1.0 > beginner=0.5
#   (EDA Obs 14.3)
# - `log1p(duration) / log1p(12)`: normalises duration so 12 months = 1.0 baseline.
#   Log scale prevents a 10-year candidate from dominating purely by tenure.
# - Summed across all AI skills, so a candidate with 3 expert retrieval skills will
#   significantly outscore one with 6 beginner skills.
# 
# **Design Note 3.2 — quality_score vs ai_skill_total**
# `ai_skill_total` (Phase 2) asks: *how many AI skills does this candidate have?*
# `quality_score` (Phase 3) asks: *how credible is each skill?*
# 
# Example from EDA:
# - CAND_0000021 (Project Manager): FAISS `intermediate` 8mo, LangChain `intermediate` 7mo
# - CAND_0000031 (Rec Systems Eng): FAISS `advanced` 35mo, Embeddings `expert` 60mo
# 
# Phase 2 scores them similarly. Phase 3 separates them correctly.
# 
# **Design Note 3.3 — max_endorsements_ai**
# Endorsements are a weak proxy for peer validation. High endorsements on AI skills
# from colleagues is a mild positive signal. Kept as a separate feature rather than
# folded into quality_score — gives Notebook 03 flexibility to weight or drop it.

# %% [markdown]
# ## 📊 Phase 4 — Assessment Features

# %%
assessment_rows = []

for c in candidates:
    cid    = c["candidate_id"]
    scores = c["redrob_signals"].get("skill_assessment_scores", {})

    all_scores = list(scores.values())
    ai_scores  = [v for k, v in scores.items() if k in AI_ALL_SKILLS]

    assessment_rows.append({
        "candidate_id"          : cid,
        "has_assessment"        : int(len(all_scores) > 0),
        "has_ai_assessment"     : int(len(ai_scores) > 0),
        "assessment_count"      : len(all_scores),
        "avg_assessment_score"  : np.mean(all_scores)  if all_scores  else 0.0,
        "avg_ai_assessment_score": np.mean(ai_scores)  if ai_scores   else 0.0,
    })

assess_df = pd.DataFrame(assessment_rows)
print("Assessment features shape:", assess_df.shape)
print(assess_df[["has_assessment","has_ai_assessment","avg_assessment_score","avg_ai_assessment_score"]].describe().round(2))
n_any = assess_df["has_assessment"].sum()
n_ai  = assess_df["has_ai_assessment"].sum()
print(f"\nCoverage: {n_any:,} ({100*n_any/N:.1f}%) have any assessment")
print(f"Coverage: {n_ai:,}  ({100*n_ai/N:.1f}%) have AI-skill assessment")

# %% [markdown]
# **Design Note 4.1 — Assessment Scores Are Sparse But Gold**
# Coverage is low (EDA Obs 14.4 projected ~15-20% on the full dataset).
# When present, these scores are third-party platform-verified — they cannot be
# self-reported or inflated. A candidate with `FAISS: 87` in their assessments
# is categorically stronger evidence than one who merely lists FAISS as a skill.
# 
# **Design Note 4.2 — Two Separate Assessment Signals**
# `avg_assessment_score` → general platform engagement (writes code, takes tests)
# `avg_ai_assessment_score` → domain-specific verified evidence
# 
# These are kept separate because a candidate who aced general assessments but
# has no AI assessments is a different profile from one who aced AI assessments specifically.
# 
# **Design Note 4.3 — Zero Fill for Missing Assessments**
# Candidates without assessments receive 0, not NaN.
# Rationale: the absence of an assessment is a real signal — they haven't demonstrated
# platform engagement. Using NaN would require imputation in Notebook 03, which adds
# noise. Zero preserves the "not demonstrated" state cleanly.

# %% [markdown]
# ## 📡 Phase 5 — Behavioral Features

# %%
behavior_rows = []

# Compute pool-wide max for normalization (EDA Obs 8.5: range is wide)
max_saved = max(c["redrob_signals"]["saved_by_recruiters_30d"]  for c in candidates)
max_views = max(c["redrob_signals"]["profile_views_received_30d"] for c in candidates)
max_appear = max(c["redrob_signals"]["search_appearance_30d"]   for c in candidates)

print(f"Pool-wide maxima for normalization:")
print(f"  saved_by_recruiters_30d   : {max_saved}")
print(f"  profile_views_received_30d: {max_views}")
print(f"  search_appearance_30d     : {max_appear}")

# %%
for c in candidates:
    sig = c["redrob_signals"]

    rr   = sig["recruiter_response_rate"]       # already 0-1
    ic   = sig["interview_completion_rate"]      # already 0-1
    saved_norm = sig["saved_by_recruiters_30d"]  / max_saved  if max_saved > 0 else 0
    views_norm = sig["profile_views_received_30d"] / max_views if max_views > 0 else 0
    appear_norm= sig["search_appearance_30d"]    / max_appear  if max_appear > 0 else 0

    # Composite behavior score — weights reflect EDA Obs 13.2 and 13.5
    # saved_by_recruiters is weighted highest: direct market validation (Obs 13.3)
    behavior_score = (
        0.30 * rr
      + 0.35 * saved_norm
      + 0.20 * ic
      + 0.10 * views_norm
      + 0.05 * appear_norm
    )

    behavior_rows.append({
        "candidate_id"          : c["candidate_id"],
        "recruiter_response_rate": rr,
        "interview_completion_rate": ic,
        "saved_by_recruiters_norm": saved_norm,
        "profile_views_norm"    : views_norm,
        "search_appearance_norm": appear_norm,
        "behavior_score"        : behavior_score,
    })

behavior_df = pd.DataFrame(behavior_rows)
print("Behavioral features shape:", behavior_df.shape)
print(behavior_df[["recruiter_response_rate","interview_completion_rate",
                    "saved_by_recruiters_norm","behavior_score"]].describe().round(3))

# %% [markdown]
# **Design Note 5.1 — Normalization Strategy**
# `recruiter_response_rate` and `interview_completion_rate` are already 0-1 fractions.
# `saved_by_recruiters_30d`, `profile_views_received_30d`, `search_appearance_30d`
# are counts — divided by pool-wide max to bring into 0-1 range.
# This preserves the relative ordering within the pool while making features comparable.
# 
# **Design Note 5.2 — Composite Weights**
# `saved_by_recruiters_30d` receives the highest weight (0.35) because it represents
# independent market validation from multiple recruiters (EDA Obs 13.3).
# `recruiter_response_rate` is second (0.30) — engagement signal (EDA Obs 8.2).
# `interview_completion_rate` third (0.20) — reliability signal (EDA Obs 8.6).
# These weights are hypotheses for Notebook 03 to test.
# 
# **Design Note 5.3 — Behavioral Signals Are Independent of AI Skill Depth**
# EDA Obs 13.7 showed that `ai_total` and `saved_by_recruiters` have near-zero
# Pearson correlation. Behavioral signals carry independent information.
# Do NOT collapse behavior_score into ai_skill_total — they must remain separate features.

# %% [markdown]
# ## 📍 Phase 6 — Availability Features

# %%
def location_score(country, location):
    """
    Score geographic fit against JD requirements.
    EDA Obs 15.3 + JD: Pune/Noida preferred, India major cities acceptable.
    Outside India: case-by-case, no visa sponsorship.
    """
    if country != "India":
        return 0.3   # case-by-case per JD

    city = location.split(",")[0].strip().lower()
    if city in PREFERRED_CITIES:   return 1.0
    if city in ACCEPTABLE_CITIES:  return 0.8
    return 0.6   # India but other city

avail_rows = []

for c in candidates:
    sig = c["redrob_signals"]
    p   = c["profile"]

    # Recency (EDA Obs 15.1 — majority inactive >90 days)
    last_active_raw = sig.get("last_active_date")
    if not last_active_raw:
        days_inactive = 365          # treat missing as fully stale
    else:
        last_active   = date.fromisoformat(last_active_raw)
        days_inactive = (REFERENCE_DATE - last_active).days
    # Decay: 1.0 at 0 days → 0.2 at 365 days (minimum floor 0.2)
    recency_score = max(0.2, 1.0 - days_inactive / 365.0)

    # Notice period (JD: sub-30 ideal, 30+ bar rises)
    notice = sig["notice_period_days"]
    if notice <= 30:    notice_score = 1.0
    elif notice <= 60:  notice_score = 0.7
    elif notice <= 90:  notice_score = 0.4
    else:               notice_score = 0.2

    # Work mode (JD: hybrid preferred)
    wm = sig.get("preferred_work_mode","flexible")
    work_mode_score = {"hybrid":1.0, "flexible":0.9, "onsite":0.7, "remote":0.6}.get(wm, 0.7)

    # Openness / verification signals
    openness = (
        0.4 * int(sig["open_to_work_flag"])
      + 0.2 * int(sig["willing_to_relocate"])
      + 0.2 * int(sig["verified_email"])
      + 0.1 * int(sig["verified_phone"])
      + 0.1 * int(sig["linkedin_connected"])
    )

    loc_score = location_score(p["country"], p.get("location",""))

    # Composite availability score
    # Recency is the most critical gate (EDA Obs 15.4)
    avail_score = (
        0.35 * recency_score
      + 0.25 * openness
      + 0.20 * loc_score
      + 0.10 * notice_score
      + 0.10 * work_mode_score
    )

    avail_rows.append({
        "candidate_id"   : c["candidate_id"],
        "days_since_active": days_inactive,
        "recency_score"  : recency_score,
        "notice_period"  : notice,
        "notice_score"   : notice_score,
        "work_mode_score": work_mode_score,
        "location_score" : loc_score,
        "openness_score" : openness,
        "availability_score": avail_score,
    })

avail_df = pd.DataFrame(avail_rows)
print("Availability features shape:", avail_df.shape)
print(avail_df[["recency_score","notice_score","location_score","openness_score","availability_score"]].describe().round(3))

# %% [markdown]
# **Design Note 6.1 — Availability Is a Multiplier Gate, Not an Additive Score**
# `availability_score` is computed as a standalone score here.
# In Notebook 03 it will be used as a **multiplier** on the final skill/semantic score:
# ```python
# final_score = skill_fit_score * availability_score
# ```
# A candidate who is perfect on skills but inactive for 11 months gets scaled down.
# A candidate who is actively looking with sub-30 day notice gets full weight.
# This matches the JD's explicit statement (EDA Obs 15.1).
# 
# **Design Note 6.2 — Recency Has the Highest Weight (0.35)**
# `days_since_active` is the most critical availability gate.
# The decay function `max(0.2, 1 - days/365)` gives a floor of 0.2 to avoid
# completely zeroing out candidates who may just not use the platform actively.
# 
# **Design Note 6.3 — Location Score Is Independent**
# Country and city are scored strictly against JD requirements.
# Outside-India candidates always receive 0.3 — not 0 — because the JD says
# "case-by-case" not "never." India candidates in wrong cities get 0.6, not 0.

# %% [markdown]
# ## ⚠️ Phase 7 — Sentinel Value Features

# %%
sentinel_rows = []
# ⚠️  Fix #5 — RANKING RULE:
#   Use has_github and has_offer_history as PRIMARY signals in Notebook 03.
#   Do NOT use github_score_clean or offer_acceptance_clean directly — they have
#   64,637 and 59,554 NaN values respectively (~65% and ~60% of candidates).
#   Using them raw creates unequal comparisons: NaN candidates get unfairly penalized
#   or ignored. The binary has_X flags are the safe primary signal.
#   github_score_clean / offer_acceptance_clean can only be used as a SECONDARY
#   tiebreaker when both candidates have has_X = 1.

for c in candidates:
    sig = c["redrob_signals"]

    # GitHub (EDA Obs 9.1: -1 = no GitHub linked, NOT negative activity)
    raw_github = sig["github_activity_score"]
    has_github       = int(raw_github != -1)
    github_score_clean = raw_github if raw_github != -1 else np.nan

    # Offer acceptance (EDA Obs 9.4: -1 = no offer history)
    raw_offer = sig["offer_acceptance_rate"]
    has_offer_history     = int(raw_offer != -1)
    offer_acceptance_clean = raw_offer if raw_offer != -1 else np.nan

    sentinel_rows.append({
        "candidate_id"          : c["candidate_id"],
        "has_github"            : has_github,
        "github_score_clean"    : github_score_clean,
        "has_offer_history"     : has_offer_history,
        "offer_acceptance_clean": offer_acceptance_clean,
    })

sentinel_df = pd.DataFrame(sentinel_rows)

print("Sentinel features shape:", sentinel_df.shape)
print(f"  has_github         : {sentinel_df['has_github'].mean():.1%} of candidates")
print(f"  has_offer_history  : {sentinel_df['has_offer_history'].mean():.1%} of candidates")
print(f"  github_score (mean, where present): {sentinel_df['github_score_clean'].mean():.2f}")
print(f"  offer_rate   (mean, where present): {sentinel_df['offer_acceptance_clean'].mean():.2f}")

# %% [markdown]
# **Design Note 7.1 — Never Use Raw -1 Values**
# Running `MinMaxScaler()` on raw `github_activity_score` or `offer_acceptance_rate`
# would treat `-1` as a very low numeric score. It isn't — it means "no data."
# (EDA Observations 9.1 and 9.4)
# 
# This phase creates two features per sentinel column:
# - `has_X` (binary): presence of the signal itself is informative
# - `X_clean` (float | NaN): the actual value when present
# 
# **Design Note 7.2 — NaN Is Intentional Here**
# Unlike Phase 4 (assessments) where 0 fill was appropriate, NaN is correct here
# because the signal was explicitly not collected. Notebook 03 can decide whether to
# impute with the pool mean, fill with 0, or train a model that handles NaN natively.
# 
# **Design Note 7.3 — has_github as a Proxy for Technical Engagement**
# For AI/ML engineers, having an active GitHub is a mild positive signal.
# `has_github = 1` → candidate at minimum linked their GitHub.
# `github_score_clean` (when present) tells us how active it is.

# %% [markdown]
# ## 🔎 Phase 8 — Career Features & Honeypot Detection

# %%
def get_consulting_ratio(c):
    """
    Returns the fraction of total career months spent at consulting/IT services firms.
    0.0 = no consulting experience   1.0 = entirely consulting
    More nuanced than a binary flag — a candidate who went Infosys → Uber → Swiggy
    gets 0.33, not 1.0 (EDA Obs 11.4; reviewer Fix 3).
    """
    total_months = sum(j.get("duration_months", 0) for j in c["career_history"])
    if total_months == 0:
        return 0.0
    consulting_months = sum(
        j.get("duration_months", 0)
        for j in c["career_history"]
        if any(cf in j["company"].lower() for cf in CONSULTING_FIRMS)
    )
    return consulting_months / total_months

def detect_honeypot(c):
    """
    Detect subtly impossible profiles (submission spec: ~80 honeypots in 100k).
    Red flags (from submission spec):
      1. Expert proficiency with 0 months used
      2. Suspiciously high expert-skill count (>8) with short durations
      3. Total career duration_months >> years_of_experience * 12 (impossible overlap)
    """
    flags = 0

    # Flag 1: expert + 0 months
    for s in c.get("skills", []):
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0:
            flags += 2

    # Flag 2: many expert skills with very short durations
    expert_skills = [s for s in c.get("skills",[]) if s.get("proficiency")=="expert"]
    short_experts = [s for s in expert_skills if s.get("duration_months",0) < 6]
    if len(short_experts) >= 3:
        flags += 1

    # Flag 3: impossible career timeline
    total_career_mo = sum(j.get("duration_months", 0) for j in c.get("career_history",[]))
    experience_mo   = c["profile"]["years_of_experience"] * 12
    if total_career_mo > experience_mo * 1.5 and total_career_mo > 36:
        flags += 1

    return int(flags >= 2)

# %%
career_rows = []

for c in candidates:
    exp  = c["profile"]["years_of_experience"]
    jobs = c["career_history"]

    career_rows.append({
        "candidate_id"      : c["candidate_id"],
        "experience_years"  : exp,
        "job_count"         : len(jobs),
        "in_experience_range": int(5 <= exp <= 9),   # JD target (EDA Obs 3.2)
        "consulting_ratio"  : get_consulting_ratio(c),
        "is_honeypot"       : detect_honeypot(c),
        "career_depth_score": min(exp / 9.0, 1.0),   # normalized 0-1, capped at 9yr
    })

career_df = pd.DataFrame(career_rows)

print("Career features shape:", career_df.shape)
n_honeypot    = career_df["is_honeypot"].sum()
n_range       = career_df["in_experience_range"].sum()
high_consult  = (career_df["consulting_ratio"] >= 0.8).sum()
full_consult  = (career_df["consulting_ratio"] == 1.0).sum()
print(f"  in_experience_range      : {n_range:,}  ({100*n_range/N:.1f}%)")
print(f"  consulting_ratio >= 0.8  : {high_consult:,}  ({100*high_consult/N:.1f}%)")
print(f"  consulting_ratio == 1.0  : {full_consult:,}  ({100*full_consult/N:.1f}%)")
print(f"  is_honeypot detected     : {n_honeypot:,}  ({100*n_honeypot/N:.1f}%)")
print()
print(career_df["consulting_ratio"].describe().round(3))

# %% [markdown]
# **Design Note 8.1 — Honeypot Detection Is a Hard Filter**
# The submission spec warns: >10% honeypots in top 100 = disqualification.
# `is_honeypot` candidates will receive a score multiplier of ~0.05 in Notebook 03
# (near-zero but not zero, to preserve rank ordering of true candidates above them).
# 
# The three detection rules target the patterns from the spec:
# 1. `expert` proficiency + 0 months used → impossible (skills require practice)
# 2. 3+ expert skills all under 6 months → statistically implausible
# 3. Career duration significantly exceeds stated experience → inconsistent record
# 
# **Design Note 8.2 — consulting_ratio: Continuous, Not Binary**
# Previous design used a binary `is_consulting_only` flag. Replaced with `consulting_ratio`
# (consulting months / total career months) because:
# 
# - A candidate with career path `Infosys → Uber → Swiggy` would have been wrongly flagged
#   as consulting-only by the binary version
# - `consulting_ratio = 0.33` correctly reflects partial exposure
# - Notebook 03 can apply a graded penalty: e.g. `score *= max(0.2, 1 - consulting_ratio)`
#   which fully penalises ratio=1.0 but barely penalises ratio=0.1
# 
# JD source: *"People who have only worked at TCS, Infosys, Wipro..."*
# (EDA Obs 11.4 — note the word "only")
# 
# **Design Note 8.3 — Experience Is a Soft Gate, Not a Score**
# EDA Obs 3.3: experience has low discriminative power within the 5–9 year range
# because most candidates already satisfy it. `in_experience_range` captures
# the binary gate; `career_depth_score` is a gentle normalised continuous version.
# Neither should dominate the final ranking formula.

# %% [markdown]
# ## 🏆 Phase 9 — Certification Features

# %%
cert_rows = []

for c in candidates:
    certs     = c.get("certifications", [])
    cert_names = {cert["name"] for cert in certs}
    ai_cert_names = cert_names & AI_CERTS
    ai_cert_count = len(ai_cert_names)

    cert_rows.append({
        "candidate_id"    : c["candidate_id"],
        "has_certification": int(len(certs) > 0),
        "has_ai_cert"     : int(ai_cert_count > 0),
        "ai_cert_count"   : ai_cert_count,
    })

cert_df = pd.DataFrame(cert_rows)

print("Certification features shape:", cert_df.shape)
n_any_cert = cert_df["has_certification"].sum()
n_ai_cert  = cert_df["has_ai_cert"].sum()
print(f"  has_certification : {n_any_cert:,} ({100*n_any_cert/N:.1f}%)")
print(f"  has_ai_cert       : {n_ai_cert:,}  ({100*n_ai_cert/N:.2f}%)")
print()
print(cert_df[["has_certification","has_ai_cert","ai_cert_count"]].describe().round(3))

# %% [markdown]
# **Design Note 9.1 — Generic Certs Are Noise**
# EDA Obs 13.3: `AWS Cloud Practitioner`, `Six Sigma Green Belt`, `Scrum Master`
# each appear ~12,000 times — uniformly distributed. They carry no differential signal.
# `has_certification` (binary) is kept only as a profile completeness proxy.
# 
# **Design Note 9.2 — AI-Specific Certs Are Rare and Valuable**
# EDA Obs 13.4: AI-specific certs appear only 100–130 times each in the full 100k pool.
# `has_ai_cert` and `ai_cert_count` are the only cert features that matter.
# These will receive a meaningful bonus weight in Notebook 03.
# 
# **Design Note 9.3 — Cert Is Tiebreaker, Not Primary Signal**
# Certifications confirm existing knowledge — they don't create it.
# A strong candidate without AI certs ranks above a weak candidate with them.
# These features should have low weight in the final formula.

# %% [markdown]
# ## 🧠 Phase 10 — Semantic Matching (JD vs Candidate Text)

# %%
# Install sentence-transformers if not present
import subprocess, sys
try:
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers already installed")
except ImportError:
    print("Installing sentence-transformers...")
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "sentence-transformers", "--quiet"], check=True)
    from sentence_transformers import SentenceTransformer
    print("Installed OK")

# %%
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_CACHE = "outputs/semantic_similarity.npy"

if os.path.exists(SIMILARITY_CACHE):
    similarities = np.load(SIMILARITY_CACHE, allow_pickle=False)
    print(f"Loaded cached similarities: {len(similarities):,}")
else:
    print("Loading model: all-MiniLM-L6-v2 (22M params, CPU-friendly)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding JD...")
    jd_emb = model.encode([JD_TEXT], show_progress_bar=False, convert_to_numpy=True)

    print(f"Encoding {N:,} candidate texts (this will take ~5-8 min on CPU)...")
    cand_embs = model.encode(
        candidate_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Computing cosine similarities...")
    similarities = cosine_similarity(jd_emb, cand_embs)[0]

    np.save(SIMILARITY_CACHE, similarities)
    print(f"Saved to {SIMILARITY_CACHE}")

print(f"\nSemantic similarity stats:")
print(f"  Mean   : {similarities.mean():.4f}")
print(f"  Median : {np.median(similarities):.4f}")
print(f"  Max    : {similarities.max():.4f}")
print(f"  Min    : {similarities.min():.4f}")
print(f"  >0.5   : {(similarities>0.5).sum():,} candidates")
print(f"  >0.7   : {(similarities>0.7).sum():,} candidates")

# %%
semantic_df = pd.DataFrame({
    "candidate_id"       : [c["candidate_id"] for c in candidates],
    "semantic_similarity": similarities,
})

# ─── Fix #2: semantic_percentile is the correct column to use in Notebook 03 ───
# Raw cosine similarity compresses into a narrow range (e.g. 0.18–0.72).
# Percentile rank spreads candidates uniformly 0.0–1.0 → more stable ranking signal.
# ⚠️  Notebook 03 must use semantic_percentile, NOT semantic_similarity.
semantic_df["semantic_percentile"] = semantic_df["semantic_similarity"].rank(pct=True)

print("Semantic features shape:", semantic_df.shape)
print()
print("semantic_similarity:")
print(semantic_df["semantic_similarity"].describe().round(4))
print()
print("semantic_percentile:")
print(semantic_df["semantic_percentile"].describe().round(4))
print(f"\nCandidates in top 10th percentile: {(semantic_df['semantic_percentile'] >= 0.90).sum():,}")
print(f"Candidates in top  5th percentile: {(semantic_df['semantic_percentile'] >= 0.95).sum():,}")

# %% [markdown]
# **Design Note 10.1 — Why Semantic Matching Is the Most Important Phase**
# EDA Obs 13.11 and the JD itself confirm:
# > *"A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their profile,
# > but if their career history shows they built a recommendation system at a product
# > company, they're a fit."*
# 
# Semantic similarity captures this. A candidate who wrote:
# *"improved NDCG@10 by 12% on the recommendation system through hybrid retrieval"*
# will score very high against JD_TEXT even without those exact skill keywords.
# A keyword-stuffer who listed every AI skill but has no career evidence will score lower.
# 
# **Design Note 10.2 — Model Choice: all-MiniLM-L6-v2**
# - 22M parameters → fast on CPU (~5-8 min for 100k on a modern laptop)
# - 384-dimensional embeddings → 150 MB RAM for full pool
# - Sufficient quality for domain matching
# - Alternative: `all-mpnet-base-v2` (better quality, 3× slower — use if time permits)
# 
# **Design Note 10.3 — Cache Strategy**
# The first run saves embeddings to `outputs/semantic_similarity.npy`.
# Subsequent runs load from cache (<1 second). This is critical for the
# submission pipeline which must complete in 5 minutes CPU-only.
# 
# **Design Note 10.4 — Semantic Score Is Not a Replacement for Structured Features**
# A candidate whose text happens to use general ML words scores moderately even
# without real retrieval experience. Semantic similarity works best as one layer
# in the final composite score, not as the sole signal.

# %% [markdown]
# ## 🔧 Phase 11 — Feature Assembly & Export

# %%
# Merge all feature dataframes on candidate_id
dfs = [
    skill_df,
    quality_df,
    assess_df,
    behavior_df,
    avail_df,
    sentinel_df,
    career_df,
    cert_df,
    semantic_df,
]

features_df = dfs[0]
for df in dfs[1:]:
    features_df = features_df.merge(df, on="candidate_id", how="left")

print(f"Feature table shape: {features_df.shape}")
print(f"  Candidates : {len(features_df):,}")
print(f"  Features   : {features_df.shape[1] - 1}  (excl. candidate_id)")
print(f"  Nulls      : {features_df.isnull().sum().sum()}")

# Integrity assertions — catch merge explosions immediately
assert len(features_df) == N, \
    f"Row count mismatch after merge: {len(features_df)} != {N}"
assert features_df["candidate_id"].nunique() == N, \
    "Duplicate candidate_ids detected after merge"
print("Assertions passed ✅  — shape and uniqueness confirmed")

# %%
# Feature group registry — Notebook 03 uses this to select/scale by group
feature_groups = {
    "skill"      : [c for c in skill_df.columns      if c != "candidate_id"],
    # skill group now includes: retrieval_score, recommendation_score (separate - Imp 2),
    # evaluation_signal_score (Imp 1 ★), production_signal_score (Imp 3 ★),
    # career_keyword_score (Imp 4 ★), hidden_signal_bonus
    "quality"    : [c for c in quality_df.columns    if c != "candidate_id"],
    "assessment" : [c for c in assess_df.columns     if c != "candidate_id"],
    "behavior"   : [c for c in behavior_df.columns   if c != "candidate_id"],
    "availability": [c for c in avail_df.columns     if c != "candidate_id"],
    "sentinel"   : [c for c in sentinel_df.columns   if c != "candidate_id"],
    "career"     : [c for c in career_df.columns     if c != "candidate_id"],
    "cert"       : [c for c in cert_df.columns       if c != "candidate_id"],
    "semantic"   : [c for c in semantic_df.columns   if c != "candidate_id"],
}

print("Feature groups:")
for group, cols in feature_groups.items():
    print(f"  {group:<12s}: {cols}")

# %%
# Null audit — only sentinel NaN values should remain
null_summary = features_df.isnull().sum()
null_cols = null_summary[null_summary > 0]
print("Columns with NaN (expected: only sentinel clean columns):")
print(null_cols.to_string())

# %%
# Final column list
print("\nAll feature columns:")
non_id_cols = [c for c in features_df.columns if c != "candidate_id"]
for i, col in enumerate(non_id_cols, 1):
    dtype = features_df[col].dtype
    n_null = features_df[col].isnull().sum()
    null_str = f" [{n_null:,} NaN]" if n_null > 0 else ""
    print(f"  {i:2d}. {col:<35s} {str(dtype):<10}{null_str}")

# %%
# Export
features_df.to_pickle("outputs/features_df.pkl")
features_df.to_csv("outputs/features_df.csv", index=False)

print("Saved:")
print("  outputs/features_df.pkl")
print("  outputs/features_df.csv")
print()
print("Preview (top 5 rows, key columns):")
preview_cols = [
    "candidate_id","retrieval_score","llm_score","ml_score",
    "quality_score","behavior_score","availability_score",
    "semantic_similarity","consulting_ratio","is_honeypot"
]
print(features_df[preview_cols].head().to_string(index=False))

# %% [markdown]
# **Design Note 11.1 — Feature Table Summary**
# 
# | Group | Features | Source |
# |---|---|---|
# | Skill breadth | `retrieval_score`, `recommendation_score` (separate), `llm_score`, `ml_score`, `ai_skill_total` | Phase 2 |
# | Evaluation evidence ★ | `evaluation_signal_score` — NDCG/MRR/MAP/A-B in career text | Phase 2 |
# | Production evidence ★ | `production_signal_score` — deployed/shipped/live traffic in career text | Phase 2 |
# | Domain vocabulary ★ | `career_keyword_score` — fraction of core IR keywords found | Phase 2 |
# | Skill depth | `avg_ai_duration`, `advanced_ai_skills`, `expert_ai_skills`, `quality_score` | Phase 3 |
# | Platform verified | `has_assessment`, `avg_ai_assessment_score` | Phase 4 |
# | Behavioral | `recruiter_response_rate`, `saved_norm`, `behavior_score` | Phase 5 |
# | Availability | `recency_score`, `notice_score`, `location_score`, `availability_score` | Phase 6 |
# | Sentinel | `has_github`, `github_score_clean`, `has_offer_history` | Phase 7 |
# | Career | `experience_years`, `consulting_ratio`, `is_honeypot`, `career_depth_score` | Phase 8 |
# | Certifications | `has_ai_cert`, `ai_cert_count` | Phase 9 |
# | Semantic | `semantic_similarity`, `semantic_percentile` | Phase 10 |
# 
# **Design Note 11.2 — What Notebook 03 Receives**
# `features_df` is the complete, clean input to the ranking notebook.
# Notebook 03 needs to:
# 1. Design the composite scoring formula using these features
# 2. Apply hard filters (`is_honeypot`) and graded consulting penalty (`consulting_ratio`)
# 3. Apply the availability multiplier
# 4. Produce the top-100 ranked list
# 5. Generate the reasoning column from raw profile fields
# 
# No raw candidate JSON is needed in Notebook 03 — everything is in `features_df`,
# except reasoning generation which reads back specific fields for the submission.
# 
# **Design Note 11.3 — Sentinel NaN Values Are Intentional**
# `github_score_clean` and `offer_acceptance_clean` will contain NaN for ~65% and ~60%
# of candidates respectively. This is correct. Notebook 03 can use their binary
# counterparts (`has_github`, `has_offer_history`) as primary features and handle
# NaN in the continuous version explicitly.

# %% [markdown]
# ---
# ## ✅ Notebook 02 Complete
# 
# ```
# Raw candidates.jsonl (100k)
#          ↓
# Phase 1  candidate_text
# Phase 2  skill category scores
# Phase 3  skill quality scores
# Phase 4  assessment scores
# Phase 5  behavioral composite
# Phase 6  availability composite
# Phase 7  sentinel-safe features
# Phase 8  career + honeypot flags
# Phase 9  certification signals
# Phase 10 semantic JD similarity
#          ↓
# outputs/features_df.pkl  ← Notebook 03 input
# ```
# 
# **Next: Notebook 03 — Ranking Formula + Top-100 Generation**

# %% [markdown]
# ## ✅ Fixes Applied (Pre-Notebook-03)
# 
# All 5 reviewer fixes implemented directly in this notebook:
# 
# | Fix | What changed | Where |
# |---|---|---|
# | #1 | Header doc updated: `100k × 47 features + candidate_id` | Cell header |
# | #2 | Explicit `⚠️` note: use `semantic_percentile`, NOT `semantic_similarity` | Phase 10 |
# | #3 | `hidden_signal_bonus = min(count × 0.03, 0.15)` added as column | Phase 2 |
# | #4 | `quality_score_log = log1p(quality_score)` added as column | Phase 3 |
# | #5 | Explicit `⚠️` warning: use `has_github` / `has_offer_history`, not `_clean` | Phase 7 |
# 
# **Notebook 03 receives a clean, ready-to-rank feature table.**
# 

# %% [markdown]
# 


