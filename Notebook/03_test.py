# %% [markdown]
# # 🎯 Notebook 03 — Recruiter Decision Engine
# ### REDRO AI Hackathon | features_df → Top 100 → submission.csv
# 
# Central question for every candidate: *Would a strong recruiter spend an interview slot on this person?*
# 
# Four engines answer this:
# - **Capability** — Evidence they can do the job (production retrieval + evaluation work)
# - **Validation** — Market has already confirmed them (recruiter behavior)
# - **Availability** — Can we actually hire them (recency, notice, location)
# - **Risk** — Reasons not to hire (consulting background, honeypots)
# 
# | Input | `outputs/features_df.pkl` | Output | `outputs/submission.csv` |
# |---|---|---|---|
# | Runtime | < 1 minute | Strategy | Evidence > keywords |
# 
# ---

# %% [markdown]
# ## ⚙️ Phase 0 — Load & Integrity Check

# %%
import json, os, warnings, math
from datetime import date
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
os.makedirs("outputs", exist_ok=True)
print("Imports OK")

# %%
df = pd.read_pickle("EDA/outputs/features_df.pkl")
N  = len(df)

print(f"Shape      : {df.shape}")
print(f"Candidates : {N:,}")
print(f"Features   : {df.shape[1]-1}")

assert df["candidate_id"].nunique() == N, "Duplicate candidate_ids detected!"
print("Duplicates : 0 ✅")

# Expected NaN only in sentinel-encoded columns
null_cols = df.isnull().sum()
null_cols = null_cols[null_cols > 0]
print("\nNaN columns (expected):")
for col, n in null_cols.items():
    print(f"  {col:<35}: {n:,} NaN")

# %%
DATASET_PATH = "raw_dataset/candidates.jsonl"
print(f"Loading candidate profiles for reasoning generation...")
candidates_lookup = {}
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            c = json.loads(line)
            candidates_lookup[c["candidate_id"]] = c
print(f"Loaded: {len(candidates_lookup):,} candidates")

# %%
# Product-company ratio: fraction of career months at NAMED product companies
# Named list is more precise than industry classification (reviewer-specified, Fix 2)
PRODUCT_COMPANIES = {
    "google","amazon","uber","swiggy","zomato","flipkart",
    "microsoft","netflix","meta","apple","linkedin","twitter",
    "spotify","airbnb","stripe","instacart","doordash","lyft",
    "salesforce","adobe","atlassian","shopify","square","paypal",
}

def compute_product_ratio(c):
    """
    Fraction of career months at known product companies.
    Returns 0.0 to 1.0. Product company experience gets a +10% multiplier
    on final_score (not added to capability — it is a separate gate).
    """
    jobs = c.get("career_history", [])
    total_mo = sum(j.get("duration_months", 0) for j in jobs)
    if total_mo == 0:
        return 0.0
    product_mo = sum(
        j.get("duration_months", 0) for j in jobs
        if any(pc in j.get("company", "").lower() for pc in PRODUCT_COMPANIES)
    )
    return product_mo / total_mo

product_ratio_map = {cid: compute_product_ratio(c) for cid, c in candidates_lookup.items()}
print(f"Product-company ratio computed for {len(product_ratio_map):,} candidates")
sample_vals = list(product_ratio_map.values())
import numpy as np
print(f"  Mean   : {np.mean(sample_vals):.3f}")
print(f"  >0     : {sum(1 for v in sample_vals if v>0):,}  ({100*np.mean([v>0 for v in sample_vals]):.1f}%)")
print(f"  =1.0   : {sum(1 for v in sample_vals if v==1.0):,}  ({100*np.mean([v==1.0 for v in sample_vals]):.1f}%)")

# %% [markdown]
# **Design Note 0.1 — Two Inputs, One Purpose**
# `features_df` drives all scoring. `candidates_lookup` is used only in Phase 9 to
# generate reasoning strings that reference actual profile data (anti-hallucination requirement).
# 
# **Design Note 0.2 — Key Features Added in NB02 v2**
# NB03 is built around four new evidence features from NB02 v2:
# `evaluation_signal_score` (NDCG/MRR/MAP in career text — JD requirement #2),
# `production_signal_score` (deployed/shipped/live — JD requirement #1),
# `career_keyword_score` (IR domain vocabulary density),
# `quality_score_log` (log-normalised skill depth — use this, not raw `quality_score`).

# %% [markdown]
# ## 🔍 Phase 1 — Feature Audit

# %%
AUDIT_COLS = [
    "retrieval_score","evaluation_signal_score","production_signal_score",
    "career_keyword_score","quality_score_log","avg_ai_assessment_score",
    "semantic_percentile","behavior_score","saved_by_recruiters_norm",
    "recruiter_response_rate","availability_score","consulting_ratio","is_honeypot",
]
print("Feature Audit — Key Signal Ranges")
print("=" * 68)
print(df[AUDIT_COLS].describe().T[["mean","50%","min","max"]].rename(columns={"50%":"median"}).round(4).to_string())

# %%
print("Evidence signal sparsity (% candidates with value > 0):")
evidence_cols = [
    "evaluation_signal_score","production_signal_score",
    "career_keyword_score","avg_ai_assessment_score","ai_cert_count",
]
for col in evidence_cols:
    nonzero_pct  = 100 * (df[col] > 0).mean()
    mean_present = df.loc[df[col] > 0, col].mean() if (df[col] > 0).any() else 0
    print(f"  {col:<35}: {nonzero_pct:.1f}% have signal  (mean when present: {mean_present:.4f})")

print()
n_honeypots = df["is_honeypot"].sum()
print(f"Honeypots flagged   : {n_honeypots:,}  ({100*n_honeypots/N:.2f}%)")
print(f"Consulting ≥ 0.8    : {(df['consulting_ratio']>=0.8).sum():,}")
print(f"Inactive > 180 days : {(df['days_since_active']>180).sum():,}")

# %% [markdown]
# **Design Note 1.1 — Evidence Signals Are Intentionally Sparse**
# `evaluation_signal_score` and `production_signal_score` are sparse by design:
# only candidates who actually built and measured retrieval systems wrote these
# keywords in their career descriptions. That sparsity is signal, not noise.
# A candidate who scores 0 on both evidence features has not demonstrated
# the JD's core requirements, regardless of how many AI skills they list.
# 
# **Design Note 1.2 — Why This Changes the Formula**
# Most teams will weight `retrieval_score` (skill keywords) highest.
# The JD says the #1 requirement is *operational production experience*, not skill names.
# This notebook weights `evaluation_signal_score` and `production_signal_score`
# above `retrieval_score` in the capability formula — matching recruiter intent, not keywords.

# %% [markdown]
# ## 📐 Phase 2 — Normalization via Percentile Rank

# %%
ndf = df.copy()

TO_RANK = [
    "retrieval_score","llm_score","ml_score","recommendation_score","ai_skill_total",
    "quality_score_log",          # use log version — NB02 v2 Fix #4
    "avg_ai_duration","advanced_ai_skills","expert_ai_skills","max_endorsements_ai",
    "assessment_count","avg_assessment_score","avg_ai_assessment_score","ai_cert_count",
    "evaluation_signal_score",    # ★ sparse — rank handles this correctly
    "production_signal_score",    # ★ sparse — rank handles this correctly
    "career_keyword_score",
    "hidden_signal_bonus",
    "saved_by_recruiters_norm","profile_views_norm","search_appearance_norm",
]

for col in TO_RANK:
    ndf[f"{col}_pct"] = ndf[col].fillna(0).rank(pct=True, method="average")

print(f"Created {len(TO_RANK)} percentile-rank columns (_pct suffix)")
print()
for col in ["evaluation_signal_score_pct","production_signal_score_pct",
            "quality_score_log_pct","retrieval_score_pct","semantic_percentile"]:
    s = ndf[col]
    print(f"  {col:<42}: median={s.median():.3f}  p90={s.quantile(0.9):.3f}  max={s.max():.3f}")

# ── Derived normalised features ────────────────────────────────────────────────

# 1. Experience fit multiplier (Fix Major #1)
# JD target: 5-9 years. Apply directly; no need to rank.
def experience_multiplier(exp):
    # Softer multipliers — graded, not binary gates
    if 5 <= exp <= 9:   return 1.00   # JD target band: full score
    elif 4 <= exp < 5:  return 0.90   # slightly junior — 10% discount
    elif 9 < exp <= 12: return 0.85   # slightly senior — still viable
    elif 3 <= exp < 4:  return 0.75   # clearly junior
    else:               return 0.60   # <3yr or >12yr — significant mismatch

ndf["experience_fit"] = ndf["experience_years"].apply(experience_multiplier)
print(f"\nexperience_fit distribution (new softer multipliers):")
print(ndf["experience_fit"].value_counts().sort_index().to_string())

# 2. Evaluation signal combo (Fix Major #3: sparse feature handling)
# Binary flag + percentile → avoids pure-pct ceiling effect for zero candidates
ndf["has_evaluation_signal"] = (ndf["evaluation_signal_score"] > 0).astype(float)
ndf["evaluation_signal_combo"] = (
    0.6 * ndf["evaluation_signal_score_pct"] +
    0.4 * ndf["has_evaluation_signal"]
)
print(f"evaluation_signal_combo: median={ndf['evaluation_signal_combo'].median():.3f}")

# 3. Semantic percentile capped (Minor #1: prevent ceiling saturation)
ndf["semantic_pct_capped"] = ndf["semantic_percentile"].clip(upper=0.97)
print(f"semantic_pct_capped: p90={ndf['semantic_pct_capped'].quantile(0.9):.3f}  max={ndf['semantic_pct_capped'].max():.3f}")

# 4. Product company ratio percentile (Missing feature)
ndf["product_company_ratio"] = ndf["candidate_id"].map(product_ratio_map).fillna(0.0)
ndf["product_company_ratio_pct"] = ndf["product_company_ratio"].rank(pct=True, method="average")
print(f"product_company_ratio_pct: median={ndf['product_company_ratio_pct'].median():.3f}")

# %% [markdown]
# **Design Note 2.1 — `rank(pct=True)` vs `MinMaxScaler`**
# For `evaluation_signal_score` where ~80% of values are 0:
# - MinMaxScaler compresses all 80k zeros to 0.0, clusters non-zeros between 0 and 1
# - `rank(pct=True)` gives each zero the average rank of all zeros (~0.40 if 80% are zero),
#   and spreads non-zero values from 0.40 to 1.0 — meaningful separation within each group
# 
# `rank(pct=True)` is more robust to outliers and handles sparsity correctly.
# 
# **Design Note 2.2 — `semantic_percentile` Used Directly**
# `semantic_percentile` is already `rank(pct=True)` of the cosine similarity from NB02 v2.
# No re-ranking needed — use it directly in the scoring formulas.

# %% [markdown]
# ## 🧠 Phase 3 — Capability Engine

# %%
# JD priority order (from actual JD text):
# #1 "Production experience with embeddings retrieval" → production_signal_score
# #2 "Evaluation frameworks (NDCG, MRR, MAP, A/B)" → evaluation_signal_score
# #3 Core tools (vector DBs, hybrid search) → retrieval_score
# The JD says tools dont matter — operational experience does.
# Evidence signals (production, evaluation) outweigh keyword signals (retrieval skills).

# Final capability weights — reviewer-approved (Fix 3 + Fix 4):
# Fix 3: reduce eval 0.13→0.15 and prod 0.13→0.15; increase semantic 0.20→0.25
# Fix 4: increase avg_ai_assessment 0.05→0.09 (assessments >> certs; 9.8% coverage)
# Product company bonus REMOVED from capability — now a final_score multiplier (Fix 2)
CAP_WEIGHTS = {
    "semantic_pct_capped"         : 0.25,   # PRIMARY — JD holistic alignment
    "evaluation_signal_combo"     : 0.15,   # JD req #2 — NDCG/MRR evidence (sparse, reliable)
    "production_signal_score_pct" : 0.15,   # JD req #1 — deployed/shipped
    "retrieval_score_pct"         : 0.18,   # core retrieval tools
    "quality_score_log_pct"       : 0.11,   # skill depth (proficiency × duration)
    "career_keyword_score_pct"    : 0.07,   # domain vocabulary density
    "avg_ai_assessment_score_pct" : 0.09,   # INCREASED — platform-verified, 9.8% coverage
}

assert abs(sum(CAP_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

ndf["capability_score"] = sum(ndf[col] * w for col, w in CAP_WEIGHTS.items())

print("Capability Engine — Weight Table")
print(f"{'Feature':<43} {'Weight':>7}")
print("-" * 52)
for col, w in sorted(CAP_WEIGHTS.items(), key=lambda x: -x[1]):
    print(f"  {col:<41} {w:>7.2f}")
print(f"  {'TOTAL':<41} {sum(CAP_WEIGHTS.values()):>7.2f}")
print()
print(ndf["capability_score"].describe().round(4))

# %% [markdown]
# **Design Note 3.1 — Why Evidence Signals Lead**
# The old formula put `retrieval_score` (0.25) above `evaluation_signal_score` (0.10).
# The JD inverts this: *"the specific tech doesn't matter; the operational experience does."*
# A candidate who mentions FAISS in their skills but never measured a ranking system
# is less qualified than one who wrote "improved NDCG@10 by 12%" in their job description
# but lists no FAISS skill. `evaluation_signal_score` at 0.22 reflects this.
# 
# **Design Note 3.2 — semantic_percentile Captures Context**
# `semantic_percentile` computes JD alignment from the full candidate text including
# career descriptions. It therefore partially captures evaluation and production signals
# already. The 0.22 weight reflects this complementary role — it's not double-counting.

# %% [markdown]
# ## 📡 Phase 4 — Validation Engine

# %%
# Validation engine: Has the market already confirmed this candidate?
# Recruiters save profiles, respond, attend interviews for a reason.
# saved_by_recruiters_30d is the strongest signal — independent market validation.

VAL_WEIGHTS = {
    "saved_by_recruiters_norm_pct" : 0.40,  # strongest: independent recruiter votes
    "recruiter_response_rate"      : 0.30,  # responsiveness signal
    "interview_completion_rate"    : 0.20,  # reliability signal
    "profile_views_norm_pct"       : 0.10,  # passive market interest
}

assert abs(sum(VAL_WEIGHTS.values()) - 1.0) < 1e-9

ndf["validation_score"] = sum(ndf[col] * w for col, w in VAL_WEIGHTS.items())

print("Validation Engine — Weight Table")
print(f"{'Feature':<43} {'Weight':>7}")
print("-" * 52)
for col, w in sorted(VAL_WEIGHTS.items(), key=lambda x: -x[1]):
    print(f"  {col:<41} {w:>7.2f}")
print()
print(ndf["validation_score"].describe().round(4))

# %% [markdown]
# **Design Note 4.1 — Validation Is Independent of Capability**
# EDA-01 Observation 13.7 showed near-zero correlation between `ai_skill_total`
# and `saved_by_recruiters_30d`. This means validation and capability are
# genuinely independent dimensions — neither can proxy for the other.
# A candidate who is technically strong but never responds to recruiters
# is a bad hiring outcome. The validation engine captures this separately.
# 
# **Design Note 4.2 — saved_by_recruiters Is the Strongest Behavioral Signal**
# This is a real-world relevance signal: multiple different recruiters independently
# saved this profile. It cannot be gamed by the candidate the way a high response
# rate can (they could respond quickly to every message). It represents true demand.

# %% [markdown]
# ## 📍 Phase 5 — Availability Engine

# %%
# Availability multiplier: Can we actually hire this candidate?
# Used as a MULTIPLIER on the final score — not an additive component.
# A perfect-on-paper candidate inactive for 11 months is functionally unavailable.
#
# Range: 0.30 (stale/wrong location/long notice) to 1.15 (actively looking)

def compute_availability_multiplier(row):
    base = row["availability_score"]  # composite 0.2-1.0 from NB02

    # Bonuses for high-availability signals
    bonus = 0.0
    if row["openness_score"] >= 0.7:   # actively looking + verified contact
        bonus += 0.05
    if row["recency_score"]  >= 0.9:   # active within last 30 days
        bonus += 0.05
    if row["notice_score"]   >= 0.9:   # sub-30 day notice (JD preferred)
        bonus += 0.05

    return float(np.clip(base + bonus, 0.30, 1.15))

ndf["availability_multiplier"] = ndf.apply(compute_availability_multiplier, axis=1)

print("Availability Multiplier Distribution")
print(ndf["availability_multiplier"].describe().round(4))
print()
above_1 = (ndf["availability_multiplier"] > 1.0).sum()
below_half = (ndf["availability_multiplier"] < 0.5).sum()
print(f"Candidates with multiplier > 1.0  : {above_1:,}  (bonus — actively available)")
print(f"Candidates with multiplier < 0.5  : {below_half:,}  (penalty — functionally unavailable)")

# %% [markdown]
# **Design Note 5.1 — Multiplier Not Additive**
# Availability at 0.15 additive weight would let a brilliant candidate inactive
# for a year still score in the top 10. A multiplier ensures availability gates
# the final score proportionally — a 0.3 multiplier halves even the best base score.
# 
# **Design Note 5.2 — Range Design: 0.30 to 1.15**
# - Floor 0.30: even the worst availability case passes (the JD says "case-by-case"
#   for outside-India, not "never") — no candidate hits absolute 0
# - Ceiling 1.15: small bonus for truly hot candidates (open to work, active today,
#   sub-30 day notice, Pune/Noida based) — signals a recruiter should act now
# 
# **Design Note 5.3 — availability_score from NB02 Is the Foundation**
# `availability_score` already composites recency, notice, location, work_mode, openness.
# The multiplier computes from that foundation — no re-computation needed.

# %% [markdown]
# ## ⚠️ Phase 6 — Risk Engine

# %%
# Risk multiplier: Why should we NOT hire this candidate?
# Two hard risk signals from the JD:
#   1. Consulting-only background ("we have bad fit experience")
#   2. Honeypot profiles (impossible career timelines / expert with 0 months)

def compute_risk_multiplier(row):
    # Consulting penalty: graded by ratio
    # consulting_ratio=1.0 → multiplier=0.20  (JD: "only at TCS/Infosys = bad fit")
    # consulting_ratio=0.5 → multiplier=0.60  (partial exposure, moderate penalty)
    # consulting_ratio=0.0 → multiplier=1.00  (no penalty)
    consulting_multiplier = 1.0 - (0.80 * row["consulting_ratio"])

    # Honeypot penalty: near-zero score to push below all real candidates
    honeypot_multiplier = 0.05 if int(row["is_honeypot"]) == 1 else 1.0

    return float(consulting_multiplier * honeypot_multiplier)

ndf["risk_multiplier"] = ndf.apply(compute_risk_multiplier, axis=1)

print("Risk Multiplier Distribution")
print(ndf["risk_multiplier"].describe().round(4))
print()
pure_consulting = (ndf["consulting_ratio"] >= 1.0).sum()
high_consulting  = (ndf["consulting_ratio"] >= 0.8).sum()
n_hp = ndf["is_honeypot"].sum()
print(f"Pure consulting (ratio=1.0)  : {pure_consulting:,}  (multiplier=0.20)")
print(f"High consulting (ratio≥0.8)  : {high_consulting:,}  (multiplier≤0.36)")
print(f"Honeypots detected           : {n_hp:,}  (multiplier=0.05)")

# %% [markdown]
# **Design Note 6.1 — Consulting Penalty Is Graded**
# Binary `is_consulting_only` was replaced by `consulting_ratio` in NB02 v2.
# `1 - 0.8 × ratio` means:
# - ratio=1.0 → multiplier=0.20 (strong penalty for pure consulting)
# - ratio=0.5 → multiplier=0.60 (moderate — some product company experience)
# - ratio=0.0 → multiplier=1.00 (no penalty)
# 
# This correctly handles the JD's qualifier "only worked at" — partial experience is fine.
# 
# **Design Note 6.2 — Honeypot Floor at 0.05, Not 0**
# Setting honeypots to 0 makes all honeypots identical and randomly ordered.
# 0.05 preserves relative ordering so they all land well below real candidates
# while remaining sortable. Submission spec requires < 10% honeypots in top 100.

# %% [markdown]
# ## 🏆 Phase 7 — Recruiter Score

# %%
# Final composite formula (from NB03 specification):
# base_score = 0.60 * capability + 0.25 * validation + 0.15 * availability
# final_score = base_score * risk_multiplier * availability_multiplier

# Normalise availability_score to 0-1 for additive component
ndf["availability_score_pct"] = ndf["availability_score"].rank(pct=True, method="average")

ndf["base_score"] = (
    0.60 * ndf["capability_score"]
  + 0.25 * ndf["validation_score"]
  + 0.15 * ndf["availability_score_pct"]
)

# Product company bonus multiplier: up to +10% for named companies (Google, Amazon, etc.)
# Applied at final_score level — independent of capability composition
ndf["product_company_multiplier"] = (
    1.0 + 0.10 * ndf["candidate_id"].map(product_ratio_map).fillna(0.0)
)

ndf["final_score"] = (
    ndf["base_score"]
  * ndf["risk_multiplier"]
  * ndf["availability_multiplier"]
  * ndf["experience_fit"]               # JD experience band (0.60–1.00)
  * ndf["product_company_multiplier"]   # product company bonus (1.00–1.10)
)

print("Recruiter Score Summary")
print("=" * 40)
print(ndf[["capability_score","validation_score","base_score","final_score"]].describe().round(4))
print()
print(f"Top-1 percentile threshold  : {ndf['final_score'].quantile(0.99):.4f}")
print(f"Top-5 percentile threshold  : {ndf['final_score'].quantile(0.95):.4f}")
print(f"Median final score          : {ndf['final_score'].median():.4f}")

# %%
# Correlation check: are the four components genuinely independent?
comp_corr = ndf[["capability_score","validation_score","availability_score_pct",
                  "risk_multiplier"]].corr().round(3)
print("Component Correlation Matrix (want low cross-correlations):")
print(comp_corr.to_string())
print()
cap_val_corr = ndf["capability_score"].corr(ndf["validation_score"])
print(f"capability vs validation : {cap_val_corr:.3f}  ", end="")
print("(independent ✅)" if abs(cap_val_corr) < 0.4 else "(correlated ⚠️ — review weights)")

# %% [markdown]
# **Design Note 7.1 — Formula Structure**
# ```
# base_score   = 0.60 × capability + 0.25 × validation + 0.15 × availability
# final_score  = base_score × risk_multiplier × availability_multiplier
# ```
# Capability dominates (60%) because the JD asks a technical question first.
# Validation is second (25%) — market confirmation matters independently.
# Availability is third (15% additive + separate multiplier) — necessary but not sufficient.
# Risk is a gate — bad signals reduce the score multiplicatively.
# 
# **Design Note 7.2 — The Formula Is a Hypothesis**
# These weights are evidence-guided but will be tested in Phase 8 (V1 vs V2).
# If the correlation check shows capability and validation are highly correlated,
# reduce the validation weight and increase the availability weight.

# %% [markdown]
# ## 🎖️ Phase 8 — Tier Assignment

# %%
# Tier assignment based on final_score percentile rank
# Mirrors the JD's honest language: very few candidates are a real fit
#
# Tier 1 = Top 1%   (top 1,000)  — strong fit, should interview
# Tier 2 = Top 5%   (top 5,000)  — good fit, worth considering
# Tier 3 = Top 15%  (top 15,000) — moderate fit
# Tier 4 = Top 30%  (top 30,000) — adjacent skills, unlikely to progress
# Tier 5 = Rest     (70,000+)    — not a fit for this role

score_pct = ndf["final_score"].rank(pct=True, method="average")

def assign_tier(pct):
    if pct >= 0.99: return 1
    if pct >= 0.95: return 2
    if pct >= 0.85: return 3
    if pct >= 0.70: return 4
    return 5

ndf["tier"] = score_pct.map(assign_tier)

tier_counts = ndf["tier"].value_counts().sort_index()
print("Tier Distribution")
print("-" * 40)
TIER_LABELS = {1:"Strong fit",2:"Good fit",3:"Moderate fit",4:"Adjacent",5:"Not a fit"}
for tier, cnt in tier_counts.items():
    bar = "█" * int(cnt / N * 50)
    print(f"  Tier {tier} ({TIER_LABELS[tier]:<14}): {cnt:>7,}  {bar}")
print()
print(f"Our top-100 submission = Tier 1 top 0.1% of pool")

# %% [markdown]
# **Design Note 8.1 — Tier Distribution Is Expected to Be Bottom-Heavy**
# The JD explicitly says: *"We're not expecting to find many matches in a 100K pool."*
# If Tier 1 contains 1,000 candidates and Tier 5 contains 70,000+, that is correct.
# A system that promotes 20% of candidates to Tier 1 is not reading the JD — it's
# keyword matching.
# 
# **Design Note 8.2 — Our Top-100 Submission Is Tier 1 Top 0.1%**
# The submitted candidates are the very top of Tier 1. The tier label is included
# in the final export as context for the reasoning generation.

# %% [markdown]
# ## 💬 Phase 9 — Explanation Engine (Rule-Based)

# %%
# Rule-based reasoning — no LLM, no templates, no hallucination
# Each reasoning string references actual profile data.
#
# Structure:
#   1. Role + experience + top retrieval skills (from actual profile)
#   2. Strongest evidence signal (evaluation or production)
#   3. Market validation signal (if strong)
#   4. Concern (if present — honest, required by submission spec)

RETRIEVAL_PRIORITY = [
    "FAISS","Embeddings","Elasticsearch","Information Retrieval",
    "Pinecone","Milvus","Vector Search","BM25",
    "Weaviate","Dense Retrieval","Hybrid Search","Sentence Transformers",
    "Learning to Rank","Recommendation Systems",
]
LLM_PRIORITY = ["LangChain","RAG","Prompt Engineering","Fine-tuning LLMs"]

def generate_reasoning(cid, feat_row, lookup):
    c = lookup.get(cid)
    if c is None:
        return "Profile data unavailable."

    profile   = c.get("profile", {})
    skills_set = {s["name"] for s in c.get("skills", [])}
    title      = profile.get("current_title", "")
    exp        = profile.get("years_of_experience", 0)
    city       = profile.get("location", "").split(",")[0].strip()

    parts    = []
    concerns = []

    # 1. Opening: experience + title + top skills (only skills actually in profile)
    ret_skills = [sk for sk in RETRIEVAL_PRIORITY if sk in skills_set][:3]
    if ret_skills:
        parts.append(f"{exp:.0f}yr {title.lower()} with {'/'.join(ret_skills)}")
    else:
        llm_skills = [sk for sk in LLM_PRIORITY if sk in skills_set][:2]
        if llm_skills:
            parts.append(f"{exp:.0f}yr {title.lower()} with {'/'.join(llm_skills)}")
        else:
            parts.append(f"{exp:.0f}yr {title.lower()}")

    # 2. Evidence signals (highest-value JD signals)
    if feat_row["evaluation_signal_score"] >= 0.4:
        parts.append("career history documents evaluation metric ownership (NDCG/MRR)")
    elif feat_row["evaluation_signal_score"] >= 0.2:
        parts.append("evaluation metrics mentioned in career history")

    if feat_row["production_signal_score"] >= 0.4:
        parts.append("evidence of production deployment at scale")
    elif feat_row["production_signal_score"] >= 0.2:
        parts.append("some production deployment evidence")

    # 3. Skill depth
    if int(feat_row["expert_ai_skills"]) >= 2:
        parts.append(f"{int(feat_row['expert_ai_skills'])} expert-level AI skills")

    # 4. Platform assessment (objective evidence)
    if feat_row["avg_ai_assessment_score"] >= 70:
        parts.append(f"platform-verified {feat_row['avg_ai_assessment_score']:.0f}/100")

    # 5. Market validation
    rr = feat_row["recruiter_response_rate"]
    if rr >= 0.80 and feat_row["saved_by_recruiters_norm"] >= 0.3:
        parts.append(f"strong recruiter engagement ({rr:.0%} response rate)")
    elif rr >= 0.65:
        parts.append(f"responsive to outreach ({rr:.0%})")

    # 6. Location (positive signal only)
    if city.lower() in {"pune","noida"}:
        parts.append(f"based in {city}")

    # Concerns (honest — required by submission spec Stage 4)
    if feat_row["consulting_ratio"] >= 0.8:
        concerns.append("primarily consulting background")
    if feat_row["days_since_active"] > 180:
        concerns.append(f"inactive {int(feat_row['days_since_active'])//30}mo on platform")
    if feat_row["notice_period"] > 90:
        concerns.append(f"{int(feat_row['notice_period'])}d notice period")
    if int(feat_row["is_honeypot"]) == 1:
        concerns.append("profile inconsistencies flagged")

    # Assemble
    reasoning = "; ".join(parts[:4])
    if concerns:
        reasoning += ". Concerns: " + ", ".join(concerns)

    return (reasoning + ".").strip()[:250]

# Test on first candidate
sample_id  = df["candidate_id"].iloc[0]
sample_row = ndf.set_index("candidate_id").loc[sample_id]
print("Reasoning engine test:")
print(" ", generate_reasoning(sample_id, sample_row, candidates_lookup))

# %% [markdown]
# **Design Note 9.1 — Anti-Hallucination by Design**
# The function builds skill mentions only from `skills_set = {s["name"] for s in profile["skills"]}`.
# It never invents skills, company names, or role descriptions not in the profile.
# Submission Stage 4 specifically penalises hallucinated reasoning.
# 
# **Design Note 9.2 — Concerns Are Mandatory When Present**
# The submission spec samples 10 random rows for Stage 4 manual review and explicitly rewards
# *"honest concerns."* Any candidate with `consulting_ratio >= 0.8`, `days_since_active > 180`,
# or `notice_period > 90` will have the concern surfaced in their reasoning string.
# 
# **Design Note 9.3 — Rule-Based Is Faster and More Reliable Than LLM Here**
# An LLM per candidate would violate the 5-minute CPU budget and risk hallucination.
# Rule-based generation is deterministic, fast (<5 seconds for 100 candidates), and
# honest because it only references features that came from the actual profile.

# %% [markdown]
# ## 🥇 Phase 10 — Top 100 Extraction

# %%
# Sort all 100k by final_score, take top 100
ranked = (
    ndf[["candidate_id","final_score","capability_score",
         "validation_score","tier","consulting_ratio","is_honeypot"]]
    .sort_values("final_score", ascending=False)
    .reset_index(drop=True)
)
ranked["rank"] = ranked.index + 1
ranked = ranked.rename(columns={"final_score": "score"})

top_100 = ranked.head(100).copy()

# Score must be non-increasing
score_diffs = top_100["score"].diff().dropna()
assert (score_diffs <= 1e-10).all(), "Scores are not non-increasing! Check sort."
print("Score non-increasing check : ✅")

# Generate reasoning for top 100 only
print("Generating reasoning for top 100...")
feat_idx = ndf.set_index("candidate_id")
top_100["reasoning"] = top_100["candidate_id"].apply(
    lambda cid: generate_reasoning(cid, feat_idx.loc[cid], candidates_lookup)
)

print(f"\nTop 100 extracted ✅")
print(f"  Rank 1   score: {top_100['score'].iloc[0]:.6f}")
print(f"  Rank 10  score: {top_100['score'].iloc[9]:.6f}")
print(f"  Rank 50  score: {top_100['score'].iloc[49]:.6f}")
print(f"  Rank 100 score: {top_100['score'].iloc[99]:.6f}")

# %%
# Honeypot check — must be < 10% in top 100 (submission spec requirement)
n_hp_top100 = int(top_100["is_honeypot"].sum())
print(f"Honeypots in top 100 : {n_hp_top100}  ({n_hp_top100:.0f}%)")
if n_hp_top100 > 10:
    print("⚠️  WARNING: >10% honeypots — submission will be DISQUALIFIED")
    print("   Increase honeypot penalty in Phase 6 (currently 0.05)")
else:
    print("Honeypot count safe ✅  (< 10%)")

print()
print("Top 10 preview:")
print(top_100[["rank","candidate_id","score","tier","reasoning"]].head(10).to_string(index=False))

# %% [markdown]
# **Design Note 10.1 — Reasoning Generated Only for Top 100**
# Running the reasoning engine for all 100k would take ~2 minutes.
# The submission requires reasoning only for the 100 submitted candidates.
# With O(1) lookup, generating 100 reasoning strings takes < 5 seconds.
# 
# **Design Note 10.2 — NDCG@10 = 50% of Final Score**
# Getting ranks 1–10 right is worth half the hackathon score.
# Always run spot checks on the top 10 specifically (Phase 11).

# %% [markdown]
# ## 🔎 Phase 11 — Sanity Validation

# %%
# Check 1: Title distribution in top 100
print("=== Title Distribution — Top 100 ===")
top100_profiles = [candidates_lookup.get(cid, {}).get("profile", {}) for cid in top_100["candidate_id"]]
title_counts = pd.Series([p.get("current_title","—") for p in top100_profiles]).value_counts()
print(title_counts.head(15).to_string())

# %%
# Check 2: JD skill presence in top 20 candidates
JD_SKILL_CHECK = [
    "FAISS","Embeddings","Information Retrieval","LangChain",
    "Elasticsearch","Pinecone","Vector Search","BM25",
    "Sentence Transformers","Learning to Rank",
]
print("=== JD Skill Coverage — Top 20 Candidates ===")
top20_ids = top_100["candidate_id"].head(20).tolist()
hits = Counter()
for cid in top20_ids:
    skills = {s["name"] for s in candidates_lookup.get(cid,{}).get("skills",[])}
    for sk in JD_SKILL_CHECK:
        if sk in skills:
            hits[sk] += 1

for sk, cnt in sorted(hits.items(), key=lambda x: -x[1]):
    bar = "█" * cnt
    print(f"  {sk:<30} {cnt:>2}/20  {bar}")

any_hit = sum(hits.values())
if any_hit == 0:
    print("⚠️  WARNING: No JD skills found in top 20 — formula may be off")
else:
    print("\nJD skills present in top-20 ✅")

# %%
# Audit: experience distribution after experience_fit is applied
print("=== Experience Distribution — Adjusted Top 100 ===")
top100_exp = top_100.merge(
    ndf[["candidate_id","experience_years","experience_fit"]],
    on="candidate_id", how="left"
)
print(top100_exp["experience_years"].describe().round(2))
buckets = pd.cut(top100_exp["experience_years"],
                 bins=[0,3,5,9,12,20],
                 labels=["<3yr","3-5yr","5-9yr (target)","9-12yr","12+yr"])
print("\nBucket distribution:")
print(buckets.value_counts().sort_index().to_string())

in_range = ((top100_exp["experience_years"] >= 5) & (top100_exp["experience_years"] <= 9)).sum()
print(f"\nIn JD range (5-9yr) : {in_range} / 100  ", end="")
print("✅" if in_range >= 80 else "⚠️  BELOW TARGET — increase experience_multiplier penalties")

# Visual distribution check
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(top100_exp["experience_years"], bins=15, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].axvspan(5, 9, alpha=0.15, color="green", label="JD target (5-9yr)")
axes[0].set_xlabel("Years of Experience")
axes[0].set_ylabel("# Candidates in Top 100")
axes[0].set_title("Experience Distribution — Final Top 100", fontweight="bold")
axes[0].legend(fontsize=8)

axes[1].hist(ndf["experience_fit"], bins=[0.55,0.65,0.70,0.80,0.85,0.90,0.95,1.05],
             color="#DD8452", edgecolor="white", alpha=0.85)
axes[1].set_xlabel("experience_fit multiplier value")
axes[1].set_ylabel("# Candidates (all 100k)")
axes[1].set_title("Experience Multiplier Distribution", fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/experience_audit.png", dpi=100, bbox_inches="tight")
plt.show()
print("experience_audit.png saved to outputs/")

# %%
# Check 3: Spot checks at rank 1, 10, 50, 100
print("=== Spot Checks ===")
for rank in [1, 10, 50, 100]:
    row    = top_100[top_100["rank"] == rank].iloc[0]
    cid    = row["candidate_id"]
    c      = candidates_lookup.get(cid, {})
    p      = c.get("profile", {})
    frow   = feat_idx.loc[cid]
    ai_sk  = [s["name"] for s in c.get("skills",[]) if s["name"] in set(JD_SKILL_CHECK)][:4]

    print(f"\nRank {rank:3d} | {cid} | Tier {int(row['tier'])}")
    print(f"  Title         : {p.get('current_title','—')}")
    print(f"  Experience    : {p.get('years_of_experience','—')} yr")
    print(f"  Location      : {p.get('location','—')}")
    print(f"  JD skills     : {ai_sk}")
    print(f"  eval_signal   : {frow['evaluation_signal_score']:.4f}")
    print(f"  prod_signal   : {frow['production_signal_score']:.4f}")
    print(f"  semantic_pct  : {frow['semantic_percentile']:.3f}")
    print(f"  capability    : {frow['capability_score']:.4f}")
    print(f"  validation    : {frow['validation_score']:.4f}")
    print(f"  risk_mult     : {frow['risk_multiplier']:.3f}")
    print(f"  avail_mult    : {frow['availability_multiplier']:.3f}")
    print(f"  Final score   : {row['score']:.6f}")
    print(f"  Reasoning     : {row['reasoning'][:120]}...")

# %%
# Check 4: V2 comparison — keyword-first formula to test robustness
V2_CAP_WEIGHTS = {
    "semantic_pct_capped"         : 0.30,
    "retrieval_score_pct"         : 0.25,
    "quality_score_log_pct"       : 0.15,
    "avg_ai_assessment_score_pct" : 0.10,
    "evaluation_signal_combo"     : 0.10,
    "production_signal_score_pct" : 0.05,
    "career_keyword_score_pct"    : 0.05,
}
ndf["cap_v2"] = sum(ndf[col] * w for col, w in V2_CAP_WEIGHTS.items())
ndf["base_v2"] = 0.60 * ndf["cap_v2"] + 0.25 * ndf["validation_score"] + 0.15 * ndf["availability_score_pct"]
ndf["score_v2"] = (ndf["base_v2"] * ndf["risk_multiplier"] * ndf["availability_multiplier"] * ndf["experience_fit"] * ndf["product_company_multiplier"])

v1_top10 = set(ndf.nlargest(10,  "final_score")["candidate_id"])
v2_top10 = set(ndf.nlargest(10,  "score_v2")["candidate_id"])
v1_top100 = set(ndf.nlargest(100, "final_score")["candidate_id"])
v2_top100 = set(ndf.nlargest(100, "score_v2")["candidate_id"])

print("=== V1 (Evidence-first) vs V2 (Keyword-first) Comparison ===")
print(f"  Top-10  overlap : {len(v1_top10  & v2_top10):2d}/10")
print(f"  Top-100 overlap : {len(v1_top100 & v2_top100):3d}/100")
print()
v1_exclusive = v1_top10 - v2_top10
v2_exclusive = v2_top10 - v1_top10
if v1_exclusive:
    print("In V1 top-10 but NOT V2 (evidence-only candidates):")
    for cid in v1_exclusive:
        frow = feat_idx.loc[cid]
        title = candidates_lookup.get(cid,{}).get("profile",{}).get("current_title","—")
        print(f"  {cid} | {title[:30]} | eval={frow['evaluation_signal_score']:.3f} prod={frow['production_signal_score']:.3f}")
if v2_exclusive:
    print("In V2 top-10 but NOT V1 (keyword-only candidates):")
    for cid in v2_exclusive:
        frow = feat_idx.loc[cid]
        title = candidates_lookup.get(cid,{}).get("profile",{}).get("current_title","—")
        print(f"  {cid} | {title[:30]} | eval={frow['evaluation_signal_score']:.3f} prod={frow['production_signal_score']:.3f}")

# %% [markdown]
# **Design Note 11.1 — What Good Spot Checks Look Like**
# Rank 1 should be a senior AI/ML engineer with hands-on retrieval experience,
# evaluation metric evidence in career descriptions, India-based, actively engaged.
# Rank 100 should have some retrieval skills and be defensible — not random.
# If rank 1 is a Marketing Manager with keyword-stuffed skills, the formula needs tuning.
# 
# **Design Note 11.2 — V1 vs V2 Overlap Interpretation**
# - ≥85% top-100 overlap → formula is robust, weights barely matter, stick with V1
# - 70–85% overlap → investigate V1-exclusive candidates: do they have genuine evidence signals?
# - <70% overlap → formula is sensitive; review spot checks before deciding which to submit
# 
# **Design Note 11.3 — V1-Exclusive Candidates Are the Key Test**
# Candidates in V1 top-10 but not V2 are those whose `evaluation_signal_score` or
# `production_signal_score` elevated them above pure-skill candidates.
# Inspect their career descriptions: do they show real evaluation or deployment work?
# If yes, V1 is correct. If their scores seem inflated by noise, reduce evidence weights.

# %% [markdown]
# ## 📤 Phase 12 — Submission Export & Validation

# %%
# Build submission in exact spec format: candidate_id, rank, score, reasoning
submission = top_100[["candidate_id","rank","score","reasoning"]].copy()

# ── Submission spec validation ─────────────────────────────────────────────────
errors = []

if len(submission) != 100:
    errors.append(f"Row count: {len(submission)} (expected 100)")

if sorted(submission["rank"].tolist()) != list(range(1, 101)):
    errors.append("Ranks are not exactly 1–100")

if submission["candidate_id"].nunique() != 100:
    errors.append(f"Duplicate candidate_ids: {100 - submission['candidate_id'].nunique()}")

if not (submission["score"].diff().dropna() <= 1e-10).all():
    errors.append("Scores not non-increasing with rank")

empty_r = (submission["reasoning"].isna() | (submission["reasoning"].str.strip() == "")).sum()
if empty_r > 0:
    errors.append(f"{empty_r} empty reasoning strings")

valid_ids = set(candidates_lookup.keys())
bad_ids   = [cid for cid in submission["candidate_id"] if cid not in valid_ids]
if bad_ids:
    errors.append(f"{len(bad_ids)} invalid candidate_ids")

if errors:
    print("❌ Validation FAILED:")
    for e in errors:
        print(f"   - {e}")
else:
    print("Submission validation PASSED ✅")
    print(f"  Rows     : {len(submission)}")
    print(f"  Ranks    : 1 – {submission['rank'].max()}")
    print(f"  Score    : {submission['score'].min():.6f} → {submission['score'].max():.6f}")
    print(f"  Unique IDs: {submission['candidate_id'].nunique()}")

# %%
# Export files
submission.to_csv("outputs/submission.csv", index=False)

# Full ranking (all 100k — useful for README and debugging)
full_ranked = ranked[["candidate_id","rank","score","tier"]].copy()
full_ranked.to_csv("outputs/final_ranked_all.csv", index=False)

# Top 100 with full feature context (for analysis)
top_100_full = top_100.merge(
    ndf[["candidate_id","capability_score","validation_score",
         "availability_multiplier","risk_multiplier","tier"]],
    on="candidate_id", how="left"
)
top_100_full.to_csv("outputs/top100_candidates.csv", index=False)

print("Saved:")
print("  outputs/submission.csv        ← rename to {team_id}.csv for upload")
print("  outputs/final_ranked_all.csv  ← full 100k ranking")
print("  outputs/top100_candidates.csv ← top 100 with feature context")
print()
print("Final Top 10:")
print(submission.head(10).to_string(index=False))

# %% [markdown]
# ---
# ## ✅ Notebook 03 Complete
# 
# ```
# features_df.pkl (100k × ~50 features)
#        ↓
# Phase 0   Load + integrity check
# Phase 1   Feature audit + sparsity check
# Phase 2   Normalize via rank(pct=True)
# Phase 3   Capability engine  (evidence-first: eval + production + retrieval)
# Phase 4   Validation engine  (saved_by_recruiters leads)
# Phase 5   Availability multiplier  (0.30–1.15)
# Phase 6   Risk multiplier  (consulting penalty + honeypot near-zero)
# Phase 7   Recruiter score  (0.60×cap + 0.25×val + 0.15×avail) × risk × avail_mult
# Phase 8   Tier assignment  (T1=top 1% → T5=rest)
# Phase 9   Rule-based explanation engine  (no LLM, anti-hallucination)
# Phase 10  Top 100 extraction + non-increasing score verification
# Phase 11  Sanity checks: titles, JD skills, spot checks, V1/V2 comparison
# Phase 12  Validation + export
#        ↓
# outputs/submission.csv  ← rename to {team_id}.csv and run validate_submission.py
# ```
# 
# **What remains before final submission:**
# 1. Run pipeline on full 100k dataset — verify spot checks pass
# 2. Review V1 vs V2 overlap — if top-10 diverges, inspect evidence-signal candidates
# 3. README.md — one `python rank.py` command, architecture diagram
# 4. Sandbox link (HuggingFace Spaces or Streamlit on small sample)
# 5. `submission_metadata.yaml` — team name, repo, AI tools declaration
# 6. Rename `submission.csv` → `{team_id}.csv` and run `validate_submission.py`


