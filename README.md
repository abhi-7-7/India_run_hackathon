# REDRO AI — Intelligent Candidate Ranking

> **Hackathon:** REDRO AI — Intelligent Candidate Discovery & Ranking  
> **Team:** Aarsh Bhatnagar  
> **Approach:** Evidence-first recruiter decision engine  
> **GitHub:** https://github.com/abhi-7-7/India_run_hackathon

---

## Quick Start

```bash
git clone https://github.com/abhi-7-7/India_run_hackathon
cd India_run_hackathon
pip install -r requirements.txt

# Place dataset at:
#   raw_dataset/candidates.jsonl

python rank.py
# → REDRO_AI.csv  (100 ranked candidates, ready to submit)
```

---

## Repository Structure

```
India_run_hackathon/
│
├── Notebook/
│   ├── 01_eda.ipynb           EDA — 16 phases, 40+ observations
│   ├── 02_fea_eng.ipynb       Feature engineering — 55 features
│   ├── 03_rank.ipynb          Ranking engine — 4-engine formula
│   ├── 04_final.ipynb         Evaluation, ablation, business insights
│   │
│   └── outputs/               ← created when notebooks are run
│       ├── features_df.pkl    55-feature table (100k × 55)
│       ├── semantic_similarity.npy  cached embeddings
│       ├── submission.csv     submitted top-100
│       ├── final_ranked_all.csv     full 100k ranking
│       └── top100_candidates.csv    top 100 with feature context
│
├── raw_dataset/
│   └── candidates.jsonl       100,000 candidate profiles (not in repo)
│
├── rank.py                    End-to-end script → REDRO_AI.csv
├── app.py                     Streamlit demo
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

---

## Data Flow

```
candidates.jsonl  (raw input)
       │
       ▼
Notebook/02_fea_eng.ipynb
  builds 55 features per candidate
  saves → Notebook/outputs/features_df.pkl
  saves → Notebook/outputs/semantic_similarity.npy
       │
       ▼
Notebook/03_rank.ipynb
  loads features_df.pkl
  applies 4-engine scoring formula
  saves → Notebook/outputs/submission.csv
  saves → Notebook/outputs/final_ranked_all.csv
       │
       ▼
Notebook/04_final.ipynb
  loads all outputs
  runs ablation, validation, business insights
       │
       ▼
rank.py  (standalone production script)
  reproduces full pipeline from raw data
  loads embedding cache if present (skips 8-min step)
  exports → REDRO_AI.csv  ← submit this file
```

**Note on rank.py:** `rank.py` is a self-contained reproduction script.
It rebuilds all features from `candidates.jsonl` so the submission can be
reproduced on any machine without running the notebooks first.
It automatically uses the embedding cache at `Notebook/outputs/semantic_similarity.npy`
if it exists, cutting runtime from ~10 minutes to ~1 minute.

---

## Reproducing Results

### Option A — Run notebooks in order (recommended for full analysis)

```bash
cd India_run_hackathon/Notebook
jupyter notebook

# Run in order:
# 1. 01_eda.ipynb
# 2. 02_fea_eng.ipynb     ← produces features_df.pkl
# 3. 03_rank.ipynb        ← produces submission.csv
# 4. 04_final.ipynb       ← validation and analysis
```

Outputs appear in `Notebook/outputs/`.

### Option B — Single command reproduction

```bash
cd India_run_hackathon
python rank.py
```

**Runtime:**
- First run with no cache: ~10 minutes (sentence-transformer embeddings)
- Subsequent runs or if `Notebook/outputs/semantic_similarity.npy` exists: ~1 minute

Output: `REDRO_AI.csv`

### Validate the submission

```bash
python validate_submission.py REDRO_AI.csv
```

Expected: `PASS`

---

## Streamlit Demo

```bash
cd India_run_hackathon
streamlit run app.py
```

### Two modes:

**Default JD (Submission)**
Shows the exact submitted top-100 candidates loaded from
`Notebook/outputs/submission.csv`. Results are identical to `REDRO_AI.csv`.
Demo = Submission. No divergence.

**Custom JD (Explore)**
User pastes any job description. The app re-ranks a **stratified sample
of 3,000 candidates** (top by evaluation evidence + top by retrieval skills
+ random) using the same `all-MiniLM-L6-v2` model and identical NB03 formula.
Note: custom JD mode ranks the 3,000-candidate sample for responsiveness,
not the full 100k pool. Use `rank.py` with a modified `JD_TEXT` for full-pool ranking.

---

## Scoring Formula

```
capability_score = (
    0.25 × semantic_percentile_capped    # JD holistic alignment
  + 0.15 × evaluation_signal_combo       # NDCG/MRR/MAP in career descriptions
  + 0.15 × production_signal_score       # deployed/shipped evidence
  + 0.18 × retrieval_score               # core retrieval tool skills
  + 0.11 × quality_score_log             # skill depth (proficiency × duration)
  + 0.07 × career_keyword_score          # domain vocabulary density
  + 0.09 × avg_ai_assessment_score       # platform-verified skill scores
)

validation_score = (
    0.40 × saved_by_recruiters            # independent market validation
  + 0.30 × recruiter_response_rate        # engagement signal
  + 0.20 × interview_completion_rate      # reliability
  + 0.10 × profile_views                  # passive interest
)

base_score   = 0.60 × capability + 0.25 × validation + 0.15 × availability

final_score  = base_score
             × risk_multiplier            # consulting penalty + honeypot gate
             × availability_multiplier    # recency gate (0.30–1.10)
             × experience_fit             # JD band: 1.0 at 5–9yr, min 0.60
             × product_company_bonus      # +10% for known product companies
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Evidence > Keywords | NDCG/MRR in career text > listing FAISS as a skill |
| `rank(pct=True)` normalization | Handles 1.8% eval signal sparsity better than MinMaxScaler |
| Availability as multiplier | Stale candidate (11 months inactive) → score × 0.30 |
| Experience fit gate | Graded: 1.0 at 5–9yr, 0.60 below 3yr or above 12yr |
| Product company bonus | +10% for Google/Amazon/Uber/Swiggy/Flipkart etc. |
| Honeypot near-zero (×0.05) | Keeps sort order; all below real candidates |

---

## Ablation Study (from Notebook 04)

| Component Removed | Top-100 Overlap | Importance |
|---|---|---|
| Experience Fit | 69% | **Critical** |
| Evaluation Signal | 83% | High |
| Validation (Behavioral) | 89% | High |
| Production Signal | 91% | Moderate |
| Risk Filter | 94% | Gate (correct) |

---

## Submission Validation

| Check | Result |
|---|---|
| 100 rows | ✅ |
| Unique candidate IDs | ✅ |
| Ranks 1–100 | ✅ |
| Scores non-increasing | ✅ |
| No empty reasoning | ✅ |
| Honeypot rate < 10% | ✅ |
| In JD experience range (5–9yr) | 95/100 |
| V1 vs V2 formula overlap | 87/100 (stable) |