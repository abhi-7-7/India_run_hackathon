
"""
REDRO AI Hackathon — End-to-End Candidate Ranking Pipeline
==========================================================
Usage:  python rank.py
Output: REDRO_AI.csv  (100 candidates, ranked 1-100)

Runtime:
  First run  (~10 min) — computes sentence-transformer embeddings, caches to outputs/
  After that (~1 min)  — loads cache, re-runs scoring only

No external API calls. Fully reproducible on CPU.
"""

import sys, os, json, math, warnings
from datetime import date
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.4f}".format)

ROOT       = os.path.dirname(os.path.abspath(__file__))
NB_OUTPUTS = os.path.join(ROOT, "Notebook", "outputs")
os.makedirs(NB_OUTPUTS, exist_ok=True) 

DATASET_PATH          = os.path.join(ROOT, "raw_dataset", "candidates.jsonl")
CACHE_EMBEDDINGS      = os.path.join(NB_OUTPUTS, "semantic_similarity.npy")
CACHE_EMBEDDINGS_RAW  = os.path.join(NB_OUTPUTS, "candidate_embeddings.npy")
FAISS_INDEX_PATH      = os.path.join(NB_OUTPUTS, "faiss_index.bin")
FAISS_IDS_PATH        = os.path.join(NB_OUTPUTS, "faiss_ids.npy")
SCORING_THRESHOLDS    = os.path.join(NB_OUTPUTS, "scoring_thresholds.pkl")
OUTPUT_CSV            = os.path.join(ROOT, "REDRO_AI.csv")

REFERENCE_DATE   = date(2026, 6, 5)

RETRIEVAL_SKILLS = {
    "Embeddings","FAISS","Milvus","Elasticsearch","BM25",
    "Information Retrieval","Vector Search","Pinecone",
    "Weaviate","Qdrant","OpenSearch","Recommendation Systems",
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
AI_ALL_SKILLS = RETRIEVAL_SKILLS | LLM_SKILLS | ML_SKILLS

PROF_WEIGHT = {"beginner":0.5,"intermediate":1.0,"advanced":2.0,"expert":3.0}

AI_CERTS = {
    "Deep Learning Specialization","NLP Specialization",
    "AWS Certified Machine Learning Specialty",
    "Google Cloud Professional ML Engineer",
    "LangChain for LLM Application Development",
    "TensorFlow Developer Certificate",
    "MLOps Professional Certificate",
    "Natural Language Processing Specialization",
}

CONSULTING_FIRMS = {
    "tcs","infosys","wipro","accenture","cognizant",
    "capgemini","hcl","tech mahindra","mphasis","ltimindtree",
}
PRODUCT_COMPANIES = {
    "google","amazon","uber","swiggy","zomato","flipkart",
    "microsoft","netflix","meta","apple","linkedin","twitter",
    "spotify","airbnb","stripe","doordash","lyft","salesforce",
    "adobe","atlassian","shopify","square","paypal",
}
PREFERRED_CITIES  = {"pune","noida"}
ACCEPTABLE_CITIES = {
    "hyderabad","mumbai","bangalore","bengaluru",
    "delhi","new delhi","gurgaon","gurugram","navi mumbai",
}

TECH_TITLE_KEYWORDS = {
    "engineer","scientist","developer","researcher","programmer",
    "sde","data scientist","machine learning","applied scientist",
    "nlp","data analyst","analytics engineer","ml engineer",
}
NON_TECH_TITLE_KEYWORDS = {
    "marketing","sales","human resources"," hr ","recruiter","recruitment",
    "operations","finance","accountant","accounting","graphic designer",
    "content writer","business development","customer success",
    "customer support","legal","administrator","executive assistant",
}
NON_CODING_TITLE_KEYWORDS = {
    "director","vp","vice president","head of","chief","cto",
    "engineering manager","architect",
}

SENIORITY_LEVELS = [
    (4, ["director","vp","vice president","chief"]),
    (3, ["principal","head of"]),
    (2, ["staff","lead "]),
    (1, ["senior","sr."]),
    (0, []),  
]
def _seniority_level(title):
    t = title.lower()
    for level, kws in SENIORITY_LEVELS:
        if any(kw in t for kw in kws):
            return level
    return 0

JD_TEXT = """
Senior AI Engineer — Retrieval, Search, Ranking and Recommendation Systems
Production experience with embeddings-based retrieval systems.
Vector databases: FAISS, Pinecone, Milvus, Weaviate, Qdrant, Elasticsearch.
Hybrid search, dense retrieval, BM25, semantic search, information retrieval.
Evaluation: NDCG, MRR, MAP, offline evaluation, A/B testing, online evaluation.
Learning to rank, search relevance, retrieval quality, ranking systems.
Recommendation systems, candidate matching, similarity search.
LLM: RAG, retrieval augmented generation, LangChain, Prompt Engineering.
Fine-tuning: LoRA, QLoRA, PEFT, model adaptation.
Sentence transformers, embedding models, embedding drift, index refresh.
NLP, natural language processing, text ranking, search quality.
Python, production ML, MLOps, MLflow, feature engineering.
Machine learning, deep learning, PyTorch, TensorFlow, scikit-learn.
5-9 years at product companies. Pune, Noida, Hyderabad, Mumbai, Delhi NCR.
"""

PRIORITY_SKILLS = [
    "FAISS","Embeddings","Elasticsearch","Information Retrieval",
    "Pinecone","Milvus","Vector Search","BM25","Sentence Transformers",
    "LangChain","RAG","Learning to Rank","Recommendation Systems",
    "Machine Learning","PyTorch","TensorFlow",
]


def load_candidates(path):
    print(f"Loading candidates from {path}...")
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"  Loaded: {len(candidates):,} candidates")
    return candidates

def build_candidate_text(c):
    parts = []
    parts.append(c["profile"].get("current_title", ""))
    parts.append(c["profile"].get("headline", ""))
    parts.append(c["profile"].get("summary", ""))
    parts.append(" ".join(s["name"] for s in c.get("skills", [])))
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(p for p in parts if p).strip()


def get_semantic_similarities(candidates, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        similarities = np.load(cache_path, allow_pickle=False)
        print(f"  Loaded: {len(similarities):,} similarity scores")
        return similarities

    print("Computing embeddings from scratch (first run ~8-10 min on CPU)...")
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("  Building candidate texts...")
    texts = [build_candidate_text(c) for c in candidates]

    print("  Encoding JD...")
    jd_emb = model.encode([JD_TEXT], show_progress_bar=False, convert_to_numpy=True)

    print(f"  Encoding {len(texts):,} candidates (this takes ~8-10 min)...")
    cand_embs = model.encode(
        texts, batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("  Computing cosine similarities...")
    sims = cos_sim(jd_emb, cand_embs)[0]

    np.save(cache_path, sims)
    np.save(CACHE_EMBEDDINGS_RAW, cand_embs)  
    print(f"  Cached to {cache_path}")
    return sims


def build_faiss_index(candidates):
    """
    Builds a FAISS IndexFlatIP over all-MiniLM-L6-v2 candidate embeddings.
    Requires CACHE_EMBEDDINGS_RAW (written when running rank.py from scratch).
    If raw embeddings aren't present (legacy cache), prints a one-time note
    and skips gracefully — all other pipeline steps still run normally.
    """
    if not os.path.exists(CACHE_EMBEDDINGS_RAW):
        print("  FAISS: raw embeddings not found — delete semantic_similarity.npy "
              "and re-run rank.py once to build the index.")
        return False
    try:
        import faiss
    except ImportError:
        print("  FAISS: faiss-cpu not installed (pip install faiss-cpu) — skipping.")
        return False

    print("Building FAISS index...")
    cand_embs = np.load(CACHE_EMBEDDINGS_RAW).astype("float32")
    faiss.normalize_L2(cand_embs)                     

    index = faiss.IndexFlatIP(cand_embs.shape[1])     
    index.add(cand_embs)
    faiss.write_index(index, FAISS_INDEX_PATH)

    cand_ids = np.array([c["candidate_id"] for c in candidates])
    np.save(FAISS_IDS_PATH, cand_ids)

    print(f"  FAISS: {index.ntotal:,} vectors, dim={cand_embs.shape[1]} → {FAISS_INDEX_PATH}")
    return True


def save_scoring_thresholds(features_df):
    """
    Saves per-signal percentile breakpoints (1001 points per signal) so
    app.py can estimate where a new candidate's raw feature values land
    in the 100k distribution without re-running the full pipeline.
    ~100KB total — safe to commit to the repo.
    """
    import pickle
    KEY_COLS = [
        "retrieval_score", "evaluation_signal_score", "production_signal_score",
        "quality_score_log", "avg_ai_assessment_score", "career_keyword_score",
        "availability_score", "recruiter_response_rate", "interview_completion_rate",
        "saved_by_recruiters_raw", "profile_views_raw", "semantic_similarity",
    ]
    pct_points = np.linspace(0, 100, 1001)
    thresholds = {}
    for col in KEY_COLS:
        if col in features_df.columns:
            thresholds[col] = np.percentile(
                features_df[col].fillna(0).values.astype(float), pct_points
            )
    with open(SCORING_THRESHOLDS, "wb") as f:
        pickle.dump(thresholds, f)
    print(f"  Scoring thresholds saved ({len(thresholds)} signals → {SCORING_THRESHOLDS})")


def engineer_features(candidates, similarities):
    print("Engineering features...")
    rows = []

    texts = [build_candidate_text(c).lower() for c in candidates]

    for idx, c in enumerate(candidates):
        cid     = c["candidate_id"]
        profile = c["profile"]
        sig     = c["redrob_signals"]
        skills  = c.get("skills", [])
        jobs    = c.get("career_history", [])
        certs   = c.get("certifications", [])

        skill_names = {s["name"] for s in skills}
        text        = texts[idx]

        ret = len(skill_names & RETRIEVAL_SKILLS)
        llm = len(skill_names & LLM_SKILLS)
        ml  = len(skill_names & ML_SKILLS)
        ai_total = ret * 3 + llm * 2 + ml

        hidden_count = sum(1 for sig_kw in HIDDEN_SIGNALS if sig_kw.lower() in text)

        eval_kws  = ["ndcg","mrr","map","a/b test","offline eval","online eval","ranking quality","retrieval quality"]
        eval_score = min(1.0, sum(1 for kw in eval_kws if kw in text) / 3)

        prod_kws   = ["deployed","production","shipped","live traffic","real users","millions","at scale","launched"]
        prod_score = min(1.0, sum(1 for kw in prod_kws if kw in text) / 3)

        career_kws = ["retrieval","ranking","search","recommendation","information retrieval",
                      "embedding","vector","similarity","relevance","re-ranking"]
        career_score = min(1.0, sum(1 for kw in career_kws if kw in text) / 5)

        ai_skills_list = [s for s in skills if s["name"] in AI_ALL_SKILLS]
        if ai_skills_list:
            quality_raw = sum(
                PROF_WEIGHT.get(s.get("proficiency","beginner"), 0.5) *
                math.log1p(s.get("duration_months", 0)) / math.log1p(12)
                for s in ai_skills_list
            )
            quality_log     = math.log1p(quality_raw)
            avg_ai_duration = sum(s.get("duration_months",0) for s in ai_skills_list) / len(ai_skills_list)
            adv_ai          = sum(1 for s in ai_skills_list if s.get("proficiency") in ("advanced","expert"))
            exp_ai          = sum(1 for s in ai_skills_list if s.get("proficiency") == "expert")
            max_endorse     = max((s.get("endorsements",0) for s in ai_skills_list), default=0)
        else:
            quality_log = avg_ai_duration = adv_ai = exp_ai = max_endorse = 0.0

        assess = sig.get("skill_assessment_scores", {})
        all_scores = list(assess.values())
        ai_scores  = [v for k, v in assess.items() if k in AI_ALL_SKILLS]
        has_assess    = int(len(all_scores) > 0)
        has_ai_assess = int(len(ai_scores) > 0)
        avg_assess    = float(np.mean(all_scores)) if all_scores else 0.0
        avg_ai_assess = float(np.mean(ai_scores))  if ai_scores  else 0.0

        rr   = sig["recruiter_response_rate"]
        ic   = sig["interview_completion_rate"]
        saved_raw = sig["saved_by_recruiters_30d"]
        views_raw = sig["profile_views_received_30d"]
        appear_raw= sig["search_appearance_30d"]

        last_raw = sig.get("last_active_date")
        if not last_raw:
            days_inactive = 365
        else:
            days_inactive = (REFERENCE_DATE - date.fromisoformat(last_raw)).days
        recency_score = max(0.2, 1.0 - days_inactive / 365.0)

        notice = sig["notice_period_days"]
        if notice <= 30:    notice_score = 1.0
        elif notice <= 60:  notice_score = 0.7
        elif notice <= 90:  notice_score = 0.4
        else:               notice_score = 0.2

        wm = sig.get("preferred_work_mode", "flexible")
        wm_score = {"hybrid":1.0,"flexible":0.9,"onsite":0.7,"remote":0.6}.get(wm, 0.7)

        openness = (
            0.4 * int(sig["open_to_work_flag"]) +
            0.2 * int(sig["willing_to_relocate"]) +
            0.2 * int(sig["verified_email"]) +
            0.1 * int(sig["verified_phone"]) +
            0.1 * int(sig["linkedin_connected"])
        )

        city = profile.get("location","").split(",")[0].strip().lower()
        country = profile.get("country","")
        if country != "India":    loc_score = 0.3
        elif city in PREFERRED_CITIES:  loc_score = 1.0
        elif city in ACCEPTABLE_CITIES: loc_score = 0.8
        else:                     loc_score = 0.6

        avail_score = (0.35 * recency_score + 0.25 * openness +
                       0.20 * loc_score + 0.10 * notice_score + 0.10 * wm_score)

        raw_gh    = sig["github_activity_score"]
        has_gh    = int(raw_gh != -1)
        raw_offer = sig["offer_acceptance_rate"]
        has_offer = int(raw_offer != -1)

        exp_years = profile["years_of_experience"]
        job_count = len(jobs)

        total_mo    = sum(j.get("duration_months",0) for j in jobs)
        consult_mo  = sum(j.get("duration_months",0) for j in jobs
                          if any(cf in j.get("company","").lower() for cf in CONSULTING_FIRMS))
        consulting_ratio = consult_mo / total_mo if total_mo > 0 else 0.0

        product_mo = sum(j.get("duration_months",0) for j in jobs
                         if any(pc in j.get("company","").lower() for pc in PRODUCT_COMPANIES))
        product_ratio = product_mo / total_mo if total_mo > 0 else 0.0

        all_titles = (profile.get("current_title","") + " " +
                      " ".join(j.get("title","") for j in jobs)).lower()
        has_tech_title    = any(kw in all_titles for kw in TECH_TITLE_KEYWORDS)
        has_nontech_title = any(kw in all_titles for kw in NON_TECH_TITLE_KEYWORDS)
        if has_tech_title:
            title_credibility = 1.0
        elif has_nontech_title:
            title_credibility = 0.3
        else:
            title_credibility = 0.7

        llm_duration    = sum(s.get("duration_months",0) for s in skills if s["name"] in LLM_SKILLS)
        pre_llm_duration = sum(s.get("duration_months",0) for s in skills
                               if s["name"] in (ML_SKILLS | RETRIEVAL_SKILLS))
        recent_llm_only = int(llm_duration > 0 and llm_duration <= 12 and pre_llm_duration < 24)

        cur_title_lower = profile.get("current_title","").lower()
        non_coding_role = int(
            exp_years >= 5 and
            any(kw in cur_title_lower for kw in NON_CODING_TITLE_KEYWORDS)
        )
        avg_tenure_months = total_mo / job_count if job_count > 0 else 999
        
        chrono_levels = [_seniority_level(j.get("title","")) for j in reversed(jobs)]
        climbed = (len(chrono_levels) >= 3 and
                   all(b >= a for a, b in zip(chrono_levels, chrono_levels[1:])) and
                   (max(chrono_levels) - min(chrono_levels)) >= 2)
        job_hopper = int(climbed and job_count >= 3 and avg_tenure_months < 18)

        hp_flags = 0
        for s in skills:
            if s.get("proficiency") == "expert" and s.get("duration_months",1) == 0:
                hp_flags += 2
        short_experts = [s for s in skills if s.get("proficiency")=="expert" and s.get("duration_months",0) < 6]
        if len(short_experts) >= 3:
            hp_flags += 1
        if total_mo > exp_years * 12 * 1.5 and total_mo > 36:
            hp_flags += 1
        is_honeypot = int(hp_flags >= 2)

        cert_names  = {cert["name"] for cert in certs}
        has_ai_cert = int(bool(cert_names & AI_CERTS))
        ai_cert_cnt = len(cert_names & AI_CERTS)

        rows.append({
            "candidate_id"          : cid,
            "retrieval_score"       : ret * 3,
            "llm_score"             : llm * 2,
            "ml_score"              : ml,
            "ai_skill_total"        : ai_total,
            "hidden_signal_count"   : hidden_count,
            "evaluation_signal_score": eval_score,
            "production_signal_score": prod_score,
            "career_keyword_score"  : career_score,
            "quality_score_log"     : quality_log,
            "avg_ai_duration"       : avg_ai_duration,
            "advanced_ai_skills"    : adv_ai,
            "expert_ai_skills"      : exp_ai,
            "max_endorsements_ai"   : max_endorse,
            "has_assessment"        : has_assess,
            "has_ai_assessment"     : has_ai_assess,
            "avg_assessment_score"  : avg_assess,
            "avg_ai_assessment_score": avg_ai_assess,
            "recruiter_response_rate"  : rr,
            "interview_completion_rate": ic,
            "saved_by_recruiters_raw"  : saved_raw,
            "profile_views_raw"        : views_raw,
            "search_appearance_raw"    : appear_raw,
            "days_since_active"    : days_inactive,
            "recency_score"        : recency_score,
            "notice_period"        : notice,
            "notice_score"         : notice_score,
            "work_mode_score"      : wm_score,
            "location_score"       : loc_score,
            "openness_score"       : openness,
            "availability_score"   : avail_score,
            "has_github"           : has_gh,
            "github_activity_raw"  : raw_gh,
            "has_offer_history"    : has_offer,
            "experience_years"     : exp_years,
            "job_count"            : job_count,
            "consulting_ratio"     : consulting_ratio,
            "product_ratio"        : product_ratio,
            "is_honeypot"          : is_honeypot,
            "title_credibility"    : title_credibility,
            "recent_llm_only"      : recent_llm_only,
            "non_coding_role"      : non_coding_role,
            "job_hopper"           : job_hopper,
            "has_ai_cert"          : has_ai_cert,
            "ai_cert_count"        : ai_cert_cnt,
            "semantic_similarity"  : float(similarities[idx]),
        })

    df = pd.DataFrame(rows)
    print(f"  Features engineered: {df.shape}")
    return df


def rank_candidates(df, stages=None):
    """
    stages: set of active fix names, e.g. {"title_cred","prod_gate","recent_llm",
            "non_coding","job_hop","github"}. None or empty set = original V1 behavior.
    """
    stages = stages or set()
    print(f"Scoring candidates... [stages active: {sorted(stages) if stages else 'none (V1 baseline)'}]")
    ndf = df.copy()
    N   = len(ndf)

    if "title_cred" in stages:
        ndf["retrieval_score_eff"]     = ndf["retrieval_score"]     * ndf["title_credibility"]
        ndf["career_keyword_score_eff"] = ndf["career_keyword_score"] * ndf["title_credibility"]
    else:
        ndf["retrieval_score_eff"]      = ndf["retrieval_score"]
        ndf["career_keyword_score_eff"] = ndf["career_keyword_score"]

    RANK_COLS = [
        "retrieval_score_eff","llm_score","ml_score","ai_skill_total",
        "quality_score_log","avg_ai_duration","advanced_ai_skills",
        "expert_ai_skills","max_endorsements_ai","avg_assessment_score",
        "avg_ai_assessment_score","ai_cert_count","evaluation_signal_score",
        "production_signal_score","career_keyword_score_eff","hidden_signal_count",
        "saved_by_recruiters_raw","profile_views_raw","search_appearance_raw",
    ]
    for col in RANK_COLS:
        ndf[f"{col}_pct"] = ndf[col].fillna(0).rank(pct=True, method="average")

    ndf["eval_combo"]   = (0.6 * ndf["evaluation_signal_score_pct"] +
                           0.4 * (ndf["evaluation_signal_score"] > 0).astype(float))
    ndf["sem_capped"]   = ndf["semantic_similarity"].rank(pct=True).clip(upper=0.97)
    ndf["avail_pct"]    = ndf["availability_score"].rank(pct=True)
    ndf["saved_pct"]    = ndf["saved_by_recruiters_raw_pct"]
    ndf["views_pct"]    = ndf["profile_views_raw_pct"]

    CAP = {
        "sem_capped"                     : 0.25,
        "eval_combo"                     : 0.15,
        "production_signal_score_pct"    : 0.15,
        "retrieval_score_eff_pct"        : 0.18,
        "quality_score_log_pct"          : 0.11,
        "career_keyword_score_eff_pct"   : 0.07,
        "avg_ai_assessment_score_pct"    : 0.09,
    }
    ndf["capability_score"] = sum(ndf[col] * w for col, w in CAP.items())

    ndf["validation_score"] = (
        0.40 * ndf["saved_pct"] +
        0.30 * ndf["recruiter_response_rate"] +
        0.20 * ndf["interview_completion_rate"] +
        0.10 * ndf["views_pct"]
    )

    ndf["base_score"] = (
        0.60 * ndf["capability_score"] +
        0.25 * ndf["validation_score"] +
        0.15 * ndf["avail_pct"]
    )

    def exp_mult(e):
        if 5 <= e <= 9:   return 1.00
        elif 4 <= e < 5:  return 0.90
        elif 9 < e <= 12: return 0.85
        elif 3 <= e < 4:  return 0.75
        else:             return 0.60

    ndf["experience_fit"]          = ndf["experience_years"].apply(exp_mult)
    ndf["availability_multiplier"] = ndf["availability_score"].clip(lower=0.30, upper=1.10)
    ndf["product_multiplier"]      = 1.0 + 0.10 * ndf["product_ratio"]
    ndf["risk_multiplier"]         = ((1 - 0.80 * ndf["consulting_ratio"]) *
                                      ndf["is_honeypot"].map({0:1.0, 1:0.05}))

    if "prod_gate" in stages:
        ndf["production_gate_mult"] = np.where(ndf["production_signal_score"] == 0, 0.5, 1.0)
    else:
        ndf["production_gate_mult"] = 1.0

    if "recent_llm" in stages:
        ndf["recent_llm_mult"] = ndf["recent_llm_only"].map({0:1.0, 1:0.6})
    else:
        ndf["recent_llm_mult"] = 1.0

    if "non_coding" in stages:
        ndf["non_coding_mult"] = ndf["non_coding_role"].map({0:1.0, 1:0.6})
    else:
        ndf["non_coding_mult"] = 1.0

    if "job_hop" in stages:
        ndf["job_hop_mult"] = ndf["job_hopper"].map({0:1.0, 1:0.8})
    else:
        ndf["job_hop_mult"] = 1.0
    if "github" in stages:
        ndf["github_mult"] = np.where(
            ndf["github_activity_raw"] >= 0,
            1.0 + 0.05 * (ndf["github_activity_raw"] / 100.0),
            1.0
        )
    else:
        ndf["github_mult"] = 1.0

    ndf["final_score"] = (
        ndf["base_score"]
      * ndf["risk_multiplier"]
      * ndf["availability_multiplier"]
      * ndf["experience_fit"]
      * ndf["product_multiplier"]
      * ndf["production_gate_mult"]
      * ndf["recent_llm_mult"]
      * ndf["non_coding_mult"]
      * ndf["job_hop_mult"]
      * ndf["github_mult"]
    )

    print(f"  Score range: {ndf['final_score'].min():.4f} – {ndf['final_score'].max():.4f}")
    return ndf


def generate_reasoning(cid, feat_row, candidates_lookup):
    c = candidates_lookup.get(cid, {})
    p = c.get("profile", {})
    skills_set = {s["name"] for s in c.get("skills", [])}

    title  = p.get("current_title", "")
    exp    = p.get("years_of_experience", 0)
    city   = p.get("location", "").split(",")[0].strip()

    parts    = []
    concerns = []

    ret_skills = [sk for sk in PRIORITY_SKILLS if sk in skills_set][:3]
    if ret_skills:
        parts.append(f"{exp:.0f}yr {title.lower()} with {'/'.join(ret_skills)}")
    else:
        parts.append(f"{exp:.0f}yr {title.lower()}")

    if feat_row["evaluation_signal_score"] >= 0.4:
        parts.append("career documents evaluation metric ownership (NDCG/MRR)")
    elif feat_row["evaluation_signal_score"] >= 0.2:
        parts.append("evaluation metrics in career history")

    if feat_row["production_signal_score"] >= 0.4:
        parts.append("production deployment at scale")
    elif feat_row["production_signal_score"] >= 0.2:
        parts.append("production deployment evidence")

    if int(feat_row["expert_ai_skills"]) >= 2:
        parts.append(f"{int(feat_row['expert_ai_skills'])} expert-level AI skills")

    if feat_row["avg_ai_assessment_score"] >= 70:
        parts.append(f"platform-verified {feat_row['avg_ai_assessment_score']:.0f}/100")

    rr = feat_row["recruiter_response_rate"]
    if rr >= 0.75:
        parts.append(f"strong recruiter engagement ({rr:.0%})")
    elif rr >= 0.55:
        parts.append(f"responsive ({rr:.0%} response rate)")

    if city.lower() in PREFERRED_CITIES:
        parts.append(f"based in {city}")

    if feat_row["consulting_ratio"] >= 0.8:
        concerns.append("primarily consulting background")
    if feat_row["days_since_active"] > 180:
        concerns.append(f"inactive {int(feat_row['days_since_active'])//30}mo on platform")
    if feat_row["notice_period"] > 90:
        concerns.append(f"{int(feat_row['notice_period'])}d notice period")
    if int(feat_row["is_honeypot"]) == 1:
        concerns.append("profile inconsistencies flagged")
    if "title_credibility" in feat_row and feat_row["title_credibility"] < 0.5:
        concerns.append("title/skill mismatch")
    if "recent_llm_only" in feat_row and int(feat_row["recent_llm_only"]) == 1:
        concerns.append("recent LLM-only experience, no pre-LLM ML depth")
    if "non_coding_role" in feat_row and int(feat_row["non_coding_role"]) == 1:
        concerns.append("current role is non-coding (management/architecture)")
    if "job_hopper" in feat_row and int(feat_row["job_hopper"]) == 1:
        concerns.append("frequent job changes")

    reasoning = "; ".join(parts[:4])
    if concerns:
        reasoning += ". Concerns: " + ", ".join(concerns)
    return (reasoning + ".").strip()[:250]


def validate_and_export(top100_df, submission_path):
    sub = top100_df[["candidate_id","rank","final_score","reasoning"]].copy()
    sub = sub.rename(columns={"final_score": "score"})
    sub = sub[["candidate_id","rank","score","reasoning"]]

    errors = []
    if len(sub) != 100:
        errors.append(f"Row count {len(sub)} != 100")
    if sorted(sub["rank"].tolist()) != list(range(1, 101)):
        errors.append("Ranks not exactly 1-100")
    if sub["candidate_id"].nunique() != 100:
        errors.append("Duplicate candidate_ids")
    diffs = sub.sort_values("rank")["score"].diff().dropna()
    if not (diffs <= 1e-10).all():
        errors.append("Scores not non-increasing")
    empty = (sub["reasoning"].isna() | (sub["reasoning"].str.strip() == "")).sum()
    if empty > 0:
        errors.append(f"{empty} empty reasoning strings")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ❌ {e}")
        sys.exit(1)

    sub.to_csv(submission_path, index=False)
    print(f"\n{'='*55}")
    print(f"✅  SUBMISSION VALID — {submission_path}")
    print(f"{'='*55}")
    print(f"  Rows   : {len(sub)}")
    print(f"  Rank 1  score : {sub[sub['rank']==1]['score'].values[0]:.4f}")
    print(f"  Rank 100 score: {sub[sub['rank']==100]['score'].values[0]:.4f}")
    print(f"  Reasoning avg : {sub['reasoning'].str.len().mean():.0f} chars")
    print(f"\n  Ready to upload as REDRO_AI.csv")


def sync_notebook_outputs(sub_df, full_features_df):
    """
    Fix: Notebook/outputs/submission.csv and features_df.csv were stale —
    last written by an earlier notebook run, before later formula edits in
    rank.py. app.py's "Default JD" tab reads from these files, so judges
    viewing the live demo were seeing 86/100 different candidates than the
    actual REDRO_AI.csv submission, contradicting the README's explicit
    "Demo = Submission, no divergence" claim.
    rank.py is now the single source of truth: every run refreshes both
    files, so this can't drift again regardless of whether notebooks are
    re-executed.
    """
    out_dir = os.path.join(ROOT, "Notebook", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    sub_df.to_csv(os.path.join(out_dir, "submission.csv"), index=False)

    export_df = full_features_df.copy()
    if "semantic_percentile" not in export_df.columns and "sem_capped" in export_df.columns:
        export_df["semantic_percentile"] = export_df["sem_capped"]
    export_df.to_csv(os.path.join(out_dir, "features_df.csv"), index=False)
    print(f"  Synced Notebook/outputs/submission.csv + features_df.csv (no more drift)")


def main():
    print("\n" + "="*55)
    print("  REDRO AI — Candidate Ranking Pipeline")
    print("="*55 + "\n")

    candidates = load_candidates(DATASET_PATH)
    candidates_lookup = {c["candidate_id"]: c for c in candidates}

    similarities = get_semantic_similarities(candidates, CACHE_EMBEDDINGS)

    features_df = engineer_features(candidates, similarities)

    build_faiss_index(candidates)
    save_scoring_thresholds(features_df)

    ALL_STAGES = {"title_cred","prod_gate","recent_llm","non_coding","job_hop","github"}
    scored_df = rank_candidates(features_df, stages=ALL_STAGES)

    ranked = (scored_df
              .sort_values("final_score", ascending=False)
              .reset_index(drop=True))
    ranked["rank"] = ranked.index + 1
    top100 = ranked.head(100).copy()

    print("Generating reasoning for top 100...")
    feat_idx = scored_df.set_index("candidate_id")
    top100["reasoning"] = top100["candidate_id"].apply(
        lambda cid: generate_reasoning(cid, feat_idx.loc[cid], candidates_lookup)
    )

    validate_and_export(top100, OUTPUT_CSV)
    sync_notebook_outputs(top100[["candidate_id","rank","final_score","reasoning"]]
                          .rename(columns={"final_score":"score"}), scored_df)

    print("\nTop 10 Candidates:")
    print(f"{'Rank':>5}  {'Candidate ID':<15}  {'Score':>8}  Reasoning")
    print("-" * 80)
    for _, row in top100.head(10).iterrows():
        print(f"{int(row['rank']):>5}  {row['candidate_id']:<15}  {row['final_score']:>8.4f}  {row['reasoning'][:50]}...")

    print(f"\n✅  Done. Submit: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()