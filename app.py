"""
REDRO AI — Candidate Ranking Demo
===================================
Run: streamlit run app.py

Two modes:
  Default JD  — shows precomputed submission.csv results (exact match)
  Custom JD   — live ranking using same sentence-transformers + NB03 formula
"""

import json, os, math, re, pickle, warnings, hashlib
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Optional FAISS (local only — degrades gracefully on Streamlit Cloud) ──────
try:
    import faiss as _faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="REDRO AI — Candidate Ranking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NB03-identical constants ────────────────────────────────────────────────────
RETRIEVAL_SKILLS = {
    "Embeddings","FAISS","Milvus","Elasticsearch","BM25",
    "Information Retrieval","Vector Search","Pinecone","Recommendation Systems",
}
LLM_SKILLS = {"LangChain","Prompt Engineering","Fine-tuning LLMs","RAG","PEFT","LoRA"}
ML_SKILLS  = {
    "Machine Learning","Deep Learning","PyTorch","TensorFlow","MLflow",
    "Sentence Transformers","scikit-learn","Learning to Rank","XGBoost",
}
AI_ALL = RETRIEVAL_SKILLS | LLM_SKILLS | ML_SKILLS

CONSULTING_FIRMS = {"tcs","infosys","wipro","accenture","cognizant","capgemini","hcl","tech mahindra"}
PRODUCT_COS      = {"google","amazon","uber","swiggy","zomato","flipkart","microsoft",
                    "netflix","meta","apple","linkedin","spotify","airbnb","stripe"}
PREFERRED_CITIES  = {"pune","noida"}
ACCEPTABLE_CITIES = {"hyderabad","mumbai","bangalore","bengaluru","delhi","new delhi","gurgaon","gurugram"}
REFERENCE_DATE    = date(2026, 6, 5)

# NB03 CAP_WEIGHTS — identical to NB03 notebook
CAP_WEIGHTS = {
    "sem_capped"                  : 0.25,
    "eval_combo"                  : 0.15,
    "production_signal_score_pct" : 0.15,
    "retrieval_score_pct"         : 0.18,
    "quality_score_log_pct"       : 0.11,
    "career_keyword_score_pct"    : 0.07,
    "avg_ai_assessment_score_pct" : 0.09,
}

PRIORITY_SKILLS = [
    "FAISS","Embeddings","Elasticsearch","Information Retrieval","Pinecone",
    "Milvus","Vector Search","BM25","Sentence Transformers","LangChain",
    "RAG","Learning to Rank","Recommendation Systems","Machine Learning",
]

DEFAULT_JD = """Senior AI Engineer — Retrieval, Search, Ranking and Recommendation Systems

Production experience with embeddings-based retrieval systems.
Vector databases: FAISS, Pinecone, Milvus, Weaviate, Elasticsearch.
Hybrid search, BM25, semantic search, information retrieval.
Evaluation: NDCG, MRR, MAP, A/B testing, offline evaluation.
Learning to rank, search relevance, retrieval quality, ranking systems.
Recommendation systems, candidate matching, similarity search.
LLM: RAG, LangChain, Prompt Engineering, fine-tuning.
Sentence transformers, embedding models.
Python, production ML, MLOps, 5-9 years experience.
Pune, Noida, Hyderabad, Mumbai, Delhi NCR."""


# ── Data loading (cached) ───────────────────────────────────────────────────────
def _find_output(filename):
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "Notebook", "outputs", filename)
    return path if os.path.exists(path) else None

@st.cache_data(show_spinner="Loading feature table…")
def load_features():
    path = _find_output("features_df.pkl")
    if path:
        try:
            return pd.read_pickle(path)
        except Exception as e:
            st.warning(f"⚠️ pkl load failed ({e}) — falling back to CSV.")

    path = _find_output("features_df.csv")
    if path:
        df = pd.read_csv(path)
        if "saved_by_recruiters_norm" not in df.columns and "saved_by_recruiters_raw" in df.columns:
            df["saved_by_recruiters_norm"] = (
                df["saved_by_recruiters_raw"] / df["saved_by_recruiters_raw"].max()
            )
        if "profile_views_norm" not in df.columns and "profile_views_raw" in df.columns:
            df["profile_views_norm"] = (
                df["profile_views_raw"] / df["profile_views_raw"].max()
            )
        if "search_appearance_norm" not in df.columns and "search_appearance_raw" in df.columns:
            df["search_appearance_norm"] = (
                df["search_appearance_raw"] / df["search_appearance_raw"].max()
            )
        if "hidden_signal_bonus" not in df.columns and "hidden_signal_count" in df.columns:
            df["hidden_signal_bonus"] = (df["hidden_signal_count"] * 0.03).clip(upper=0.15)
        return df

    return None


@st.cache_data(show_spinner="Loading submission…")
def load_submission():
    path = _find_output("submission.csv")
    return pd.read_csv(path) if path else None


@st.cache_data(show_spinner="Loading candidate profiles…")
def load_candidates_lookup(n_read=20000):
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "raw_dataset", "candidates.jsonl")
    if not os.path.exists(path):
        return {}
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_read:
                break
            line = line.strip()
            if line:
                c = json.loads(line)
                lookup[c["candidate_id"]] = c
    return lookup


@st.cache_resource(show_spinner="Loading sentence-transformer model…")
def load_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


@st.cache_resource(show_spinner=False)
def load_faiss_index():
    if not _FAISS_AVAILABLE:
        return None, None
    idx_path = _find_output("faiss_index.bin")
    ids_path = _find_output("faiss_ids.npy")
    if not idx_path or not ids_path:
        return None, None
    try:
        index = _faiss.read_index(idx_path)
        ids   = np.load(ids_path, allow_pickle=True)
        return index, ids
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def load_scoring_thresholds():
    path = _find_output("scoring_thresholds.pkl")
    if not path:
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


# FIX #8: Pass a stable hash of features_df instead of the DataFrame itself
# to avoid expensive re-hashing on every call.
@st.cache_data(show_spinner="Pre-computing candidate embeddings for custom JD mode…")
def get_sample_texts_and_ids(_features_df_hash: str, features_df, n=3000):
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "raw_dataset", "candidates.jsonl")
    if not os.path.exists(path):
        return [], []

    feat = features_df.copy()
    tier1 = feat.nlargest(n // 3, "evaluation_signal_score")["candidate_id"].tolist()
    tier2 = feat.nlargest(n // 3, "retrieval_score")["candidate_id"].tolist()
    selected = list(dict.fromkeys(tier1 + tier2))[:n]
    remaining_n = n - len(selected)
    if remaining_n > 0:
        not_selected = feat[~feat["candidate_id"].isin(selected)]
        selected += not_selected.sample(min(remaining_n, len(not_selected)),
                                        random_state=42)["candidate_id"].tolist()
    selected_set = set(selected)

    texts, ids = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid = c["candidate_id"]
            if cid in selected_set:
                texts.append(_build_text(c))
                ids.append(cid)
                if len(ids) >= n:
                    break
    return ids, texts


def _build_text(c):
    parts = [
        c["profile"].get("current_title", ""),
        c["profile"].get("headline", ""),
        c["profile"].get("summary", ""),
        " ".join(s["name"] for s in c.get("skills", [])),
    ]
    for job in c.get("career_history", []):
        parts += [job.get("title", ""), job.get("description", "")]
    return " ".join(p for p in parts if p)


# ── Scoring — IDENTICAL to NB03 ─────────────────────────────────────────────────
def apply_nb03_formula(feat_df, semantic_percentile_override=None):
    ndf = feat_df.copy()

    if semantic_percentile_override is not None:
        ndf["semantic_percentile"] = semantic_percentile_override

    RANK_COLS = [
        "retrieval_score", "llm_score", "ml_score", "ai_skill_total",
        "quality_score_log", "avg_ai_duration", "advanced_ai_skills",
        "expert_ai_skills", "max_endorsements_ai", "avg_assessment_score",
        "avg_ai_assessment_score", "ai_cert_count", "evaluation_signal_score",
        "production_signal_score", "career_keyword_score", "hidden_signal_bonus",
        "saved_by_recruiters_norm", "profile_views_norm", "search_appearance_norm",
    ]
    for col in RANK_COLS:
        if col in ndf.columns:
            ndf[f"{col}_pct"] = ndf[col].fillna(0).rank(pct=True, method="average")

    ndf["eval_combo"]  = (0.6 * ndf["evaluation_signal_score_pct"] +
                          0.4 * (ndf["evaluation_signal_score"] > 0).astype(float))
    ndf["sem_capped"]  = ndf["semantic_percentile"].rank(pct=True).clip(upper=0.97)
    ndf["avail_pct"]   = ndf["availability_score"].rank(pct=True, method="average")
    ndf["saved_pct"]   = ndf["saved_by_recruiters_norm"].rank(pct=True, method="average")
    ndf["views_pct"]   = ndf["profile_views_norm"].rank(pct=True, method="average")

    ndf["capability_score"] = sum(ndf[col] * w for col, w in CAP_WEIGHTS.items())

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
        return 0.60

    ndf["experience_fit"]  = ndf["experience_years"].apply(exp_mult)
    ndf["avail_mult"]      = ndf["availability_score"].clip(lower=0.30, upper=1.10)
    ndf["risk_multiplier"] = ((1 - 0.80 * ndf["consulting_ratio"]) *
                               ndf["is_honeypot"].map({0: 1.0, 1: 0.05}))

    ndf["final_score"] = (
        ndf["base_score"]
        * ndf["risk_multiplier"]
        * ndf["avail_mult"]
        * ndf["experience_fit"]
    )
    return ndf


def generate_reasoning(cid, feat_row, cands_lookup):
    c  = cands_lookup.get(cid, {})
    p  = c.get("profile", {})
    sk = {s["name"] for s in c.get("skills", [])}
    exp   = p.get("years_of_experience", feat_row.get("experience_years", 0))
    title = p.get("current_title", "")
    city  = p.get("location", "").split(",")[0].strip()

    ret   = [s for s in PRIORITY_SKILLS if s in sk][:3]
    parts = [f"{exp:.0f}yr {title.lower()}" + (f" · {'/'.join(ret)}" if ret else "")]

    if feat_row.get("evaluation_signal_score", 0) >= 0.35:
        parts.append("evaluation metrics in career history (NDCG/MRR)")
    if feat_row.get("production_signal_score", 0) >= 0.35:
        parts.append("production deployment evidence")
    if feat_row.get("expert_ai_skills", 0) >= 2:
        parts.append(f"{int(feat_row['expert_ai_skills'])} expert AI skills")
    rr = feat_row.get("recruiter_response_rate", 0)
    if rr >= 0.75:
        parts.append(f"strong recruiter engagement ({rr:.0%})")

    concerns = []
    if feat_row.get("consulting_ratio", 0) >= 0.8: concerns.append("consulting background")
    if feat_row.get("days_since_active", 0) > 180:  concerns.append(f"inactive {int(feat_row['days_since_active'])//30}mo")
    if feat_row.get("notice_period", 0) > 90:        concerns.append(f"{int(feat_row['notice_period'])}d notice")

    out = "; ".join(parts[:3])
    if concerns: out += ". Concerns: " + ", ".join(concerns)
    return (out + ".").strip()[:250]


# ── UI helpers ──────────────────────────────────────────────────────────────────
def render_candidate_card(rank, row, cands_lookup, show_breakdown):
    cid    = row["candidate_id"]
    c      = cands_lookup.get(cid, {})
    p      = c.get("profile", {})
    skills = {s["name"] for s in c.get("skills", [])}

    title  = p.get("current_title", row.get("title", "—"))
    exp    = p.get("years_of_experience", row.get("experience_years", 0))
    loc    = p.get("location", "—")
    jd_sk  = [s for s in PRIORITY_SKILLS if s in skills][:5]
    score  = row.get("final_score", row.get("score", 0))
    reason = row.get("reasoning", generate_reasoning(cid, row, cands_lookup))

    if rank == 1:   medal, badge_bg, badge_fg, tier_border = "🥇", "#fef3c7", "#92400e", "#fbbf24"
    elif rank == 2: medal, badge_bg, badge_fg, tier_border = "🥈", "#e2e8f0", "#334155", "#94a3b8"
    elif rank == 3: medal, badge_bg, badge_fg, tier_border = "🥉", "#ffedd5", "#9a3412", "#fb923c"
    elif rank <= 10:medal, badge_bg, badge_fg, tier_border = "",   "#eef2ff", "#3730a3", "#818cf8"
    else:           medal, badge_bg, badge_fg, tier_border = "",   "#f1f5f9", "#475569", "#cbd5e1"

    prefix = f"{medal} " if medal else ""
    exp_label = f"{exp:.0f} yr exp"
    score_label = f"{score:.4f}"

    # FIX #10: Use a single `with col1` block — don't re-enter the same column context.
    with st.expander(
        f"{prefix}#{rank} · {title}  ·  {exp_label}  ·  {loc}  ·  Score {score_label}",
        expanded=(rank <= 5)
    ):
        col1, col2 = st.columns([3, 2] if show_breakdown else [1, 0])

        with col1:
            # Header row
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:16px;"
                f"padding-bottom:14px;border-bottom:1px solid #e9edf6'>"
                f"<div style='display:flex;flex-direction:column;gap:4px;flex:1'>"
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<span style='font-size:10px;font-weight:700;color:#9ca3af;"
                f"text-transform:uppercase;letter-spacing:0.1em'>Candidate ID</span>"
                f"<code style='background:#ede9fe;color:#4c1d95;border-radius:6px;"
                f"padding:3px 10px;font-size:13px;font-weight:700;letter-spacing:0.03em;"
                f"border:1px solid #ddd6fe'>{cid}</code>"
                f"</div>"
                f"<div style='display:flex;align-items:center;gap:6px;margin-top:2px'>"
                f"<span style='background:{badge_bg};color:{badge_fg};"
                f"border:1px solid {tier_border};border-radius:9999px;"
                f"padding:3px 12px;font-size:11px;font-weight:700;letter-spacing:0.02em'>"
                f"{'🏆 ' if rank == 1 else ''}Rank #{rank}</span>"
                f"<span style='background:#f0fdf4;color:#15803d;"
                f"border:1px solid #bbf7d0;border-radius:9999px;"
                f"padding:3px 12px;font-size:11px;font-weight:600'>⏱ {exp_label}</span>"
                f"<span style='background:#f8faff;color:#4f46e5;"
                f"border:1px solid #e0e7ff;border-radius:9999px;"
                f"padding:3px 12px;font-size:11px;font-weight:700'>Score: {score_label}</span>"
                f"</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            # JD-relevant skills chips
            if jd_sk:
                chips = "".join(
                    f"<span style='display:inline-flex;align-items:center;"
                    f"background:#eef2ff;color:#3730a3;"
                    f"border-radius:9999px;padding:4px 13px;font-size:12px;font-weight:600;"
                    f"margin:3px 5px 3px 0;border:1px solid #e0e7ff;"
                    f"letter-spacing:0.01em'>{s}</span>"
                    for s in jd_sk
                )
                st.markdown(
                    f"<div style='margin-bottom:16px'>"
                    f"<div style='font-size:10px;font-weight:700;color:#9ca3af;"
                    f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>"
                    f"✅ JD-Relevant Skills</div>"
                    f"<div style='display:flex;flex-wrap:wrap;gap:2px'>{chips}</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='background:#fffbeb;border:1px solid #fde68a;"
                    "border-radius:8px;padding:9px 14px;margin-bottom:16px;"
                    "font-size:12px;color:#92400e;font-weight:500;line-height:1.5'>"
                    "⚠️ &nbsp;No exact JD-priority skills found in profile"
                    "</div>",
                    unsafe_allow_html=True
                )

            # AI Reasoning card
            st.markdown(
                f"<div style='background:linear-gradient(135deg,#f0f7ff 0%,#f5f0ff 100%);"
                f"border-left:3px solid #4f46e5;"
                f"border-radius:0 10px 10px 0;padding:14px 18px;margin-top:4px;"
                f"border:1px solid #e0e7ff;border-left:3px solid #4f46e5'>"
                f"<div style='font-size:10px;font-weight:700;color:#4f46e5;"
                f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;"
                f"display:flex;align-items:center;gap:6px'>"
                f"<span>🎯</span><span>AI Match Reasoning</span></div>"
                f"<div style='font-size:13px;color:#1e293b;line-height:1.75;"
                f"font-weight:400'>{reason}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        if show_breakdown:
            with col2:
                st.markdown(
                    "<div style='font-size:10px;font-weight:700;color:#6b7280;"
                    "text-transform:uppercase;letter-spacing:0.1em;"
                    "margin-bottom:16px;padding-bottom:10px;"
                    "border-bottom:1px solid #e9edf6'>"
                    "📊 Signal Breakdown"
                    "</div>",
                    unsafe_allow_html=True
                )
                components = {
                    "Semantic Align"  : min(1.0, float(row.get("semantic_percentile", 0))),
                    "Eval Evidence"   : float(row.get("evaluation_signal_score", 0)),
                    "Production"      : float(row.get("production_signal_score", 0)),
                    "Retrieval Skills": min(1.0, float(row.get("retrieval_score", 0)) / 45),
                    "Validation"      : float(row.get("recruiter_response_rate", 0)),
                    "Availability"    : float(row.get("availability_score", 0)),
                }
                for name, val in components.items():
                    if val >= 0.7:
                        bar_color, badge_color, badge_text_color = "#16a34a", "#f0fdf4", "#15803d"
                    elif val >= 0.4:
                        bar_color, badge_color, badge_text_color = "#d97706", "#fffbeb", "#92400e"
                    else:
                        bar_color, badge_color, badge_text_color = "#ef4444", "#fef2f2", "#991b1b"
                    pct = int(val * 100)
                    st.markdown(
                        f"<div style='margin-bottom:14px'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:center;margin-bottom:6px'>"
                        f"<span style='font-size:11px;font-weight:600;color:#374151;"
                        f"letter-spacing:0.01em'>{name}</span>"
                        f"<span style='background:{badge_color};color:{badge_text_color};"
                        f"border-radius:6px;padding:2px 8px;"
                        f"font-size:11px;font-weight:700'>{val:.2f}</span>"
                        f"</div>"
                        f"<div style='background:#e9edf6;border-radius:9999px;"
                        f"height:7px;overflow:hidden'>"
                        f"<div style='width:{pct}%;height:100%;background:{bar_color};"
                        f"border-radius:9999px;"
                        f"transition:width 0.4s ease'></div>"
                        f"</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


# ── New-candidate evaluation helpers ─────────────────────────────────────────

_EVAL_TRIGGERS = {
    "fit","evaluate","assess","rank this","compare this",
    "is this","new candidate","hire","should we","what do you think",
    "i have a candidate","i have a candid","candidate with",
    "years of experience","yrs of experience","year of experience",
    "this candidate","working as","previously working",
    "deployed project","deployed system","deployed model",
    "should i hire","would you hire","is he a fit","is she a fit",
}
_INPUT_FIELD_KEYS = {"title","experience","exp","skills","evaluation","eval",
                     "production","prod","location","consulting","notice","github"}

def _is_eval_intent(q, query):
    if any(t in q for t in _EVAL_TRIGGERS):
        return True
    if ":" in query:
        keys_present = {line.split(":")[0].strip().lower()
                        for line in query.split("\n") if ":" in line}
        return bool(keys_present & _INPUT_FIELD_KEYS)
    return False


def _parse_candidate_input(text):
    parsed = {
        "title": "", "experience": 0.0, "skills": [],
        "has_eval_signal": False, "has_production": False,
        "location": "", "consulting_ratio": 0.0,
        "notice_days": 30, "github_score": -1.0,
    }
    q = text.lower()

    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip().lower(), val.strip()

        if key in ("title", "current title", "role", "designation"):
            parsed["title"] = val
        elif key in ("experience", "exp", "years", "yrs", "experience years"):
            m = re.search(r"\d+\.?\d*", val)
            if m: parsed["experience"] = float(m.group())
        elif key in ("skills", "tech stack", "technologies", "tools"):
            parsed["skills"] = [s.strip() for s in re.split(r"[,;]", val) if s.strip()]
        elif key in ("evaluation", "eval", "metrics", "evaluation metrics"):
            parsed["has_eval_signal"] = any(
                kw in val.lower() for kw in ["ndcg","mrr","map","a/b","offline","ranking quality"]
            )
        elif key in ("production", "prod", "deployed", "deployment"):
            parsed["has_production"] = any(
                kw in val.lower() for kw in ["deploy","ship","production","serving","million","scale","users"]
            )
        elif key in ("location", "city", "base"):
            parsed["location"] = val.lower()
        elif key in ("consulting", "consulting_ratio", "outsourcing"):
            m = re.search(r"\d+\.?\d*", val)
            if m:
                v = float(m.group())
                parsed["consulting_ratio"] = v / 100 if v > 1 else v
        elif key in ("notice", "notice period", "notice_period"):
            m = re.search(r"\d+", val)
            if m: parsed["notice_days"] = int(m.group())
        elif key in ("github", "github score", "open source", "opensource"):
            m = re.search(r"\d+\.?\d*", val)
            if m: parsed["github_score"] = float(m.group())

    if parsed["experience"] == 0.0:
        m = re.search(r'(\d+\.?\d*)\s*(?:yr|year)', q)
        if m:
            parsed["experience"] = float(m.group(1))
    if not parsed["has_eval_signal"]:
        parsed["has_eval_signal"] = any(kw in q for kw in ["ndcg","mrr","map","a/b test","offline eval","ranking metric"])
    if not parsed["has_production"]:
        parsed["has_production"] = any(kw in q for kw in [
            "deployed","shipped","production","at scale","serving",
            "deployed project","deployed system","deployed model","in production",
        ])
    ALL_KNOWN_SKILLS = (RETRIEVAL_SKILLS | LLM_SKILLS | ML_SKILLS |
                        {"Python","Docker","Kubernetes","AWS","GCP","Azure"})
    if not parsed["skills"]:
        found = [s for s in ALL_KNOWN_SKILLS if s.lower() in q]
        if found:
            parsed["skills"] = found
    if not parsed["title"]:
        m = re.search(r'working as (?:an? )?([a-z ]{3,30}?)(?:\.|,|$| in)', q)
        if m:
            parsed["title"] = m.group(1).strip().title()

    return parsed


def _estimate_pct(thresholds, col, value):
    arr = thresholds.get(col)
    if arr is None or len(arr) == 0:
        return 0.5
    idx = np.searchsorted(arr, value, side="right")
    return float(idx) / len(arr)


def _score_new_candidate(parsed, thresholds, faiss_index, faiss_ids, features_df, model):
    nearest_ids, semantic_pct = [], 0.55
    text = (f"{parsed['title']} " +
            " ".join(parsed["skills"]) +
            (" NDCG MRR MAP evaluation metrics" if parsed["has_eval_signal"] else "") +
            (" deployed production serving at scale" if parsed["has_production"] else ""))

    if faiss_index is not None and model is not None:
        try:
            emb = model.encode([text], convert_to_numpy=True).astype("float32")
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb /= np.where(norm == 0, 1.0, norm)
            scores, indices = faiss_index.search(emb, 5)
            nearest_ids = [str(faiss_ids[i]) for i in indices[0] if i >= 0]
            top_sim = float(scores[0][0])
            if "semantic_similarity" in thresholds:
                semantic_pct = _estimate_pct(thresholds, "semantic_similarity", top_sim)
            else:
                semantic_pct = min(top_sim, 0.97)
        except Exception:
            pass
    else:
        jd_all = {s.lower() for s in (RETRIEVAL_SKILLS | LLM_SKILLS | ML_SKILLS)}
        n_matched = len({s.lower() for s in parsed["skills"]} & jd_all)
        semantic_pct = 0.50 + 0.45 * min(n_matched / 8.0, 1.0)

    skill_set = {s.strip() for s in parsed["skills"]}
    skill_lower = {s.lower() for s in skill_set}

    ret_count   = sum(1 for rs in RETRIEVAL_SKILLS if rs.lower() in skill_lower)
    ret_raw     = ret_count * 5.0
    eval_raw    = 0.75 if parsed["has_eval_signal"] else 0.0
    prod_raw    = 0.65 if parsed["has_production"]  else 0.0
    quality_raw = min(1.0, ret_count * 0.2)
    kw_raw      = min(1.0, sum(1 for s in skill_lower if any(s in a.lower() for a in AI_ALL)) * 0.08)
    assessment  = 0.65

    notice = parsed["notice_days"]
    avail  = 1.0 if notice <= 30 else (0.85 if notice <= 60 else (0.70 if notice <= 90 else 0.50))

    sem_capped  = min(semantic_pct, 0.97)
    eval_pct    = _estimate_pct(thresholds, "evaluation_signal_score",   eval_raw)
    prod_pct    = _estimate_pct(thresholds, "production_signal_score",   prod_raw)
    ret_pct     = _estimate_pct(thresholds, "retrieval_score",           ret_raw)
    qual_pct    = _estimate_pct(thresholds, "quality_score_log",         quality_raw)
    kw_pct      = _estimate_pct(thresholds, "career_keyword_score",      kw_raw)
    assess_pct  = _estimate_pct(thresholds, "avg_ai_assessment_score",   assessment)
    avail_pct   = _estimate_pct(thresholds, "availability_score",        avail)

    eval_combo  = 0.6 * eval_pct + 0.4 * float(eval_raw > 0)

    capability  = (0.25*sem_capped + 0.15*eval_combo + 0.15*prod_pct +
                   0.18*ret_pct    + 0.11*qual_pct   + 0.07*kw_pct + 0.09*assess_pct)
    validation  = 0.40*0.50 + 0.30*0.60 + 0.20*0.80 + 0.10*0.40
    base        = 0.60*capability + 0.25*validation + 0.15*avail_pct

    exp = parsed["experience"]
    if 5 <= exp <= 9:    exp_fit = 1.00
    elif 4 <= exp < 5:   exp_fit = 0.90
    elif 9 < exp <= 12:  exp_fit = 0.85
    elif 3 <= exp < 4:   exp_fit = 0.75
    else:                exp_fit = 0.60

    consulting_mult = 1 - 0.80 * parsed["consulting_ratio"]
    github_mult     = 1.0 + 0.05 * (parsed["github_score"] / 100.0) if parsed["github_score"] >= 0 else 1.0
    prod_gate       = 1.0 if parsed["has_production"] else 0.5

    final_score = base * consulting_mult * avail * exp_fit * github_mult * prod_gate

    if final_score >= 0.87 and exp_fit == 1.0 and parsed["has_eval_signal"] and parsed["has_production"]:
        verdict = "STRONG FIT ✅"
    elif final_score >= 0.82 and exp_fit >= 0.9:
        verdict = "MODERATE FIT 🟡"
    elif exp_fit < 0.7:
        verdict = f"NOT A FIT ❌ — experience {exp:.1f}yr outside JD range (5–9yr)"
    elif not parsed["has_production"]:
        verdict = "WEAK FIT 🔴 — no production deployment evidence"
    else:
        verdict = "WEAK FIT 🔴"

    if   final_score >= 0.97: rank_band = "top 5"
    elif final_score >= 0.93: rank_band = "top 10"
    elif final_score >= 0.88: rank_band = "top 25"
    elif final_score >= 0.83: rank_band = "top 50"
    elif final_score >= 0.78: rank_band = "top 100"
    else:                     rank_band = "outside top 100"

    jd_matched = [s for s in parsed["skills"]
                  if any(s.lower() == r.lower() for r in RETRIEVAL_SKILLS)]
    missing    = [m for m in [
        "Evaluation metrics in career history (NDCG/MRR/MAP)" if not parsed["has_eval_signal"] else None,
        "Production deployment evidence"                        if not parsed["has_production"]  else None,
        f"Experience {exp:.1f}yr outside JD band (5–9yr)"     if exp_fit < 0.9               else None,
    ] if m]

    return {
        "score": final_score, "rank_band": rank_band, "verdict": verdict,
        "exp_fit": exp_fit,   "nearest_ids": nearest_ids,
        "has_faiss": faiss_index is not None,
        "signals": {
            "Semantic align":    f"{sem_capped:.2f}",
            "Eval evidence":     f"{eval_raw:.2f}  {'✅' if parsed['has_eval_signal'] else '❌'}",
            "Production signal": f"{prod_raw:.2f}  {'✅' if parsed['has_production']  else '❌'}",
            "Retrieval skills":  f"{ret_count} matched: {', '.join(jd_matched[:4]) or 'none'}",
            "Experience fit":    f"{exp_fit:.2f}  ({exp:.1f}yr)",
            "Consulting risk":   f"{parsed['consulting_ratio']:.0%}" if parsed["consulting_ratio"] > 0 else "None",
            "GitHub activity":   f"{parsed['github_score']:.0f}/100" if parsed["github_score"] >= 0 else "Not linked",
        },
        "missing": missing,
    }


def _render_eval_result(result, chat_df, cands_lookup):
    verdict_color = (
        "#16a34a" if "STRONG" in result["verdict"]
        else "#f59e0b" if "MODERATE" in result["verdict"]
        else "#ef4444"
    )
    st.markdown(
        f"<div style='background:{verdict_color}18;border:1px solid {verdict_color}40;"
        f"border-radius:10px;padding:14px 18px;margin-bottom:16px'>"
        f"<div style='font-size:18px;font-weight:700;color:{verdict_color}'>{result['verdict']}</div>"
        f"<div style='font-size:13px;color:#464555;margin-top:4px'>"
        f"Score <strong>{result['score']:.4f}</strong> → "
        f"<strong>{result['rank_band']}</strong> in pool</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    if not result["has_faiss"]:
        st.caption("⚠️  FAISS index not loaded — semantic similarity estimated from skills only. "
                   "Run `python rank.py` locally for full accuracy.")
    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:#9ca3af;"
        "text-transform:uppercase;letter-spacing:0.09em;margin-bottom:8px'>"
        "Signal Breakdown</div>",
        unsafe_allow_html=True
    )
    rows = [{"Signal": k, "Value": v} for k, v in result["signals"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if result["missing"]:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;color:#9ca3af;"
            "text-transform:uppercase;letter-spacing:0.09em;margin:12px 0 8px'>"
            "What's Missing</div>",
            unsafe_allow_html=True
        )
        for m in result["missing"]:
            st.markdown(
                f"<div style='background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;"
                f"padding:8px 12px;margin-bottom:6px;font-size:13px;color:#9a3412'>⚠ {m}</div>",
                unsafe_allow_html=True
            )
    if result["nearest_ids"]:
        st.markdown(
            "<div style='font-size:10px;font-weight:700;color:#9ca3af;"
            "text-transform:uppercase;letter-spacing:0.09em;margin:12px 0 8px'>"
            "Most Similar Candidates in Top-100</div>",
            unsafe_allow_html=True
        )
        for nid in result["nearest_ids"][:3]:
            row = chat_df[chat_df["candidate_id"] == nid]
            if not row.empty:
                render_candidate_card(int(row.iloc[0]["rank"]), row.iloc[0], cands_lookup, False)


# ── Chat helpers ─────────────────────────────────────────────────────────────────
def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else default
    except (TypeError, ValueError):
        return default


def _chat_parse_and_respond(query, chat_df, cands_lookup, faiss_index=None,
                             faiss_ids=None, features_df=None, thresholds=None, model=None):
    q    = query.lower().strip()
    nums = [int(n) for n in re.findall(r'\b(\d{1,3})(?:st|nd|rd|th)?\b', q) if 1 <= int(n) <= 100]

    if _is_eval_intent(q, query):
        parsed = _parse_candidate_input(query)
        if parsed["experience"] == 0 and not parsed["skills"] and not parsed["title"]:
            return {
                "role": "assistant",
                "content": (
                    "Paste candidate details in this format:\n\n"
                    "```\n"
                    "title: Senior ML Engineer\n"
                    "experience: 6\n"
                    "skills: FAISS, Embeddings, BM25, Elasticsearch\n"
                    "evaluation: improved NDCG@10 by 12%\n"
                    "production: deployed recommendation system at scale\n"
                    "location: Pune\n"
                    "consulting: 0\n"
                    "notice: 30\n"
                    "github: 75\n"
                    "```\n"
                    "I'll score them against the JD and compare to the ranked pool."
                ),
                "show_ranks": [], "comparison": None, "eval_result": None,
            }
        result = _score_new_candidate(
            parsed, thresholds or {}, faiss_index, faiss_ids, features_df, model
        )
        return {
            "role": "assistant",
            "content": f"**New candidate evaluation** — `{parsed['title'] or 'Unknown title'}` · {parsed['experience']:.0f}yr",
            "show_ranks": [], "comparison": None, "eval_result": result,
        }

    is_compare = any(w in q for w in ["compare","vs","versus","why","better","differ","difference"])
    is_range   = bool(re.search(r'\b\d+\s*(?:to|through)\s*\d+\b|\b\d+-\d+\b', query, re.IGNORECASE)) and len(nums) == 2
    has_multi  = ("," in query) or (re.search(r'\band\b', q) and len(nums) > 1)

    if is_compare and len(nums) >= 2:
        r1, r2 = nums[0], nums[1]
        row1 = chat_df[chat_df["rank"] == r1]
        row2 = chat_df[chat_df["rank"] == r2]
        if row1.empty:
            return {"role":"assistant","content":f"Rank {r1} not found in submission.","show_ranks":[],"comparison":None,"eval_result":None}
        if row2.empty:
            return {"role":"assistant","content":f"Rank {r2} not found in submission.","show_ranks":[],"comparison":None,"eval_result":None}

        row1, row2 = row1.iloc[0], row2.iloc[0]

        SIGNALS = [
            ("Final Score",       "score"),
            ("Experience (yr)",   "experience_years"),
            ("Semantic Align",    "semantic_percentile"),
            ("Eval Evidence",     "evaluation_signal_score"),
            ("Production Signal", "production_signal_score"),
            ("Retrieval Score",   "retrieval_score"),
            ("Recruiter Resp.",   "recruiter_response_rate"),
            ("Availability",      "availability_score"),
            ("Consulting Ratio",  "consulting_ratio"),
            ("Expert AI Skills",  "expert_ai_skills"),
        ]
        LOWER_IS_BETTER = {"Consulting Ratio"}

        comp_rows, adv1, adv2 = [], [], []
        for label, col in SIGNALS:
            v1 = round(_safe_float(row1.get(col)), 3)
            v2 = round(_safe_float(row2.get(col)), 3)
            delta = v1 - v2
            if label in LOWER_IS_BETTER:
                edge = f"#{r1} ✓" if delta < -0.001 else (f"#{r2} ✓" if delta > 0.001 else "Tie")
                if delta < -0.001: adv1.append(label)
                elif delta > 0.001: adv2.append(label)
            else:
                edge = f"#{r1} ✓" if delta > 0.001 else (f"#{r2} ✓" if delta < -0.001 else "Tie")
                if delta > 0.001: adv1.append(label)
                elif delta < -0.001: adv2.append(label)
            comp_rows.append({"Signal": label, f"Rank {r1}": v1, f"Rank {r2}": v2, "Edge": edge})

        s1 = _safe_float(row1.get("score"))
        s2 = _safe_float(row2.get("score"))
        winner, loser = (r1, r2) if s1 >= s2 else (r2, r1)
        adv = adv1 if winner == r1 else adv2

        text = (
            f"**Rank {winner}** scores higher ({max(s1,s2):.4f} vs {min(s1,s2):.4f}).  \n"
            f"Key advantages over Rank {loser}: **{', '.join(adv[:4]) if adv else 'marginal across all signals'}**."
        )
        return {"role":"assistant","content":text,"show_ranks":[r1,r2],"comparison":comp_rows,"eval_result":None}

    elif is_range and len(nums) == 2:
        r_start, r_end = min(nums), max(nums)
        cap_note = ""
        if r_end - r_start > 19:
            orig_end = r_end
            r_end    = r_start + 19
            cap_note = f" *(capped at 20; asked up to {orig_end})*"
        return {
            "role": "assistant",
            "content": f"Showing ranks **{r_start}–{r_end}**{cap_note}:",
            "show_ranks": list(range(r_start, r_end + 1)),
            "comparison": None, "eval_result": None,
        }

    elif has_multi and len(nums) > 1 and not is_compare:
        return {
            "role": "assistant",
            "content": f"Showing candidates at rank **{', '.join(str(n) for n in sorted(nums))}**:",
            "show_ranks": sorted(nums),
            "comparison": None, "eval_result": None,
        }

    elif len(nums) == 1:
        r   = nums[0]
        row = chat_df[chat_df["rank"] == r]
        if row.empty:
            return {"role":"assistant","content":f"No candidate found at rank {r}.","show_ranks":[],"comparison":None,"eval_result":None}
        return {"role":"assistant","content":f"Candidate at rank **{r}**:","show_ranks":[r],"comparison":None,"eval_result":None}

    else:
        return {
            "role": "assistant",
            "content": (
                "Here's what I understand:\n\n"
                "| Query | Example |\n"
                "|---|---|\n"
                "| Single rank | `rank 5` · `5` · `#5` |\n"
                "| Range | `11 to 20` · `11-20` |\n"
                "| Multiple specific | `42, 44` · `3 and 9 and 15` |\n"
                "| Compare | `compare 3 and 7` · `3 vs 7` |\n"
                "| Explain gap | `why is 5 better than 7` · `difference between 2 and 8` |\n"
                "| **Evaluate new** | `evaluate this candidate:` then paste details |\n\n"
                "For new candidate: paste `title:` · `experience:` · `skills:` · "
                "`evaluation:` · `production:` · `github:`"
            ),
            "show_ranks": [],
            "comparison": None,
            "eval_result": None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLES
# FIX #1: All duplicate CSS rules merged into single, non-conflicting blocks.
# FIX #6: Expander border-bottom on open summary removed (was double-line).
# FIX #9: Tab content bg changed to white for better contrast with card borders.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap');

/* ── Reset & base ─────────────────────────────────────── */
*, body, .stApp,
[data-testid="stAppViewContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  box-sizing: border-box;
}
.stApp { background: #f0f3ff !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── Top header bar ───────────────────────────────────── */
header[data-testid="stHeader"] {
  background: white !important;
  border-bottom: 1px solid #dce2f3 !important;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
}

/* ══ SIDEBAR ══════════════════════════════════════════════
   FIX: All sidebar rules merged — no duplicate selectors.
   Single rule per concept; colors unified.
══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: white !important;
  border-right: 1px solid #dce2f3 !important;
}
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* Section headings (h2 inside sidebar) */
[data-testid="stSidebar"] h2 {
  font-size: 11px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: #6b7280 !important;
  margin: 18px 0 8px !important;
  padding: 0 1rem !important;
}

/* Widget labels — unified: was defined twice with different colors */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.09em !important;
  color: #9ca3af !important;
  margin-bottom: 3px !important;
}

/* Body text */
[data-testid="stSidebar"] [data-testid="stText"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > p {
  font-size: 12px !important;
  color: #6b7280 !important;
}

/* Divider — unified: was defined twice */
[data-testid="stSidebar"] hr {
  border: none !important;
  border-top: 1px solid #e9edf6 !important;
  margin: 12px 0 !important;
}

/* Radio text */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] label,
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] label span:last-child {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: #1e293b !important;
}

/* Checkbox text */
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stCheckbox label span {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: #1e293b !important;
}

/* ══ TABS ═════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  background: white !important;
  border-bottom: 1px solid #e0e5f2 !important;
  gap: 0 !important;
  padding: 0 24px !important;
  border-radius: 0 !important;
  box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
  margin-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: #94a3b8 !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 14px 20px !important;
  border-radius: 0 !important;
  border: none !important;
  border-bottom: 2.5px solid transparent !important;
  margin-bottom: -1px !important;
  transition: color 0.18s !important;
}
.stTabs [aria-selected="true"][role="tab"] {
  color: #3525cd !important;
  border-bottom-color: #3525cd !important;
  font-weight: 700 !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* FIX #9: Tab content background → white so card borders (#e0e5f2) meet WCAG AA */
.stTabs [data-testid="stTabsContent"] {
  padding: 24px 20px !important;
  background: white !important;
}

/* ══ EXPANDER — candidate cards ════════════════════════════
   FIX #1 + #6: Removed orphaned "Method 2" comment.
   Removed `details[open]>summary border-bottom` — it caused a double separator
   line since `details>div` already has its own top padding.
   Removed conflicting catch-all `summary>*:not(svg)` — `summary p` handles text.
══════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
  background: white !important;
  border: 1px solid #e0e5f2 !important;
  border-radius: 14px !important;
  margin-bottom: 10px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
  overflow: hidden !important;
  transition: box-shadow 0.2s, border-color 0.2s !important;
}
[data-testid="stExpander"]:hover {
  border-color: #a5b4fc !important;
  box-shadow: 0 6px 22px rgba(79,70,229,0.11) !important;
}
[data-testid="stExpander"] details > summary {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  padding: 14px 20px !important;
  background: white !important;
  cursor: pointer !important;
  gap: 10px !important;
  list-style: none !important;
}
[data-testid="stExpander"] details > summary::-webkit-details-marker { display: none !important; }

/* FIX #6: Removed border-bottom on open summary; the card content's own
   padding creates sufficient visual separation. */
[data-testid="stExpander"] details[open] > summary {
  background: #f8f9ff !important;
}

/* Title text */
[data-testid="stExpander"] summary p {
  flex: 1 1 0% !important;
  min-width: 0 !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
  font-size: 13.5px !important;
  font-weight: 600 !important;
  color: #1e293b !important;
  line-height: 1.4 !important;
  margin: 0 !important;
}
[data-testid="stExpander"] details[open] summary p { white-space: normal !important; }

/* Hide Material Icons text glyph without breaking SVG chevron */
[data-testid="stExpander"] summary .material-icons,
[data-testid="stExpander"] summary .material-icons-sharp,
[data-testid="stExpander"] summary .material-icons-outlined,
[data-testid="stExpander"] summary [class*="material"],
[data-testid="stExpander"] summary [class*="Icon"] {
  font-size: 0 !important;
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary svg {
  flex-shrink: 0 !important;
  color: #94a3b8 !important;
  width: 20px !important;
  height: 20px !important;
}
[data-testid="stExpander"] details > div {
  padding: 18px 20px !important;
  background: white !important;
}

/* ══ METRIC CARDS ══════════════════════════════════════════ */
[data-testid="stMetric"] {
  background: white !important;
  border: 1px solid #e0e5f2 !important;
  border-radius: 14px !important;
  padding: 18px 20px !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricLabel"] p {
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  color: #9ca3af !important;
}
[data-testid="stMetricValue"] {
  font-size: 26px !important;
  font-weight: 800 !important;
  color: #111827 !important;
  letter-spacing: -0.03em !important;
}

/* ══ BUTTONS ═══════════════════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, #3525cd 0%, #5b50f0 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 9px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 10px 20px !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 2px 8px rgba(53,37,205,0.28) !important;
  transition: all 0.18s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #2a1cb0 0%, #4f46e5 100%) !important;
  box-shadow: 0 4px 16px rgba(53,37,205,0.38) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:focus {
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,0.28) !important;
}
[data-testid="stFormSubmitButton"] > button {
  width: 100% !important;
  background: linear-gradient(135deg, #3525cd 0%, #5b50f0 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 13px 20px !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  box-shadow: 0 3px 12px rgba(53,37,205,0.32) !important;
  transition: all 0.18s ease !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
  background: linear-gradient(135deg, #2a1cb0 0%, #4338ca 100%) !important;
  box-shadow: 0 5px 18px rgba(53,37,205,0.42) !important;
  transform: translateY(-1px) !important;
}

/* ══ INPUTS ════════════════════════════════════════════════ */
.stTextInput input, .stNumberInput input {
  border-radius: 9px !important;
  border: 1.5px solid #dde2ef !important;
  font-size: 13px !important;
  color: #111827 !important;
  background: white !important;
  padding: 10px 13px !important;
  height: 42px !important;
  transition: border-color 0.18s, box-shadow 0.18s !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: #4f46e5 !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,0.14) !important;
  outline: none !important;
}
.stTextArea textarea {
  border-radius: 9px !important;
  border: 1.5px solid #dde2ef !important;
  font-size: 13px !important;
  color: #111827 !important;
  background: white !important;
  padding: 10px 13px !important;
  transition: border-color 0.18s, box-shadow 0.18s !important;
}
.stTextArea textarea:focus {
  border-color: #4f46e5 !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,0.14) !important;
}

/* Widget labels (main area — not sidebar) */
.stTextInput label, .stNumberInput label, .stTextArea label,
.stSelectbox label, .stMultiSelect label {
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.09em !important;
  color: #9ca3af !important;
  margin-bottom: 4px !important;
}

/* ══ SELECT / MULTISELECT ══════════════════════════════════ */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
  border-radius: 9px !important;
  border: 1.5px solid #dde2ef !important;
  background: white !important;
  font-size: 13px !important;
  color: #111827 !important;
  min-height: 42px !important;
}
.stMultiSelect [data-baseweb="tag"] {
  background: #ede9fe !important;
  color: #4c1d95 !important;
  border-radius: 9999px !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  border: none !important;
  padding: 2px 12px !important;
}

/* Radio & Checkbox (main area) */
.stRadio [data-baseweb="radio"] label { font-size: 13px !important; font-weight: 500 !important; color: #374151 !important; }
.stCheckbox label { font-size: 13px !important; font-weight: 500 !important; color: #374151 !important; }

/* ══ CHAT ══════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
  background: white !important;
  border: 1px solid #dce2f3 !important;
  border-radius: 14px !important;
  margin-bottom: 10px !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
  padding: 14px 18px !important;
}
[data-testid="stChatInput"] {
  background: white !important;
  border: 1px solid #dce2f3 !important;
  border-radius: 12px !important;
  padding: 2px !important;
}
[data-testid="stChatInput"] > div {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
  border-radius: 10px !important;
  border: 1px solid #d1d5db !important;
  font-size: 13px !important;
  background: white !important;
  padding: 12px 16px !important;
  color: #111827 !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: #4f46e5 !important;
  box-shadow: 0 0 0 3px rgba(79,70,229,0.15) !important;
}

/* ══ DATAFRAME ═════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
  border-radius: 10px !important;
  border: 1px solid #dce2f3 !important;
  overflow: hidden !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}

/* ══ CODE ══════════════════════════════════════════════════ */
code:not(pre > code) {
  background: #ede9fe !important;
  color: #4c1d95 !important;
  border-radius: 5px !important;
  padding: 2px 8px !important;
  font-size: 12px !important;
}
.stCodeBlock pre {
  background: #f5f3ff !important;
  border: 1px solid #ddd6fe !important;
  border-radius: 10px !important;
}

/* ══ TYPOGRAPHY ════════════════════════════════════════════ */
h1 { font-size: 22px !important; font-weight: 800 !important; color: #111827 !important; letter-spacing: -0.03em !important; line-height: 1.2 !important; }
h2 { font-size: 18px !important; font-weight: 700 !important; color: #1e293b !important; letter-spacing: -0.01em !important; }
h3 { font-size: 15px !important; font-weight: 700 !important; color: #1e293b !important; margin-top: 0 !important; margin-bottom: 16px !important; }

/* ══ MISC ══════════════════════════════════════════════════ */
hr { border-color: #e2e8f8 !important; margin: 16px 0 !important; opacity: 1 !important; }
[data-testid="stCaptionContainer"] p { font-size: 12px !important; color: #9ca3af !important; line-height: 1.6 !important; }
[data-testid="stPyplotRootElement"] {
  background: white !important;
  border: 1px solid #dce2f3 !important;
  border-radius: 14px !important;
  padding: 18px !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}
[data-testid="stAlert"] { border-radius: 10px !important; font-size: 13px !important; }
.stSlider label, .stSlider [data-testid="stWidgetLabel"],
.stSlider [data-testid="stWidgetLabel"] p {
  color: #4b5563 !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
}
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f0f3ff; }
::-webkit-scrollbar-thumb { background: #c4b5fd; border-radius: 9999px; }
::-webkit-scrollbar-thumb:hover { background: #a78bfa; }
</style>
""", unsafe_allow_html=True)

# ── Top header ──────────────────────────────────────────────────────────────────
st.markdown(
    """<div style="display:flex;align-items:center;gap:14px;background:white;
        padding:15px 28px;border-bottom:1px solid #dce2f3;
        box-shadow:0 1px 4px rgba(0,0,0,0.05);margin-bottom:0">
        <div style="background:linear-gradient(135deg,#3525cd 0%,#4f46e5 100%);
            border-radius:10px;width:38px;height:38px;flex-shrink:0;
            display:flex;align-items:center;justify-content:center;font-size:20px">🎯</div>
        <div>
            <div style="font-weight:700;font-size:17px;color:#151c27;
                letter-spacing:-0.02em;line-height:1.2">REDRO AI</div>
            <div style="font-size:11px;color:#9ca3af;font-weight:500;margin-top:1px">
                Evidence-first recruiter decision engine &nbsp;·&nbsp;
                <a href="https://github.com/abhi-7-7/India_run_hackathon"
                   style="color:#4f46e5;text-decoration:none" target="_blank">GitHub ↗</a>
            </div>
        </div>
    </div>""",
    unsafe_allow_html=True
)

# ── Load all resources ───────────────────────────────────────────────────────────
features_df  = load_features()
submission   = load_submission()
cands_lookup = load_candidates_lookup()
faiss_index, faiss_ids = load_faiss_index()
thresholds   = load_scoring_thresholds()

if features_df is None:
    st.error("⚠️  `outputs/features_df.pkl` not found. Run `python rank.py` first.")
    st.stop()

# FIX #7: Warn if candidate lookup is empty
if not cands_lookup:
    st.warning("⚠️  `raw_dataset/candidates.jsonl` not found. "
               "Candidate cards will show placeholder data (scores and signals still work).")

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="background:linear-gradient(135deg,#3525cd 0%,#4f46e5 100%);
            padding:16px;margin:-1rem -1rem 1rem -1rem;
            display:flex;align-items:center;gap:10px">
            <div style="background:rgba(255,255,255,0.18);border-radius:8px;
                width:36px;height:36px;display:flex;align-items:center;
                justify-content:center;font-size:18px;flex-shrink:0">🎯</div>
            <div>
                <div style="color:white;font-weight:700;font-size:14px;
                    letter-spacing:-0.01em;line-height:1.2">Recruitment Pro</div>
                <div style="color:rgba(255,255,255,0.62);font-size:11px;
                    font-weight:500;margin-top:2px">Enterprise Tier</div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='font-size:10px;font-weight:700;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:0.1em;"
        "border-bottom:1px solid #e9edf6;padding-bottom:6px;margin-bottom:10px'>"
        "⚙️ &nbsp;Configuration</div>",
        unsafe_allow_html=True
    )

    mode = st.radio(
        "Ranking Mode",
        ["📋 Default JD (Submission)", "✏️  Custom JD (Explore)"],
        help="Default mode shows the exact submission. Custom mode re-ranks live."
    )

    st.divider()

    if "Custom" in mode:
        jd_text   = st.text_area("Job Description", value=DEFAULT_JD, height=220)
        st.markdown("**Display**")
        n_results = st.slider("Candidates to display", 5, 100, 10)
        show_bd   = st.checkbox("Show score breakdown", value=True)
        run_btn   = st.button("🚀 Rank Now", type="primary", use_container_width=True)
        disp_type = "Top N"
        rank_range = None
        specific_ranks_list = []

    else:
        st.markdown("**Display Mode**")
        disp_type = st.radio(
            "Pick candidates by",
            ["Top N", "Range", "Specific Ranks"],
            label_visibility="collapsed",
        )

        if disp_type == "Top N":
            n_results = st.slider("Top N candidates", 1, 100, 10)
            rank_range = None
            specific_ranks_list = []

        elif disp_type == "Range":
            c1, c2 = st.columns(2)
            r_start = int(c1.number_input("From", min_value=1, max_value=100, value=1,  step=1))
            r_end   = int(c2.number_input("To",   min_value=1, max_value=100, value=20, step=1))
            if r_start > r_end:
                r_start, r_end = r_end, r_start
            rank_range = (r_start, r_end)
            n_results  = r_end - r_start + 1
            specific_ranks_list = []

        else:
            spec_inp = st.text_input("Ranks (e.g. 42, 44, 67)", value="1, 5, 10")
            specific_ranks_list = []
            for tok in re.split(r'[\s,]+', spec_inp):
                if tok.isdigit() and 1 <= int(tok) <= 100:
                    specific_ranks_list.append(int(tok))
            specific_ranks_list = sorted(set(specific_ranks_list))
            rank_range = None
            n_results  = len(specific_ranks_list)

        show_bd = st.checkbox("Show score breakdown", value=True)
        run_btn = True

    st.divider()
    st.markdown(
        f"""<div style="padding:8px 0">
            <div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;
                letter-spacing:0.1em;margin-bottom:10px;border-bottom:1px solid #e9edf6;
                padding-bottom:6px">Pool Info</div>
            <div style="display:flex;justify-content:space-between;align-items:center;
                margin-bottom:8px">
                <span style="font-size:12px;color:#9ca3af">Candidates</span>
                <span style="font-size:12px;font-weight:700;color:#111827;
                    background:#f0f3ff;border-radius:6px;padding:2px 8px">{len(features_df):,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;
                margin-bottom:8px">
                <span style="font-size:12px;color:#9ca3af">Formula</span>
                <span style="font-size:12px;font-weight:700;color:#3525cd;
                    background:#eef2ff;border-radius:6px;padding:2px 8px">NB03 4-engine</span>
            </div>
        </div>""",
        unsafe_allow_html=True
    )
    if "Custom" in mode:
        st.markdown(
            """<div style="padding:0 0.25rem">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:12px">
                    <span style="color:#6b7280">Model</span>
                    <span style="font-weight:600;color:#151c27">MiniLM-L6</span>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:12px">
                    <span style="color:#6b7280">Sample</span>
                    <span style="font-weight:600;color:#151c27">3,000</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )

# ── Tabs ─────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Top Candidates", "Score Analysis", "Pool Insights", "Chat", "Evaluate"
])

# ── TAB 1: Top Candidates ────────────────────────────────────────────────────────
with tab1:
    if "Default" in mode:
        if submission is None:
            st.warning("submission.csv not found in outputs/. Run `python rank.py` first.")
        else:
            sub_with_feats = submission.merge(
                features_df[["candidate_id","experience_years","evaluation_signal_score",
                              "production_signal_score","retrieval_score","semantic_percentile",
                              "recruiter_response_rate","availability_score","consulting_ratio",
                              "days_since_active","notice_period","is_honeypot","expert_ai_skills"]],
                on="candidate_id", how="left"
            )

            if disp_type == "Top N":
                rows_to_show = sub_with_feats.head(n_results)
                label = f"top {n_results}"
            elif disp_type == "Range":
                rows_to_show = sub_with_feats[
                    (sub_with_feats["rank"] >= rank_range[0]) &
                    (sub_with_feats["rank"] <= rank_range[1])
                ]
                label = f"ranks {rank_range[0]}–{rank_range[1]}"
            else:
                if not specific_ranks_list:
                    st.warning("Enter at least one valid rank (1–100).")
                    rows_to_show = pd.DataFrame()
                    label = "none"
                else:
                    rows_to_show = sub_with_feats[
                        sub_with_feats["rank"].isin(specific_ranks_list)
                    ].sort_values("rank")
                    label = "ranks " + ", ".join(str(r) for r in specific_ranks_list)

            if not rows_to_show.empty:
                st.markdown(
                    f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;"
                    f"padding:7px 14px;margin-bottom:16px;font-size:12px;color:#15803d'>"
                    f"✅ &nbsp;Showing <strong>exact submission</strong> — "
                    f"<code style='background:#dcfce7;color:#15803d;font-size:11px'>REDRO_AI.csv</code>"
                    f" &nbsp;·&nbsp; {label}</div>",
                    unsafe_allow_html=True
                )
                for _, row in rows_to_show.iterrows():
                    render_candidate_card(int(row["rank"]), row, cands_lookup, show_bd)

    else:  # Custom JD mode
        if not run_btn:
            st.info("👈  Edit the Job Description in the sidebar and click **Rank Now**.")
        else:
            model = load_model()
            if model is None:
                st.error("sentence-transformers not installed. Run: `pip install sentence-transformers`")
                st.stop()

            # FIX #8: Pass a stable string hash instead of the raw DataFrame
            _feats_hash = hashlib.md5(str(len(features_df)).encode()).hexdigest()

            with st.spinner(f"Loading candidate sample…"):
                sample_ids, sample_texts = get_sample_texts_and_ids(_feats_hash, features_df)

            if not sample_ids:
                st.error("Cannot load candidate texts. Ensure `raw_dataset/candidates.jsonl` exists.")
                st.stop()

            with st.spinner(f"Encoding {len(sample_ids):,} candidates (cached after first run)…"):
                @st.cache_data(show_spinner=False)
                def encode_candidates(texts_tuple):
                    return model.encode(list(texts_tuple), batch_size=64,
                                        show_progress_bar=False, convert_to_numpy=True)
                cand_embs = encode_candidates(tuple(sample_texts))

            with st.spinner("Computing JD similarity…"):
                from sklearn.metrics.pairwise import cosine_similarity as cos_sim
                jd_emb    = model.encode([jd_text], convert_to_numpy=True)
                sims      = cos_sim(jd_emb, cand_embs)[0]

            sample_df = features_df[features_df["candidate_id"].isin(set(sample_ids))].copy()
            sim_series = pd.Series(sims, index=sample_ids, name="new_sim")
            sample_df  = sample_df.join(sim_series.rename("_new_sem"), on="candidate_id")
            new_sem_pct= sample_df["_new_sem"].rank(pct=True)

            scored = apply_nb03_formula(sample_df, semantic_percentile_override=new_sem_pct)
            top_n  = scored.sort_values("final_score", ascending=False).head(n_results).copy()
            top_n["rank"] = range(1, len(top_n) + 1)

            in_range = ((top_n["experience_years"]>=5) & (top_n["experience_years"]<=9)).sum()
            st.info(f"🔍 Custom JD ranking · Sample: {len(sample_ids):,} candidates · "
                    f"Top {n_results} shown · {in_range}/{n_results} in 5-9yr JD range")

            for _, row in top_n.iterrows():
                render_candidate_card(int(row["rank"]), row, cands_lookup, show_bd)

# ── TAB 2: Score Analysis ────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Score Distributions — Submitted Top 100")

    if submission is None:
        st.warning("Run `python rank.py` first to generate submission.csv")
    else:
        sub_feats = submission.merge(
            features_df[["candidate_id","experience_years","evaluation_signal_score",
                          "production_signal_score","semantic_percentile",
                          "retrieval_score","consulting_ratio","availability_score"]],
            on="candidate_id", how="left"
        )

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.patch.set_facecolor("white")
        for ax in axes.flatten():
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#e2e8f8")
            ax.spines["bottom"].set_color("#e2e8f8")
            ax.tick_params(colors="#6b7280", labelsize=9)
        axes = axes.flatten()

        axes[0].hist(sub_feats["score"], bins=15, color="#4f46e5", edgecolor="white", alpha=0.9)
        axes[0].axvline(sub_feats["score"].median(), color="#ef4444", linestyle="--",
                        label=f"Median {sub_feats['score'].median():.3f}")
        axes[0].set_title("Final Score Distribution", fontweight="bold")
        axes[0].set_xlabel("Score"); axes[0].legend()

        axes[1].hist(sub_feats["experience_years"], bins=12, color="#4f46e5", edgecolor="white", alpha=0.9)
        axes[1].axvspan(5, 9, alpha=0.15, color="green", label="JD target (5-9yr)")
        axes[1].set_title("Experience Distribution", fontweight="bold")
        axes[1].set_xlabel("Years"); axes[1].legend()

        axes[2].scatter(sub_feats["semantic_percentile"], sub_feats["score"],
                        alpha=0.6, color="#4f46e5", s=20)
        axes[2].set_title("Semantic Align vs Final Score", fontweight="bold")
        axes[2].set_xlabel("Semantic Percentile"); axes[2].set_ylabel("Score")

        has_eval = (sub_feats["evaluation_signal_score"] > 0).sum()
        axes[3].bar(["No eval signal","Has eval signal"],
                    [100-has_eval, has_eval], color=["#ef4444","#16a34a"], alpha=0.9)
        axes[3].set_title("Evaluation Signal in Top 100", fontweight="bold")
        axes[3].set_ylabel("# Candidates")
        for i, v in enumerate([100-has_eval, has_eval]):
            axes[3].text(i, v+0.5, f"{v}", ha="center", fontweight="bold")

        axes[4].hist(sub_feats["retrieval_score"], bins=10, color="#4f46e5", edgecolor="white", alpha=0.9)
        axes[4].set_title("Retrieval Score Distribution", fontweight="bold")
        axes[4].set_xlabel("Retrieval Score (raw)")

        sr = sub_feats.sort_values("rank") if "rank" in sub_feats else sub_feats.sort_values("score", ascending=False)
        axes[5].plot(range(1, len(sr)+1), sr["score"].values, color="#4f46e5", linewidth=2)
        axes[5].fill_between(range(1, len(sr)+1), sr["score"].values, alpha=0.15, color="#4f46e5")
        axes[5].set_title("Score by Rank", fontweight="bold")
        axes[5].set_xlabel("Rank"); axes[5].set_ylabel("Score")
        axes[5].axvline(10, color="red", linestyle="--", alpha=0.5, label="Rank 10 (NDCG@10)")
        axes[5].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Score", f"{sub_feats['score'].mean():.4f}")
        col2.metric("In JD Range (5-9yr)", f"{((sub_feats['experience_years']>=5)&(sub_feats['experience_years']<=9)).sum()}/100")
        col3.metric("With Eval Evidence", f"{has_eval}/100")
        col4.metric("Score Gap (R1–R100)",
                    f"{sub_feats['score'].max()-sub_feats['score'].min():.4f}")

# ── TAB 3: Pool Insights ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 100,000 Candidate Pool — Business Intelligence")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pool",          f"{len(features_df):,}")
    col2.metric("With Retrieval Skills",f"{(features_df['retrieval_score']>0).sum():,}")
    col3.metric("With Eval Evidence",  f"{(features_df['evaluation_signal_score']>0).sum():,}")
    col4.metric("Avg Experience",      f"{features_df['experience_years'].mean():.1f}yr")

    # FIX #2: Removed the duplicate `fig2, axes2 = plt.subplots(...)` call.
    # Only one figure is created; facecolor + axis style set in the same pass.
    n_pool   = len(features_df)
    no_eval  = (features_df["evaluation_signal_score"] == 0).sum()
    low_eval = ((features_df["evaluation_signal_score"]>0) & (features_df["evaluation_signal_score"]<=0.4)).sum()
    hi_eval  = (features_df["evaluation_signal_score"] > 0.4).sum()

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    fig2.patch.set_facecolor("white")
    for ax2 in axes2:
        ax2.set_facecolor("white")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color("#e2e8f8")
        ax2.spines["bottom"].set_color("#e2e8f8")
        ax2.tick_params(colors="#6b7280", labelsize=9)

    axes2[0].bar(["Zero","Some","Strong"], [no_eval, low_eval, hi_eval],
                 color=["#ef4444","#f59e0b","#16a34a"], alpha=0.9, edgecolor="white")
    axes2[0].set_title("Evaluation Signal\nAcross 100k Pool", fontweight="bold")
    axes2[0].set_ylabel("# Candidates")
    for i, v in enumerate([no_eval, low_eval, hi_eval]):
        axes2[0].text(i, v+200, f"{100*v/n_pool:.1f}%", ha="center", fontsize=9, fontweight="bold")

    axes2[1].hist(features_df["experience_years"], bins=20, color="#4f46e5",
                  edgecolor="white", alpha=0.9, density=True)
    axes2[1].axvspan(5, 9, alpha=0.15, color="green", label="JD target")
    axes2[1].set_title("Experience Distribution\n(Full Pool)", fontweight="bold")
    axes2[1].set_xlabel("Years"); axes2[1].legend()

    inactive_90  = (features_df["days_since_active"] > 90).sum()
    inactive_180 = (features_df["days_since_active"] > 180).sum()
    active_30    = (features_df["days_since_active"] <= 30).sum()
    axes2[2].bar(["Active\n(≤30d)","Inactive\n(90-180d)","Stale\n(>180d)"],
                 [active_30, inactive_90-inactive_180, inactive_180],
                 color=["#16a34a","#f59e0b","#ef4444"], alpha=0.9, edgecolor="white")
    axes2[2].set_title("Candidate Recency\n(Full Pool)", fontweight="bold")
    axes2[2].set_ylabel("# Candidates")
    axes2[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig2.suptitle("What Does the 100k Pool Actually Look Like?", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    with st.expander("📐 Ablation Study Results (from Notebook 04)"):
        ablation_data = {
            "Component Removed"    : ["Baseline","Experience Fit","Evaluation Signal",
                                       "Validation (Behavioral)","Production Signal","Risk Filter"],
            "Top-100 Overlap"      : [100, 69, 83, 89, 91, 94],
            "Interpretation"       : ["Full model","Critical","High","High","Moderate","Gate (correct)"],
        }
        st.dataframe(pd.DataFrame(ablation_data), width="stretch", hide_index=True)
        st.caption("Shows which features actually drive the ranking. 69% overlap for Experience Fit "
                   "means removing it moves 31 candidates out of the top-100.")

# ── TAB 4: Chat ───────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 💬 Candidate Intelligence Chat")
    st.caption(
        "Try: &nbsp;`rank 5` &nbsp;·&nbsp; `11 to 20` &nbsp;·&nbsp; "
        "`42, 44` &nbsp;·&nbsp; `compare 3 and 7` &nbsp;·&nbsp; `why is 5 better than 7` &nbsp;·&nbsp; "
        "`evaluate this candidate:` *(paste details)*"
    )

    with st.expander("📋 How to evaluate a new candidate in chat — copy this template"):
        st.code("""evaluate this candidate:
title: Senior ML Engineer
experience: 6
skills: FAISS, Embeddings, BM25, Elasticsearch, Pinecone
evaluation: improved NDCG@10 by 14% in career history
production: deployed recommendation system serving 2M users
location: Pune
consulting: 0
notice: 30
github: 75""", language=None)
        st.caption("💡 Tip: Use the **🔍 Evaluate** tab for a guided form — no typing format needed.")

    if submission is None:
        st.warning("Run `python rank.py` first to enable chat.")
    else:
        _CHAT_COLS = [c for c in [
            "candidate_id", "experience_years", "evaluation_signal_score",
            "production_signal_score", "retrieval_score", "semantic_percentile",
            "recruiter_response_rate", "availability_score", "consulting_ratio",
            "days_since_active", "notice_period", "is_honeypot", "expert_ai_skills",
            "quality_score_log", "avg_ai_assessment_score", "career_keyword_score",
        ] if c in features_df.columns]

        _chat_df = (
            submission
            .merge(features_df[_CHAT_COLS], on="candidate_id", how="left")
            .sort_values("rank")
            .reset_index(drop=True)
        )

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # FIX #4: Render all past messages first, then handle new input.
        # st.rerun() is only called when there's a new message to append —
        # it's still needed for Streamlit's chat pattern but now clearly scoped.
        for _msg in st.session_state.chat_messages:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])
                for _rank_val in _msg.get("show_ranks", []):
                    _r_row = _chat_df[_chat_df["rank"] == _rank_val]
                    if not _r_row.empty:
                        render_candidate_card(_rank_val, _r_row.iloc[0], cands_lookup, True)
                if _msg.get("comparison"):
                    st.dataframe(pd.DataFrame(_msg["comparison"]), use_container_width=True, hide_index=True)
                if _msg.get("eval_result"):
                    _render_eval_result(_msg["eval_result"], _chat_df, cands_lookup)

        if _prompt := st.chat_input("Ask about candidates or paste new candidate details…"):
            st.session_state.chat_messages.append({
                "role": "user", "content": _prompt,
                "show_ranks": [], "comparison": None, "eval_result": None,
            })
            _model = load_model() if (faiss_index is not None) else None
            _response = _chat_parse_and_respond(
                _prompt, _chat_df, cands_lookup,
                faiss_index=faiss_index, faiss_ids=faiss_ids,
                features_df=features_df, thresholds=thresholds, model=_model,
            )
            st.session_state.chat_messages.append(_response)
            st.rerun()

        if st.session_state.chat_messages:
            if st.button("🗑️  Clear chat", key="clear_chat"):
                st.session_state.chat_messages = []
                st.rerun()

# ── TAB 5: Evaluate ───────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### 🔍 Evaluate a New Candidate Against the JD")
    st.caption("Fill in the form — no specific text format needed. Get an instant JD fit verdict.")

    if submission is None:
        st.warning("Run `python rank.py` first.")
    else:
        _EVAL_CHAT_COLS = [c for c in [
            "candidate_id","experience_years","evaluation_signal_score",
            "production_signal_score","retrieval_score","semantic_percentile",
            "recruiter_response_rate","availability_score","consulting_ratio",
        ] if c in features_df.columns]
        _eval_chat_df = submission.merge(
            features_df[_EVAL_CHAT_COLS], on="candidate_id", how="left"
        ).sort_values("rank").reset_index(drop=True)

        with st.form("candidate_eval_form", clear_on_submit=False):

            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;"
                "border-bottom:2px solid #e9edf6;padding-bottom:12px;margin-bottom:20px'>"
                "<span style='font-size:18px'>👤</span>"
                "<span style='font-size:12px;font-weight:700;color:#1e293b;"
                "text-transform:uppercase;letter-spacing:0.1em'>"
                "Candidate Details</span></div>",
                unsafe_allow_html=True
            )
            ca, cb = st.columns(2)
            with ca:
                ev_title  = st.text_input("Current Title *", placeholder="Senior ML Engineer")
                ev_loc    = st.text_input("Location (City)", placeholder="Pune, Bangalore…")
            with cb:
                ev_exp    = st.number_input("Years of Experience *", min_value=0.0,
                                            max_value=30.0, value=5.0, step=0.5)
                ev_notice = st.number_input("Notice Period (days)", min_value=0,
                                            max_value=180, value=30, step=15)

            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;"
                "border-bottom:2px solid #e9edf6;padding-bottom:12px;"
                "margin:28px 0 20px'>"
                "<span style='font-size:18px'>📈</span>"
                "<span style='font-size:12px;font-weight:700;color:#1e293b;"
                "text-transform:uppercase;letter-spacing:0.1em'>"
                "Profile Signals</span></div>",
                unsafe_allow_html=True
            )
            cs1, cs2 = st.columns(2)
            with cs1:
                ev_github = st.slider("GitHub Activity  (-1 = not linked)",
                                      min_value=-1, max_value=100, value=-1)
            with cs2:
                ev_consulting = st.slider("Consulting Background (%)",
                                          min_value=0, max_value=100, value=0)

            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;"
                "border-bottom:2px solid #e9edf6;padding-bottom:12px;"
                "margin:28px 0 20px'>"
                "<span style='font-size:18px'>🛠️</span>"
                "<span style='font-size:12px;font-weight:700;color:#1e293b;"
                "text-transform:uppercase;letter-spacing:0.1em'>"
                "Technical Skills Stack</span></div>",
                unsafe_allow_html=True
            )
            _all_skills = sorted(RETRIEVAL_SKILLS | LLM_SKILLS | ML_SKILLS |
                                 {"Python","Docker","AWS","GCP","Kubernetes"})
            ev_skills = st.multiselect(
                "Select every skill the candidate has listed",
                options=_all_skills, default=[],
            )

            st.markdown(
                "<div style='display:flex;align-items:center;gap:10px;"
                "border-bottom:2px solid #e9edf6;padding-bottom:12px;"
                "margin:28px 0 20px'>"
                "<span style='font-size:18px'>📊</span>"
                "<span style='font-size:12px;font-weight:700;color:#1e293b;"
                "text-transform:uppercase;letter-spacing:0.1em'>"
                "Evidence Signals</span></div>",
                unsafe_allow_html=True
            )
            ce1, ce2 = st.columns(2)
            with ce1:
                st.markdown(
                    "<div style='background:#f0fdf4;border:1px solid #bbf7d0;"
                    "border-radius:8px;padding:10px 13px;margin-bottom:10px'>"
                    "<div style='font-size:12px;font-weight:700;color:#15803d;"
                    "margin-bottom:3px'>📐 Evaluation Metrics</div>"
                    "<div style='font-size:11px;color:#4b5563;line-height:1.5'>"
                    "NDCG, MRR, MAP, A/B tests, or offline eval in career history"
                    "</div></div>", unsafe_allow_html=True
                )
                ev_has_eval = st.checkbox("Has evaluation metrics in career history")
                if ev_has_eval:
                    st.text_input("Detail (optional)",
                                  placeholder="e.g. improved NDCG@10 by 14%",
                                  key="eval_detail")
            with ce2:
                st.markdown(
                    "<div style='background:#eff6ff;border:1px solid #bfdbfe;"
                    "border-radius:8px;padding:10px 13px;margin-bottom:10px'>"
                    "<div style='font-size:12px;font-weight:700;color:#1d4ed8;"
                    "margin-bottom:3px'>🚀 Production Deployment</div>"
                    "<div style='font-size:11px;color:#4b5563;line-height:1.5'>"
                    "Shipped or runs ML systems in production at scale"
                    "</div></div>", unsafe_allow_html=True
                )
                ev_has_prod = st.checkbox("Has deployed ML systems to production")
                if ev_has_prod:
                    st.text_input("Detail (optional)",
                                  placeholder="e.g. recommendation system, 2M users",
                                  key="prod_detail")

            st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                "🚀  Submit Full Evaluation", type="primary", use_container_width=True
            )

        if submitted:
            if not ev_title and not ev_skills:
                st.warning("Enter at least a title or some skills to evaluate.")
            else:
                _parsed = {
                    "title"           : ev_title,
                    "experience"      : float(ev_exp),
                    "skills"          : list(ev_skills),
                    "has_eval_signal" : ev_has_eval,
                    "has_production"  : ev_has_prod,
                    "location"        : ev_loc.lower(),
                    "consulting_ratio": ev_consulting / 100.0,
                    "notice_days"     : int(ev_notice),
                    "github_score"    : float(ev_github),
                }
                _eval_model = load_model() if faiss_index is not None else None
                _result = _score_new_candidate(
                    _parsed, thresholds, faiss_index, faiss_ids, features_df, _eval_model
                )
                st.divider()
                _render_eval_result(_result, _eval_chat_df, cands_lookup)