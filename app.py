"""
REDRO AI — Candidate Ranking Demo
===================================
Run: streamlit run app.py

Two modes:
  Default JD  — shows precomputed submission.csv results (exact match)
  Custom JD   — live ranking using same sentence-transformers + NB03 formula
"""

import json, os, math, re, warnings
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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
    """Notebooks always write to Notebook/outputs/ — look there only."""
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "Notebook", "outputs", filename)
    return path if os.path.exists(path) else None

@st.cache_data(show_spinner="Loading feature table…")
def load_features():
    path = _find_output("features_df.pkl")
    if path:
        try:
            return pd.read_pickle(path)
        except (ModuleNotFoundError, ImportError, Exception):
            # pkl saved with different numpy/pandas version — use CSV fallback
            pass
    path = _find_output("features_df.csv")
    if path:
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner="Loading submission…")
def load_submission():
    path = _find_output("submission.csv")
    return pd.read_csv(path) if path else None


@st.cache_data(show_spinner="Loading candidate profiles…")
def load_candidates_lookup(n_read=20000):
    """Load a large sample; we only need profiles for display."""
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


@st.cache_data(show_spinner="Pre-computing candidate embeddings for custom JD mode…")
def get_sample_texts_and_ids(features_df, n=3000):
    """
    Select a stratified sample for custom-JD ranking.
    Uses features_df to pick candidates with best non-semantic signals.
    Returns (list of candidate_ids, list of texts) — candidate texts built from profiles.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "raw_dataset", "candidates.jsonl")
    if not os.path.exists(path):
        return [], []

    # Stratify: top retrieval + top evaluation + rest
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

    # Load texts for selected candidates
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
    """
    Applies NB03 scoring formula exactly.
    If semantic_percentile_override is provided (for custom JD), replaces
    the precomputed semantic_percentile column.
    """
    ndf = feat_df.copy()

    # Override semantic percentile for custom JD
    if semantic_percentile_override is not None:
        ndf["semantic_percentile"] = semantic_percentile_override

    # Normalise via rank(pct=True) — mirrors NB03 Phase 2
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

    # Derived features (NB03 Phase 2)
    ndf["eval_combo"]  = (0.6 * ndf["evaluation_signal_score_pct"] +
                          0.4 * (ndf["evaluation_signal_score"] > 0).astype(float))
    ndf["sem_capped"]  = ndf["semantic_percentile"].rank(pct=True).clip(upper=0.97)
    ndf["avail_pct"]   = ndf["availability_score"].rank(pct=True, method="average")
    ndf["saved_pct"]   = ndf["saved_by_recruiters_norm"].rank(pct=True, method="average")
    ndf["views_pct"]   = ndf["profile_views_norm"].rank(pct=True, method="average")

    # Capability score (NB03 Phase 3)
    ndf["capability_score"] = sum(ndf[col] * w for col, w in CAP_WEIGHTS.items())

    # Validation score (NB03 Phase 4)
    ndf["validation_score"] = (
        0.40 * ndf["saved_pct"] +
        0.30 * ndf["recruiter_response_rate"] +
        0.20 * ndf["interview_completion_rate"] +
        0.10 * ndf["views_pct"]
    )

    # Base score (NB03 Phase 7)
    ndf["base_score"] = (
        0.60 * ndf["capability_score"] +
        0.25 * ndf["validation_score"] +
        0.15 * ndf["avail_pct"]
    )

    # Multipliers (NB03 Phase 5 + 6)
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

    tier_color = "#27ae60" if rank <= 10 else ("#2980b9" if rank <= 30 else "#e67e22")

    with st.expander(f"**#{rank}** — {title}  ·  {exp:.0f}yr  ·  {loc}  ·  Score: **{score:.4f}**",
                     expanded=(rank <= 5)):
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Candidate:** `{cid}`")
            if jd_sk:
                st.markdown("**JD-relevant skills:** " + "  ".join(f"`{s}`" for s in jd_sk))
            else:
                st.markdown("*No direct JD skill matches in profile*")
            st.markdown(f"**Reasoning:** {reason}")

        if show_breakdown and col2:
            with col2:
                st.markdown("**Score components:**")
                components = {
                    "Semantic align" : min(1.0, float(row.get("semantic_percentile", 0))),
                    "Eval evidence"  : float(row.get("evaluation_signal_score", 0)),
                    "Production sig" : float(row.get("production_signal_score", 0)),
                    "Retrieval"      : min(1.0, float(row.get("retrieval_score", 0)) / 45),
                    "Validation"     : float(row.get("recruiter_response_rate", 0)),
                    "Availability"   : float(row.get("availability_score", 0)),
                }
                for name, val in components.items():
                    color = "#27ae60" if val > 0.6 else ("#f39c12" if val > 0.3 else "#c0392b")
                    pct   = int(val * 100)
                    st.markdown(
                        f"<div style='font-size:0.8rem;margin-bottom:2px'>{name}: {val:.2f}</div>"
                        f"<div style='background:#eee;border-radius:4px;height:8px;margin-bottom:6px'>"
                        f"<div style='width:{pct}%;height:100%;background:{color};border-radius:4px'></div></div>",
                        unsafe_allow_html=True
                    )


# ── Chat helpers ─────────────────────────────────────────────────────────────────
def _safe_float(val, default=0.0):
    try:
        return float(val) if val is not None and not (isinstance(val, float) and math.isnan(val)) else default
    except (TypeError, ValueError):
        return default


def _chat_parse_and_respond(query, chat_df, cands_lookup):
    """Parse a natural-language query and return a response dict."""
    q    = query.lower().strip()
    nums = [int(n) for n in re.findall(r'\b(\d{1,3})(?:st|nd|rd|th)?\b', q) if 1 <= int(n) <= 100]

    is_compare = any(w in q for w in ["compare","vs","versus","why","better","differ","difference"])
    is_range   = bool(re.search(r'\b\d+\s*(?:to|through)\s*\d+\b|\b\d+-\d+\b', query, re.IGNORECASE)) and len(nums) == 2
    has_multi  = ("," in query) or (re.search(r'\band\b', q) and len(nums) > 1)

    # ── Compare two ranked candidates ─────────────────────────────────────────
    if is_compare and len(nums) >= 2:
        r1, r2 = nums[0], nums[1]
        row1 = chat_df[chat_df["rank"] == r1]
        row2 = chat_df[chat_df["rank"] == r2]
        if row1.empty:
            return {"role":"assistant","content":f"Rank {r1} not found in submission.","show_ranks":[],"comparison":None}
        if row2.empty:
            return {"role":"assistant","content":f"Rank {r2} not found in submission.","show_ranks":[],"comparison":None}

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
        return {"role":"assistant","content":text,"show_ranks":[r1,r2],"comparison":comp_rows}

    # ── Range ─────────────────────────────────────────────────────────────────
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
            "comparison": None,
        }

    # ── Specific list ─────────────────────────────────────────────────────────
    elif has_multi and len(nums) > 1 and not is_compare:
        return {
            "role": "assistant",
            "content": f"Showing candidates at rank **{', '.join(str(n) for n in sorted(nums))}**:",
            "show_ranks": sorted(nums),
            "comparison": None,
        }

    # ── Single rank ───────────────────────────────────────────────────────────
    elif len(nums) == 1:
        r   = nums[0]
        row = chat_df[chat_df["rank"] == r]
        if row.empty:
            return {"role":"assistant","content":f"No candidate found at rank {r}.","show_ranks":[],"comparison":None}
        return {"role":"assistant","content":f"Candidate at rank **{r}**:","show_ranks":[r],"comparison":None}

    # ── Help ──────────────────────────────────────────────────────────────────
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
                "| Explain gap | `why is 5 better than 7` · `difference between 2 and 8` |"
            ),
            "show_ranks": [],
            "comparison": None,
        }


# ── MAIN UI ─────────────────────────────────────────────────────────────────────
st.title("🎯 REDRO AI — Candidate Ranking Demo")
st.caption("Evidence-first recruiter decision engine · [github.com/abhi-7-7/India_run_hackathon](https://github.com/abhi-7-7/India_run_hackathon)")

# Load all resources
features_df  = load_features()
submission   = load_submission()
cands_lookup = load_candidates_lookup()

if features_df is None:
    st.error("⚠️  `outputs/features_df.pkl` not found. Run `python rank.py` first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️  Configuration")

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
        run_btn   = st.button("🚀 Rank Now", type="primary", width="stretch")
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

        else:  # Specific Ranks
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
    st.markdown(f"**Pool:** {len(features_df):,} candidates")
    st.markdown("**Formula:** NB03 4-engine architecture")
    if "Custom" in mode:
        st.markdown("**Semantic model:** all-MiniLM-L6-v2")
        st.markdown("**Sample:** 3,000 stratified candidates")
        st.markdown("**Runtime:** ~15 sec (first run, then cached)")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Candidates", "📊 Score Analysis", "💼 Pool Insights", "💬 Chat"])

# ── TAB 1: Top Candidates ────────────────────────────────────────────────────────
with tab1:
    if "Default" in mode:
        # Exact submission results — guaranteed to match REDRO_AI.csv
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
            # ── Filter by display mode ────────────────────────────────────────
            if disp_type == "Top N":
                rows_to_show = sub_with_feats.head(n_results)
                label = f"top {n_results}"
            elif disp_type == "Range":
                rows_to_show = sub_with_feats[
                    (sub_with_feats["rank"] >= rank_range[0]) &
                    (sub_with_feats["rank"] <= rank_range[1])
                ]
                label = f"ranks {rank_range[0]}–{rank_range[1]}"
            else:  # Specific Ranks
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
                st.success(f"✅  Showing **exact submission** — `REDRO_AI.csv` · {label}")
                st.caption("These are identical to the submitted candidates. Demo = Submission.")
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

            with st.spinner("Loading candidate sample…"):
                sample_ids, sample_texts = get_sample_texts_and_ids(features_df)

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

            # Build sample features_df with new semantic similarity
            sample_df = features_df[features_df["candidate_id"].isin(set(sample_ids))].copy()
            sim_series = pd.Series(sims, index=sample_ids, name="new_sim")
            sample_df  = sample_df.join(sim_series.rename("_new_sem"), on="candidate_id")
            new_sem_pct= sample_df["_new_sem"].rank(pct=True)

            # Apply NB03 formula with new semantic scores
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
        axes = axes.flatten()

        # Score distribution
        axes[0].hist(sub_feats["score"], bins=15, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[0].axvline(sub_feats["score"].median(), color="#c0392b", linestyle="--",
                        label=f"Median {sub_feats['score'].median():.3f}")
        axes[0].set_title("Final Score Distribution", fontweight="bold")
        axes[0].set_xlabel("Score"); axes[0].legend()

        # Experience distribution
        axes[1].hist(sub_feats["experience_years"], bins=12, color="#DD8452", edgecolor="white", alpha=0.85)
        axes[1].axvspan(5, 9, alpha=0.15, color="green", label="JD target (5-9yr)")
        axes[1].set_title("Experience Distribution", fontweight="bold")
        axes[1].set_xlabel("Years"); axes[1].legend()

        # Semantic vs Score scatter
        axes[2].scatter(sub_feats["semantic_percentile"], sub_feats["score"],
                        alpha=0.6, color="#4C72B0", s=20)
        axes[2].set_title("Semantic Align vs Final Score", fontweight="bold")
        axes[2].set_xlabel("Semantic Percentile"); axes[2].set_ylabel("Score")

        # Evaluation signal
        has_eval = (sub_feats["evaluation_signal_score"] > 0).sum()
        axes[3].bar(["No eval signal","Has eval signal"],
                    [100-has_eval, has_eval], color=["#c0392b","#27ae60"], alpha=0.85)
        axes[3].set_title("Evaluation Signal in Top 100", fontweight="bold")
        axes[3].set_ylabel("# Candidates")
        for i, v in enumerate([100-has_eval, has_eval]):
            axes[3].text(i, v+0.5, f"{v}", ha="center", fontweight="bold")

        # Retrieval score histogram
        axes[4].hist(sub_feats["retrieval_score"], bins=10, color="#55A868", edgecolor="white", alpha=0.85)
        axes[4].set_title("Retrieval Score Distribution", fontweight="bold")
        axes[4].set_xlabel("Retrieval Score (raw)")

        # Score vs Rank line
        sr = sub_feats.sort_values("rank") if "rank" in sub_feats else sub_feats.sort_values("score", ascending=False)
        axes[5].plot(range(1, len(sr)+1), sr["score"].values, color="#4C72B0", linewidth=2)
        axes[5].fill_between(range(1, len(sr)+1), sr["score"].values, alpha=0.15, color="#4C72B0")
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
    col1.metric("Total Pool",         f"{len(features_df):,}")
    col2.metric("With Retrieval Skills",f"{(features_df['retrieval_score']>0).sum():,}")
    col3.metric("With Eval Evidence", f"{(features_df['evaluation_signal_score']>0).sum():,}")
    col4.metric("Avg Experience",     f"{features_df['experience_years'].mean():.1f}yr")

    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))

    # Eval signal sparsity
    n_pool = len(features_df)
    no_eval  = (features_df["evaluation_signal_score"] == 0).sum()
    low_eval = ((features_df["evaluation_signal_score"]>0) & (features_df["evaluation_signal_score"]<=0.4)).sum()
    hi_eval  = (features_df["evaluation_signal_score"] > 0.4).sum()
    axes2[0].bar(["Zero","Some","Strong"], [no_eval, low_eval, hi_eval],
                 color=["#c0392b","#f39c12","#27ae60"], alpha=0.85, edgecolor="white")
    axes2[0].set_title("Evaluation Signal\nAcross 100k Pool", fontweight="bold")
    axes2[0].set_ylabel("# Candidates")
    for i, v in enumerate([no_eval, low_eval, hi_eval]):
        axes2[0].text(i, v+200, f"{100*v/n_pool:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Experience distribution
    axes2[1].hist(features_df["experience_years"], bins=20, color="#4C72B0",
                  edgecolor="white", alpha=0.85, density=True)
    axes2[1].axvspan(5, 9, alpha=0.15, color="green", label="JD target")
    axes2[1].set_title("Experience Distribution\n(Full Pool)", fontweight="bold")
    axes2[1].set_xlabel("Years"); axes2[1].legend()

    # Availability breakdown
    inactive_90  = (features_df["days_since_active"] > 90).sum()
    inactive_180 = (features_df["days_since_active"] > 180).sum()
    active_30    = (features_df["days_since_active"] <= 30).sum()
    axes2[2].bar(["Active\n(≤30d)","Inactive\n(90-180d)","Stale\n(>180d)"],
                 [active_30, inactive_90-inactive_180, inactive_180],
                 color=["#27ae60","#f39c12","#c0392b"], alpha=0.85, edgecolor="white")
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
        "`42, 44` &nbsp;·&nbsp; `compare 3 and 7` &nbsp;·&nbsp; `why is 5 better than 7`"
    )

    if submission is None:
        st.warning("Run `python rank.py` first to enable chat.")
    else:
        # Build merged dataframe for chat (submission scores + feature signals)
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

        # Session state
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Render chat history
        for _msg in st.session_state.chat_messages:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])
                for _rank_val in _msg.get("show_ranks", []):
                    _r_row = _chat_df[_chat_df["rank"] == _rank_val]
                    if not _r_row.empty:
                        render_candidate_card(_rank_val, _r_row.iloc[0], cands_lookup, True)
                if _msg.get("comparison"):
                    st.dataframe(
                        pd.DataFrame(_msg["comparison"]),
                        width="stretch",
                        hide_index=True,
                    )

        # Chat input
        if _prompt := st.chat_input("Ask about candidates…"):
            st.session_state.chat_messages.append({
                "role": "user", "content": _prompt,
                "show_ranks": [], "comparison": None,
            })
            _response = _chat_parse_and_respond(_prompt, _chat_df, cands_lookup)
            st.session_state.chat_messages.append(_response)
            st.rerun()

        # Clear button
        if st.session_state.chat_messages:
            if st.button("🗑️  Clear chat", key="clear_chat"):
                st.session_state.chat_messages = []
                st.rerun()