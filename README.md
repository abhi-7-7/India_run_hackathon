# REDRO AI – Retrieval & Recommendation Candidate Ranking System

## Overview

This project was developed for the REDRO AI Hiring Hackathon.

The challenge is to identify the strongest candidates for a Retrieval, Search, Ranking, and Recommendation Systems role from a dataset containing over 100,000 candidate profiles.

Instead of relying on simple keyword matching, this solution combines:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Semantic Similarity Modeling
* Recruiter Behavior Signals
* Availability Signals
* Risk Detection
* Multi-factor Candidate Ranking

The final output is a ranked list of the top candidates most relevant to the target job description.

---

# Problem Statement

Traditional candidate search systems often fail because they rely heavily on:

* Exact keyword matching
* Skill counts
* Resume buzzwords

However, strong candidates frequently demonstrate expertise through:

* Career history
* Project descriptions
* Ranking and retrieval experience
* Production deployments
* Evaluation metrics (NDCG, MRR, MAP)
* Recruiter engagement signals

The objective was to build a system that identifies true candidate capability rather than keyword density.

---

# Dataset

Candidate Dataset:

* 100,000 candidate profiles
* Career history
* Skills
* Certifications
* Assessments
* Recruiter behavior signals
* Availability signals

Key fields analyzed:

* Experience
* Current Title
* Skills
* Career History
* Certifications
* Recruiter Signals
* Availability Signals

---

# Project Structure

```text
REDRO_AI/
│
├── Notebook/
│   ├── 01_eda.ipynb
│   ├── 02_fea_eng.ipynb
│   ├── 03_rank.ipynb
│   ├── 01_test.py
│   ├── 02_test.py
│   ├── 03_test.py
│
├── outputs/
│   ├── submission.csv
│   ├── top100_candidates.csv
│   ├── final_ranked_all.csv
│
├── README.md
└── .gitignore
```

---

# Notebook 01 – Exploratory Data Analysis

Purpose:

Understand the structure and quality of the candidate dataset.

Major analyses:

* Candidate distribution
* Experience distribution
* Skill frequency analysis
* Degree analysis
* Certification analysis
* Recruiter signal analysis
* Correlation analysis
* Retrieval candidate discovery
* Behavioral signal exploration

Key findings:

* Retrieval skills are extremely rare
* Evaluation metrics appear primarily in career descriptions
* Recruiter behavior provides valuable validation signals
* Certifications are weak standalone indicators
* Production experience is more predictive than keyword count

---

# Notebook 02 – Feature Engineering

Purpose:

Convert raw candidate information into ranking-ready features.

Generated Features:

### Skill Features

* retrieval_score
* llm_score
* ml_score

### Quality Features

* quality_score
* career_depth_score
* assessment_score

### Semantic Features

* semantic_similarity

### Recruiter Validation Features

* behavior_score

### Availability Features

* availability_score

### Risk Features

* consulting_ratio
* is_honeypot

### Career Evidence Features

* evaluation_signal_score
* production_signal_score
* career_keyword_score

Output:

```text
features_df.pkl
features_df.csv
```

Approximately:

```text
100,000 candidates
48 engineered features
```

---

# Notebook 03 – Ranking Engine

Purpose:

Convert engineered features into recruiter-style ranking decisions.

Ranking Architecture:

## Capability Engine

Measures:

* Retrieval expertise
* Semantic relevance
* Production experience
* Evaluation experience
* Technical depth

## Validation Engine

Measures:

* Recruiter interest
* Profile engagement
* Assessment performance

## Availability Engine

Measures:

* Hiring readiness
* Candidate responsiveness

## Risk Engine

Penalizes:

* Consulting-only profiles
* Honeypot profiles
* Weak experience fit

---

# Final Ranking Formula

Final score combines:

```text
Capability
+
Validation
+
Availability
-
Risk
```

The system emphasizes evidence-based candidate quality rather than keyword matching.

---

# Methodology

Candidate Ranking Pipeline:

```text
100,000 Candidates
        ↓
Notebook
        ↓
Feature Engineering
        ↓
Semantic Similarity
        ↓
Capability Scoring
        ↓
Validation Scoring
        ↓
Availability Scoring
        ↓
Risk Adjustment
        ↓
Final Ranking
        ↓
Top 100 Candidates
```

---

# Results

Generated Outputs:

* features_df.pkl
* features_df.csv
* final_ranked_all.csv
* top100_candidates.csv
* submission.csv

The ranking system successfully identifies candidates with:

* Retrieval experience
* Recommendation system experience
* Ranking expertise
* Search relevance experience
* Production ML deployment experience

while filtering out candidates with weak evidence despite strong keyword overlap.

---

# Technologies Used

Python

Libraries:

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Sentence Transformers
* Scikit-learn

---

# Key Insights

The strongest candidates were not necessarily those with the most AI buzzwords.

High-performing candidates consistently showed:

* Retrieval system ownership
* Ranking experience
* Production deployments
* Evaluation framework familiarity
* Recruiter validation signals

This reinforces the importance of evidence-driven hiring rather than keyword-driven hiring.

---

# Future Improvements

Potential future enhancements:

* Learning-to-Rank models
* Pairwise ranking objectives
* Cross-encoder semantic matching
* Product-company quality scoring
* Recruiter feedback loop integration
* Real-time ranking API

---

# Author

Aarsh Bhatnagar,Prakhar Srivastava

Developed for the REDRO AI Hiring Hackathon.
