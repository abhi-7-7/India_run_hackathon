#!/usr/bin/env python3
"""
Standalone submission validator — checks an already-written CSV without
re-running the full pipeline. Matches the exact checks documented in
README.md's "Submission Validation" table.

Usage:
    python validate_submission.py REDRO_AI.csv
"""
import sys, os
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"❌  File not found: {csv_path}")
        sys.exit(1)

    sub = pd.read_csv(csv_path)
    errors, warnings = [], []

    # ── Structural checks (must pass) ────────────────────────────────────────
    required_cols = {"candidate_id", "rank", "score", "reasoning"}
    missing_cols = required_cols - set(sub.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ❌ {e}")
        sys.exit(1)

    if len(sub) != 100:
        errors.append(f"Row count {len(sub)} != 100")
    if sorted(sub["rank"].tolist()) != list(range(1, 101)):
        errors.append("Ranks are not exactly 1-100")
    if sub["candidate_id"].nunique() != 100:
        errors.append(f"Duplicate candidate_ids ({100 - sub['candidate_id'].nunique()} dupes)")
    diffs = sub.sort_values("rank")["score"].diff().dropna()
    if not (diffs <= 1e-9).all():
        n_bad = (diffs > 1e-9).sum()
        errors.append(f"Scores not non-increasing by rank ({n_bad} violations)")
    empty = (sub["reasoning"].isna() | (sub["reasoning"].astype(str).str.strip() == "")).sum()
    if empty > 0:
        errors.append(f"{empty} rows have empty reasoning")

    # ── Enrichment checks (informational — match README's reported stats) ─────
    root = os.path.dirname(os.path.abspath(csv_path)) or "."
    feat_path = os.path.join(root, "Notebook", "outputs", "features_df.csv")
    honeypot_rate = None
    in_jd_range = None

    if os.path.exists(feat_path):
        feats = pd.read_csv(feat_path, usecols=lambda c: c in
            {"candidate_id", "is_honeypot", "experience_years"})
        merged = sub.merge(feats, on="candidate_id", how="left")

        if "is_honeypot" in merged.columns:
            honeypot_rate = merged["is_honeypot"].fillna(0).mean()
            if honeypot_rate >= 0.10:
                errors.append(f"Honeypot rate {honeypot_rate:.1%} >= 10% threshold")

        if "experience_years" in merged.columns:
            in_jd_range = ((merged["experience_years"] >= 5) &
                           (merged["experience_years"] <= 9)).sum()
    else:
        warnings.append(
            f"features_df.csv not found at {feat_path} — skipping honeypot/"
            f"experience-range checks (run rank.py first to generate it)"
        )

    # ── Report ──────────────────────────────────────────────────────────────
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ❌ {e}")
        for w in warnings:
            print(f"  ⚠️  {w}")
        sys.exit(1)

    print("="*50)
    print(f"✅  PASS — {csv_path}")
    print("="*50)
    print(f"  100 rows                          ✅")
    print(f"  Unique candidate IDs               ✅")
    print(f"  Ranks 1–100                        ✅")
    print(f"  Scores non-increasing              ✅")
    print(f"  No empty reasoning                 ✅")
    if honeypot_rate is not None:
        print(f"  Honeypot rate                      {honeypot_rate:.1%}  (< 10% ✅)")
    if in_jd_range is not None:
        print(f"  In JD experience range (5–9yr)     {in_jd_range}/100")
    for w in warnings:
        print(f"  ⚠️  {w}")
    sys.exit(0)

if __name__ == "__main__":
    main()