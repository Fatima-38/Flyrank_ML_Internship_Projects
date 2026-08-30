# Capstone Report — Content Refresh & Traffic Decline Lane

- **Author:** Fatima Javaid
- **Lane:** Content Refresh / Traffic Decline (Applied Search Intelligence)
- **Repo:** https://github.com/Fatima-38/Flyrank_ML_Internship_Projects
- **Date:** August 2026

---

## 0. Abstract

Identifying high-value search content suffering from organic traffic decay is critical for enterprise SEO efficiency, yet traditional heuristics often trigger false alarms on healthy pages. We evaluated an anonymized corpus of 30,000 enterprise URLs spanning 44 behavioral, engagement, and crawl signals to formulate an automated prioritization system. Using a group-aware, time-honest evaluation split to prevent data leakage, we trained Gradient Boosted Trees (LightGBM) and Random Forest classifiers against a transparent multi-criteria baseline rule. The learned model achieved a top-50 precision of **0.740** (compared to the baseline's **0.240**), representing a **~3× precision lift** in correctly detecting traffic-declining assets while uncovering an inverse relationship between unoptimized word count and rank stability. These predictions are operationalized into an actionable four-tier Content Action Playbook that enforces human editorial sign-offs on high-impact interventions.

---

## 1. Problem Framing

- **Decision Supported:** Allocation of weekly editorial and technical SEO resources. The model determines which URLs enter the content rewrite and metadata refresh queues.
- **Unit of Analysis:** One individual web page snapshot (URL-level observation) observed across a 90-day evaluation window.
- **Output:** A ranked probability score (0.0 to 1.0) paired with an operational Action Code (`ACT_01` through `ACT_04`) and primary reason codes.
- **Action Taken by Humans:** Content Strategists, Copywriters, and SEO Leads execute structural revamps or metadata updates.
- **Cost of Asymmetric Errors:**
  - *False Positive:* Wastes valuable editorial hours modifying healthy or seasonally shifting pages, risking accidental loss of existing rank.
  - *False Negative:* Fails to intervene on decaying pages, leading to long-term traffic loss and competitor displacement.

---

## 2. Data Safety

- **Data Source:** Anonymized 30,000-page dataset provided by FlyRank (`data/raw/content_refresh_anonymized.csv`), containing 44 raw and derived features.
- **Explicit Exclusions:**
  - *PII & Client Entities:* All client identifiers are pseudonymous hashes used solely for grouping.
  - *Direct Target Leakage:* `trend_direction` and `trend_pct` are strictly excluded from the feature matrix because the binary classification label (`is_declining_label`) is derived from them.
  - *Post-Outcome Timestamps:* No future conversion windows or post-event metrics are present.
- **Leakage Test Results:** Automated inspection verified 0 forbidden columns in the input feature space.

---

## 3. Baseline Heuristic Rule

- **Hand-Written Rule:** Composite action score combining historical traffic decay velocity (60% weight) with normalized keyword search volume (40% weight).
- **Reason Codes:** `HIGH_TRAFFIC_DECAY`, `STAGNANT_ENGAGEMENT`, `OPTIMIZATION_WINDOW`.
- **Baseline Performance:**
  - Precision@50: **0.240**
  - Macro F1: **0.481**
  - ROC-AUC: **0.540**

---

## 4. Model & Analysis Strategy

- **Methodology:** Gradient Boosted Trees (LightGBM) and Random Forest Classifier.
- **Features Included:** Structural content metrics (`word_count`, `char_count`), commercial value signals (`cpc`, `competition`), audience interaction metrics (`impressions_90d`, `scroll_rate`), and crawling/indexation signals.
- **Target Definition:** Binary label where `is_declining_label = 1` if `trend_direction == 'down'`, else `0`.

---

## 5. Evaluation

- **Split Strategy:** Group-aware client holdout split (`GroupShuffleSplit`, 20% holdout) ensuring no client appears in both train and test partitions.
- **Benchmark Results on Identical Test Split (n=6,000):**
  - *Baseline Rule:* Precision@50 = 0.240 | F1 = 0.481 | AUC = 0.540
  - *Logistic Regression:* Precision@50 = 0.420 | F1 = 0.582 | AUC = 0.635
  - *Random Forest:* Precision@50 = 0.710 | F1 = 0.694 | AUC = 0.762
  - *LightGBM (Final Model):* Precision@50 = **0.740** | F1 = **0.728** | AUC = **0.791**
- **Precision Lift:** The LightGBM model achieves a **3.08× precision lift** over the baseline heuristic.

---

## 6. Interpretation & Discoveries

1. **Search Volume Decoupling:** Estimated keyword search volume has virtually zero correlation with 90-day impressions (`r = 0.001`), demonstrating that legacy keyword search volume alone is an unreliable proxy for actual traffic health.
2. **Word Count Paradox:** Declining pages exhibited a higher median word count (2,909 words) than growing pages (2,848 words), proving that simply creating longer content does not prevent search ranking decay.
3. **High Feature Value:** Engagement velocity drops combined with CTR degradation provided the cleanest non-linear signals of imminent decay.

---

## 7. Recommendations & Operational Playbook

### Action Queue Tiers:
- **`ACT_01` (Immediate Rewrite):** High-confidence decay (Score > 0.85) on historically high-traffic pages. Requires structural overhaul.
- **`ACT_02` (Targeted Refresh):** Moderate decay (Score 0.65 - 0.85). Metadata, title tags, and internal link updates.
- **`ACT_03` (Archive / Consolidate):** Low-traffic decaying pages (>180 days). 301 consolidation to parent topics.
- **`ACT_04` (Monitor Only):** Stable pages (Score < 0.50). Continued automated tracking.

### Safety & No-Go Safeguards:
- No automated page unpublishing or URL deletions.
- No autonomous brand copy changes without human sign-off.
- Mandatory manual review for predictions with confidence between 0.50 and 0.75.

---

## 8. Reproducibility

To re-run the full evaluation pipeline from scratch:

```bash
git clone https://github.com/Fatima-38/Flyrank_ML_Internship_Projects.git
cd Flyrank_ML_Internship_Projects
pip install -r requirements.txt
python scripts/run_all.py
```

- **Random Seeds:** `random_state = 42` across all splitters and estimators.
- **Environment:** Python 3.10+, `scikit-learn==1.4.0`, `lightgbm==4.3.0`, `duckdb==0.10.0`.

---

## 9. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset**, provided by [FlyRank.ai](https://flyrank.ai). Special thanks to Track Leads Mirza Ašćerić (ML) and Hole (Data Engineering) for guidance throughout the internship.
