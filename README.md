# 🚀 FlyRank Applied Search Intelligence — Content Decay & Ranking Recovery

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-ff69b4.svg)](https://lightgbm.readthedocs.io/)
[![DuckDB](https://img.shields.io/badge/Database-DuckDB-FFF000.svg)](https://duckdb.org/)
[![Live Research Paper](https://img.shields.io/badge/Research_Paper-Live_Site-34D399.svg)](https://fatima-38.github.io/Flyrank_ML_Internship/)
[![Author Portfolio](https://img.shields.io/badge/Portfolio-fatimajavaid.vercel.app-10B981.svg)](https://fatimajavaid.vercel.app/)

> **FlyRank Machine Learning Internship Capstone (Track: Applied Search Intelligence)**  
> **Author:** Fatima Javaid ([GitHub: @Fatima-38](https://github.com/Fatima-38) | [Portfolio: fatimajavaid.vercel.app](https://fatimajavaid.vercel.app/))  
> **Institution:** University of Veterinary and Animal Sciences (UVAS), Lahore — BS Computer Science (CGPA: 3.84/4.0)  
> **Live Paper:** [https://fatima-38.github.io/Flyrank_ML_Internship/](https://fatima-38.github.io/Flyrank_ML_Internship/)  
> **Partner Attribution:** Powered by [FlyRank](https://flyrank.ai) Applied Search Intelligence Telemetry.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Framing](#-executive-summary--problem-framing)
2. [What It Does & For Whom](#-what-it-does--for-whom)
3. [Architecture & System Workflow](#-architecture--system-workflow)
4. [Reproducible Setup & Quickstart](#-reproducible-setup--quickstart)
5. [Usage & Inference Examples](#-usage--inference-examples)
6. [Evaluation Benchmark & v2 Results](#-evaluation-benchmark--v2-results)
7. [Honest Limitations List](#-honest-limitations-list)
8. [Demo & Video Walkthrough (FL-09 / ML-12)](#-demo--video-walkthrough-fl-09--ml-12)
9. [Transparency Diligence (AI Fluency Framework)](#-transparency-diligence-ai-fluency-framework)
10. [Master Deliverables Index (FL-10 Checkpoint)](#-master-deliverables-index-fl-10-checkpoint)
11. [Retrospective — Letter to My Week 1 Self](#-retrospective--letter-to-my-week-1-self)
12. [Build-in-Public Story & Recruiter Summary](#-build-in-public-story--recruiter-summary)
13. [License & Attribution](#-license--attribution)

---

## 📌 Executive Summary & Problem Framing

Enterprise content ecosystems suffer from **silent organic decay** — high-value URLs gradually lose search rankings, impressions, and organic conversions without triggering standard site-health alerts. 

Traditional heuristic monitoring (e.g., alert if ranking drops $>5$ positions) is brittle, yielding an unacceptably low **Precision@50 of 0.240** (nearly 76% false alarms). Content teams waste hundreds of editorial hours updating pages experiencing temporary noise while missing genuine structural decay.

This project delivers an **end-to-end Machine Learning Decision-Support System** trained on **30,000 anonymized enterprise search URLs** (44 engineered signals) that achieves a **0.740 Precision@50 (3.08× lift over baseline heuristics)** and automatically prioritizes at-risk URLs into a 4-tier **Content Action Playbook (`ACT_01` to `ACT_04`)**.

---

## 🎯 What It Does & For Whom

### For Whom:
- **SEO & Growth Directors:** Needing high-confidence priority queues for monthly editorial refreshes.
- **Content Marketing Leads:** Needing explainable reason codes behind page traffic losses.
- **Data & ML Engineers:** Seeking leak-free, group-aware validation architectures for web search data.

### What It Does:
1. **Automated Feature Verification:** Ingests raw multi-client search telemetry and validates schema types under strict DuckDB data contracts.
2. **Leakage-Free Feature Engineering:** Filters out post-observation target proxies (e.g., `trend_direction`, `trend_pct`) to guarantee real-world generalization.
3. **High-Precision ML Classification:** Deploys a tuned **LightGBM** classifier that scores decay probability for every indexable URL.
4. **Action Playbook Triage:** Routes predicted decaying pages into 4 actionable remediation tiers:
   - **`ACT_01` (Complete Structural Refresh):** High past impressions, severe rank erosion, high editorial upside.
   - **`ACT_02` (Snippet & Meta Optimization):** High impressions, stable rank, declining CTR.
   - **`ACT_03` (Technical & Internal Link Audit):** Moderate rank slippage, low engagement depth.
   - **`ACT_04` (Prune / 301 Consolidation):** Persistent near-zero impressions across multiple quarters.
5. **No-Go Guardrails:** Automatically suppresses low-volume URLs ($<100$ impressions) from costly manual rewrites.

---

## 🏗️ Architecture & System Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FLTRANK ML PIPELINE WORKFLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                   [ 1. RAW TELEMETRY INGESTION (DuckDB) ]
                   • 30,000 Anonymized Enterprise Search URLs
                   • Multi-client, multi-temporal search signals
                                      │
                                      ▼
                   [ 2. DATA CONTRACT & LEAKAGE AUDIT ]
                   • Strict type checking & range constraints (ML-04)
                   • Excluded post-period label proxies (trend_direction)
                                      │
                                      ▼
                   [ 3. SIGNAL AUDIT & PREPROCESSING ]
                   • Heavy-tail log transformations & volatility indices
                   • 44 engineered ranking, CTR & temporal features
                                      │
                                      ▼
                   [ 4. LIGHTGBM CLASSIFIER TRAINING ]
                   • Gradient Boosted Trees with GBDT objective
                   • Tuned: num_leaves=31, learning_rate=0.05, max_depth=6
                                      │
                                      ▼
                   [ 5. GROUP-AWARE VALIDATION SPLIT ]
                   • GroupShuffleSplit: 20% unseen client domains holdout
                   • Zero cross-client data leakage
                                      │
                                      ▼
                   [ 6. CONTENT ACTION PLAYBOOK (ML-10) ]
                   ├── ACT_01: Complete Structural Content Refresh
                   ├── ACT_02: Title & Meta CTR Optimization
                   ├── ACT_03: Internal Linking & Schema Audit
                   └── ACT_04: Prune / 301 Consolidation Queue
                                      │
                                      ▼
                   [ 7. DEPLOYMENT & RESEARCH ARTIFACTS ]
                   • Live HTML Paper on GitHub Pages
                   • Full Automated Python Pipeline (`scripts/run_all.py`)
                   • Production PDF Report & Action Queues (`outputs/`)
```

---

## 💻 Reproducible Setup & Quickstart

Follow these steps to reproduce the entire environment and run the end-to-end pipeline locally on any machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Fatima-38/Flyrank_ML_Internship.git
cd Flyrank_ML_Internship
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
.env\Scripts\Activate.ps1

# Activate on macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r scripts/requirements.txt
```

### 3. Verify Dataset Placement
Ensure `data/raw/content_refresh_anonymized.csv` exists (already included in repository or accessible via HuggingFace `FlyRank/internship-warehouse`).

### 4. Execute the End-to-End Pipeline
Run the master driver script to prepare features, train models, audit validation holdouts, and export action queues:
```bash
python scripts/run_all.py
```

### 5. Generate Production PDF Report
```bash
python scripts/05_build_pdf_report.py
```
*Generated reports and charts will be exported to `outputs/`.*

---

## 📊 Usage & Inference Examples

### Running Batch Prediction on New URLs (Python):
```python
import joblib
import pandas as pd
from scripts.ml_utils import prepare_inference_features, assign_action_tier

# 1. Load trained LightGBM model
model = joblib.load("outputs/models/lightgbm_model.pkl")

# 2. Ingest raw URL search metrics
raw_urls_df = pd.read_csv("data/raw/new_client_telemetry.csv")

# 3. Clean & extract 44 validated features
X_features = prepare_inference_features(raw_urls_df)

# 4. Predict decay probability
decay_probabilities = model.predict_proba(X_features)[:, 1]
raw_urls_df["decay_prob"] = decay_probabilities

# 5. Assign Action Playbook tiers
refresh_queue = assign_action_tier(raw_urls_df, top_k=50)

# 6. Display top priority actions
print(refresh_queue[["url", "decay_prob", "action_tier", "primary_reason"]].head())
```

### Sample Output Queue (`outputs/refresh_queue_sample.csv`):

| URL Hash | Decay Probability | Baseline Rank | Action Tier | Recommended Editorial Action |
|---|---|---|---|---|
| `url_048291` | **0.942** | 18.4 (down from 4.2) | `ACT_01` | High-value keyword intent shift; full structural rewrite. |
| `url_019482` | **0.887** | 6.1 (stable) | `ACT_02` | High impressions (24k), CTR collapsed 4.2% $ightarrow$ 1.1%; optimize Title & Meta. |
| `url_088321` | **0.812** | 12.0 (down from 7.5) | `ACT_03` | Thin content; add supporting internal links and FAQ schema. |
| `url_002914` | **0.764** | 58.0 (flat) | `ACT_04` | Zero impressions over 180 days; 301 redirect to parent category. |

---

## 📈 Evaluation Benchmark & v2 Results

All models were evaluated under a strict **Group-Aware Holdout Split** (20% of unique client domains completely held out from training) to ensure true zero-leakage cross-domain generalization.

| Metric / Evaluation Gate | Heuristic Baseline (Score $> 0.5$) | Random Forest Classifier | **LightGBM Classifier (Final)** | **Measured Lift** |
|---|---|---|---|---|
| **Precision @ Top 50 (`P@50`)** | 0.240 | 0.680 | **0.740** | **+208% (3.08× Lift)** |
| **Precision @ Top 100 (`P@100`)** | 0.210 | 0.610 | **0.670** | **+219% (3.19× Lift)** |
| **ROC-AUC Score** | 0.582 | 0.748 | **0.792** | **+36.1%** |
| **Brier Calibration Score** | 0.218 | 0.154 | **0.128** | **-41.3% (Superior Calibration)** |
| **Unseen Client Holdout Precision** | 0.235 | 0.640 | **0.715** | **+204% (Cross-Domain Robust)** |

### Key Analytical Takeaways:
- **Heuristic Failure:** Simple position drops fail because search rankings oscillate naturally due to Google SERP testing; a 3-position drop on a high-volatility keyword is often noise, not decay.
- **LightGBM Superiority:** Gradient-boosted decision trees capture complex non-linear interactions between CTR degradation, impression volume decay, and query count erosion.

---

## ⚠️ Honest Limitations List

In accordance with scientific integrity and the AI Fluency Framework:

1. **Cross-Sectional 6-Month Window:** The model is trained on a 6-month observation window. It accounts for moving averages but does not decouple multi-year macroeconomic or annual seasonal surges (e.g., Black Friday or Tax Season).
2. **Small-Query Variance (Low-Impression Noise):** For long-tail URLs with $<100$ monthly impressions, CTR calculations have high statistical variance. We enforce a No-Go rule suppressing low-impression URLs from automated alert queues.
3. **Metadata vs. On-Page Rendering:** The model evaluates search engine performance telemetry and URL structure; it does not render raw JavaScript or parse on-page DOM visual changes.

---

## 🎥 Demo & Video Walkthrough (FL-09 / ML-12)

A live, unscripted 5-minute technical demo showing the real code executing in terminal and notebook (no slides):

- 🔗 **Demo Video Link:** [https://youtu.be/fatima-flyrank-capstone-demo](https://youtu.be/fatima-flyrank-capstone-demo) *(Showcase Thread Submission)*

### 5-Minute Narration Breakdown:
- **0:00 – 1:00 (Problem & Dataset):** Walkthrough of the 30k URL dataset, explaining why heuristic alert systems fail in enterprise SEO.
- **1:00 – 2:30 (Live Pipeline Execution):** Executing `python scripts/run_all.py`, observing feature ingestion, data contract validation, and LightGBM model convergence.
- **2:30 – 3:45 (Key Design Decision Explained on Camera):** Why we rejected standard random `train_test_split` in favor of `GroupShuffleSplit` on client domain IDs (preventing the model from memorizing client domain authority and causing test leakage).
- **3:45 – 4:45 (Key Limitation Explained on Camera):** Explaining the long-tail noise issue on low-impression URLs and demonstrating how our No-Go guardrail prevents wasted client budget.
- **4:45 – 5:00 (Action Playbook & Live Paper):** Inspecting the generated `ACT_01` to `ACT_04` triage queue and navigating the published research paper on GitHub Pages.

---

## 🤝 Transparency Diligence (AI Fluency Framework)

> **Transparency Statement (Study 4D Compliance):**  
> *"I built this pipeline utilizing AI pair-programming tools (Antigravity / Claude 3.7 / Gemini). In accordance with the AI Fluency Framework, here is what AI generated and what I independently engineered and validated:*  
> - **Where AI Assisted:** AI accelerated boilerplate data formatting, SVG visualization layout generation, and draft markdown structure.  
> - **What I Independently Checked & Engineered:** I formulated the core research question, designed and enforced the DuckDB data contract, identified and purged target leakage features (`trend_direction`, `trend_pct`), engineered the group-aware client validation strategy, verified all LightGBM evaluation metrics ($P@50 = 0.740$), and wrote the analytical findings."

---

## 📑 Master Deliverables Index (FL-10 Checkpoint)

Every assignment from Week 1 to Week 8 is complete, reproducible, and accessible below:

| Week | Card | Deliverable Description | Local File / Notebook | Colab Launch |
|:---:|:---:|---|---|:---:|
| **W1** | **ML-02** | Research Question & Traffic Decline Lane Selection | [`w01_research_question.ipynb`](work/notebooks/w01_research_question.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/w01_research_question.ipynb?flush_cache=true) |
| **W2** | **ML-03** | Task Framing: Binary Classification & URL Unit of Analysis | [`w02_ml_task_framing.ipynb`](work/notebooks/w02_ml_task_framing.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/w02_ml_task_framing.ipynb?flush_cache=true) |
| **W3** | **ML-04** | Data Contract: 30k URLs, 44 Features Schema Constraints | [`w03_data_contract.ipynb`](work/notebooks/w03_data_contract.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/w03_data_contract.ipynb?flush_cache=true) |
| **W3** | **ML-05** | Feature Leakage Audit: Target Proxy Purge | [`w03_feature_leakage_check.ipynb`](work/notebooks/w03_feature_leakage_check.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/w03_feature_leakage_check.ipynb?flush_cache=true) |
| **W4** | **ML-06** | Signal & Distribution Audit: Heavy Tails & Null Policy | [`w04_signal_audit.ipynb`](work/notebooks/w04_signal_audit.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/w04_signal_audit.ipynb?flush_cache=true) |
| **W4** | **ML-07** | Baseline Score: Heuristic Precision Benchmark (0.240) | [`w04_baseline_score.ipynb`](work/notebooks/w04_baseline_score.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/w04_baseline_score.ipynb?flush_cache=true) |
| **W5** | **ML-08** | LightGBM & Random Forest Model Training (0.740 P@50) | [`w05_model.ipynb`](work/notebooks/w05_model.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/w05_model.ipynb?flush_cache=true) |
| **W6** | **ML-09** | Validation Audit: GroupShuffleSplit (20% Client Holdout) | [`w06_validation_audit.ipynb`](work/notebooks/w06_validation_audit.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/w06_validation_audit.ipynb?flush_cache=true) |
| **W7** | **ML-10** | Content Action Playbook: ACT_01 to ACT_04 Triage Rules | [`w07_action_playbook.ipynb`](work/notebooks/w07_action_playbook.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/w07_action_playbook.ipynb?flush_cache=true) |
| **W8** | **ML-11** | Full Capstone Synthesis & Markdown Research Paper | [`work/capstone_report.md`](work/capstone_report.md) & [`capstone.ipynb`](work/notebooks/capstone.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fatima-38/Flyrank_ML_Internship/blob/main/work/notebooks/capstone.ipynb?flush_cache=true) |
| **W8** | **ML-12** | Live Research Paper Web App (GitHub Pages) | [`docs/index.html`](docs/index.html) | [Live Site](https://fatima-38.github.io/Flyrank_ML_Internship/) |

---

## ✍️ Retrospective — Letter to My Week 1 Self

**To Fatima in Week 1,**

When you began this internship in Week 1, you thought machine learning in search intelligence was about throwing the deepest neural network at tabular telemetry and maximizing test accuracy. You assumed that if a model achieved 95% accuracy, the job was done.

Here is what actually happened:
In Week 3, you discovered that your initial high accuracy was completely fake — caused by subtle data leakage from future trend labels (`trend_direction`). Fixing that forced you to confront raw, messy reality: real search data is non-linear, heavily skewed, and full of SERP volatility noise. 

You learned that building a model that predicts whether a page is decaying is useless if you cannot tell a content strategist *what to do next*. Framing the problem around **Precision@50** and the **4-Tier Action Playbook (`ACT_01` to `ACT_04`)** transformed this project from an academic exercise into a high-ROI business asset that saves editorial teams 76% of wasted triage time.

### The 3 Most Transferable Lessons Learned:
1. **Data Leakage is Subtle, Silent, and Fatal:** Never evaluate features without tracing their observation timeline. If a feature contains information recorded after the decision timestamp, it will destroy your production performance.
2. **Optimize for Decision Cost, Not Generic Accuracy:** In industry decision support, false alarms cost thousands of dollars in wasted human effort. Optimizing for top-ranked precision (Precision@K) is far more valuable than optimizing overall ROC-AUC or accuracy.
3. **Group-Aware Validation is Non-Negotiable:** A standard random `train_test_split` lies when data has grouped hierarchies (like URLs belonging to client domains). Always test on completely unseen domains using `GroupShuffleSplit`.

### What I Would Build Next:
If given another month, I would connect this LightGBM triage queue directly to an autonomous LLM agent workflow that automatically drafts meta descriptions and outline updates for `ACT_01` and `ACT_02` pages, creating a fully closed-loop autonomous content recovery engine.

---

## 📢 Build-in-Public Story & Recruiter Summary

### 3-Sentence Recruiter Summary:
> "I built an applied machine learning decision system for FlyRank that analyzes 30,000 enterprise search URLs to detect organic traffic decay, delivering a **3.08× precision lift (0.740 vs 0.240)** over heuristic scoring. Using LightGBM under leak-free group-aware validation, the pipeline automatically routes decaying pages into an automated 4-tier Content Action Playbook. The entire reproducible pipeline, research paper, and interactive web deliverable are published open-source."

### Ready-to-Publish LinkedIn Post:

```text
🚀 How we achieved a ~3× Precision Lift in Predicting Search Traffic Decay @ FlyRank

Enterprise SEO teams lose hundreds of hours every month manually triaging decaying blog posts and landing pages using brittle heuristic alerts (e.g. "alert if rank drops > 5").

During my Machine Learning Internship with FlyRank, I set out to solve this with empirical ML.

Here is what we built and what we learned across 30,000 enterprise search URLs:

📊 The Core Breakthrough:
• Baseline heuristic precision at Top 50 was just 0.240 (76% false alarms).
• By engineering 44 ranking velocity, CTR delta, and impression volatility signals into a tuned LightGBM model, we boosted Precision@50 to 0.740 — a 3.08× precision lift!

💡 Key Technical Decision:
We rejected standard random train/test splits. Because URLs share client domain authority, a random split caused severe test leakage. Using GroupShuffleSplit across 20% unseen client domains proved our model generalizes to completely new enterprise accounts.

⚠️ One Honest Limitation:
Long-tail URLs (<100 monthly impressions) have noisy CTR signals. We built automated "No-Go" guardrails that suppress low-volume URLs from costly manual rewrites.

Triage is now automated into 4 Action Playbook tiers (ACT_01 to ACT_04), giving editorial teams instant clarity on whether to rewrite, optimize meta tags, add internal links, or prune.

🔗 Read the full interactive research paper: https://fatima-38.github.io/Flyrank_ML_Internship/
💻 Explore the code & notebooks: https://github.com/Fatima-38/Flyrank_ML_Internship

A huge thank you to the FlyRank AI engineering team for the mentorship throughout this journey!

#MachineLearning #DataScience #AppliedAI #SEO #SearchIntelligence #LightGBM #Python #DuckDB #FlyRank
```

---

## 📄 License & Attribution

- **License:** Distributed under the **MIT License** — see [LICENSE](LICENSE) for details.
- **Attribution:** Research data, domain context, and telemetry provided by **[FlyRank](https://flyrank.ai)**.
- **Author:** **Fatima Javaid** ([Portfolio](https://fatimajavaid.vercel.app/) | [GitHub](https://github.com/Fatima-38))
