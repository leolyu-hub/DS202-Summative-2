# DS202W Summative 2 — Glassdoor Employee Reviews

LSE DS202W (Data Science for Social Scientists) summative assignment.
Unsupervised analysis of **838,566 Glassdoor employee reviews** answering two
research questions:

- **Part A — Similarity & Anomaly:** How linguistically similar are employees'
  descriptions of different companies, and which reviews are anomalous?
- **Part B — Temporal Shift:** How did employee vocabulary change between
  the pre-COVID period (2008–2019) and the COVID period (2020–2021)?

## Methods

Stages are listed in the order they appear in `60775.qmd`. Every code chunk
in the notebook is followed by its own prose interpretation — the analytical
reading is inline beneath the code that produced it, not collected separately
at the end.

| Stage | Technique | §  |
|---|---|---|
| 1. EDA overview | Rating / temporal / company / industry distributions | §1 |
| 2. Stratified sampling | 15-strata proportional design, n = 150,000 | §3.1 |
| 3. Text preprocessing | stopword removal + spaCy lemmatisation | §3.2 |
| 4. Vectorisation | TF-IDF, separate fits for pros / cons | §3.3 |
| 5. Latent Semantic Analysis (LSA) | TruncatedSVD → double L2 normalisation → 3-D LSA projection| §3.4-§3.5 |
| 6. Firm similarity | Centroid cosine similarity in 200-dim joint LSA space | §4 |
| 7. Anomaly detection | Isolation Forest | §5.1 |
| 8. Topic discovery | BERTopic (all-MiniLM-L6-v2 → UMAP → HDBSCAN) | §8.1–§8.2 |
| 9. Topic period analysis | COVID-excess per topic; enriched / depleted topics | §8.3 |
| 10. Term-level temporal shift | Log-ratio vocabulary comparison, pre-COVID (2008–19) vs COVID (2020–21) | §8.4 |
| 11. Sentiment by period | Text valence (1–5 scale) by period | §8.5 |

## Files

- **`60775.qmd`** — full Quarto notebook (source)
- **`60775.html`** — rendered report (~32 MB, self-contained)

## Reproducing the report

### 1. Download the data

The raw data is **not** committed to this repo because the 279 MB CSV exceeds
GitHub's 100 MB file size limit. Download it from Kaggle:

🔗 **https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews**

Place `glassdoor_reviews.csv` in the repository root.

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn \
            spacy bertopic umap-learn hdbscan sentence-transformers
python -m spacy download en_core_web_sm
```

A Quarto installation is also required (https://quarto.org).

### 3. Render

```bash
quarto render 60775.qmd
```

First run takes ~10 minutes (BERTopic embedding + HDBSCAN are the
bottlenecks). Subsequent renders use the Jupyter cache.


## Notes

- The pipeline is deterministic given fixed `random_state=42` in
  TruncatedSVD, UMAP, and the stratified sampler.
