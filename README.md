# DS202W Summative 2 — Glassdoor Employee Reviews

LSE DS202W (Data Science for Social Scientists) summative assignment.
Unsupervised analysis of **838,566 Glassdoor employee reviews** answering two
research questions:

- **Part A — Similarity & Anomaly:** How linguistically similar are employees'
  descriptions of different companies, and which reviews are anomalous?
- **Part B — Temporal Shift:** How did employee vocabulary change between
  the pre-COVID period (2008–2019) and the COVID period (2020–2021)?

## Methods

| Stage | Technique |
|---|---|
| Sampling | 15-strata (period × rating) proportional design, n = 150,000 |
| Preprocessing | spaCy lemmatisation + domain stoplist |
| Vectorisation | TF-IDF (min_df=5, max_df=0.85, max_features=8000, ngrams 1–3) |
| Dimensionality reduction | TruncatedSVD → 100 dims per field (pros / cons), joint 200-dim with double L2 normalisation |
| Firm similarity | Centroid cosine similarity in 200-dim joint LSA space |
| Anomaly detection | Isolation Forest (contamination=0.05) + Rating–Text Divergence |
| Topic discovery | BERTopic (all-MiniLM-L6-v2 → UMAP → HDBSCAN min_cluster_size=200) |
| Temporal shift | Log-ratio vocabulary comparison, pre-COVID vs COVID |

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

## Viewing the rendered HTML

The 32 MB rendered HTML is too large for GitHub's inline preview. Either:

- **Clone and open locally** — fastest.
- Use [htmlpreview.github.io](https://htmlpreview.github.io/?https://github.com/leolyu-hub/DS202-Summative-2/blob/main/60775.html)
  (slow load due to file size).

## Notes

- Dataset coverage ends June 2021 — there is no post-COVID (2022+) slice in
  the analysis.
- The pipeline is deterministic given fixed `random_state=42` in
  TruncatedSVD, UMAP, and the stratified sampler.
