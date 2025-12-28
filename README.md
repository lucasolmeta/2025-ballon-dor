### Ballon d'Or 2025 — reproducible pipeline

This repo pulls FBref season totals for Ballon d'Or finalists and runs a set of analyses to answer:
- who deserved to win 2025?
- how close was it?
- was the winner’s level worse than previous years?

### Quickstart

- **Python version**

Use **Python 3.12** (or 3.11/3.12). Some dependencies (notably `scikit-learn`) do not support Python 3.13 yet, so creating a venv from 3.13 will fail.

- **Create an environment and install deps**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **Fetch raw FBref data**

```bash
python -m setup_scripts.fetch
```

- **Preprocess (merges votes + computes vote share targets)**

```bash
python -m setup_scripts.preprocess
```

### Part 4 (vote share model)

```bash
python -m analysis.modeling.predict_vote_share
```

Outputs:
- tables: `analysis/tables/part4_2024_25_predictions_*.csv`, `analysis/tables/part4_metrics_by_season.csv`
- figures:
  - `analysis/graphs/part4_pred_vs_actual_vote_share_*`
  - `analysis/graphs/part4_top10_pred_vs_actual_2425_*`
  - `analysis/graphs/part4_residuals_over_under_2425_*`
  - `analysis/graphs/part4_*_coefficients.png`
  - `analysis/graphs/part4_*_perm_importance.png`

### Notes

- **Voting data** is frozen in `data/ballon_dor_votes.csv` for reproducibility.
- `setup_scripts/preprocess.py` computes:
  - `Vote Share Outfield` (modeling target; outfield nominees only)
  - `Vote Share Overall` (all finalists; useful for “how close overall?”)

- If you prefer running scripts by file path (e.g. `python setup_scripts/preprocess.py`), you can also do:

```bash
PYTHONPATH=. python setup_scripts/preprocess.py
```

### Optional: scrape votes (helper)

This repo keeps `data/ballon_dor_votes.csv` as a frozen snapshot for reproducibility.
If you want a convenience script to regenerate it from Wikipedia (may require tweaks):

```bash
python -m setup_scripts.scrape_votes_wikipedia --out data/ballon_dor_votes.csv
```


