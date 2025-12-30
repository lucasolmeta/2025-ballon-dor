"""
Part 4 — Predict Ballon d'Or outcome (nominees-only; vote share target).

This script:
- trains regularized linear models to predict outfield vote share within-season,
- evaluates on a train/val/test split by season,
- produces a 2024-25 table with predicted vs actual vote share + ranks + residuals,
- saves a couple of simple plots for the report.

Expected inputs:
- data/nominees_outfield.csv

Outputs:
- analysis/tables/part4_2024_25_predictions_*.csv
- analysis/tables/part4_2024_25_uncertainty_*.csv
- analysis/tables/part4_2024_25_closeness_*.csv
- analysis/tables/part4_metrics_by_season.csv
- analysis/tables/part4_model_comparison.csv
- analysis/graphs/part4_pred_vs_actual_vote_share_2324_*.png
- analysis/graphs/part4_pred_vs_actual_vote_share_2425_*.png
- analysis/graphs/part4_2024_25_predshare_ci_*.png
- analysis/graphs/part4_2024_25_top1_gap_bootstrap_*.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


DATA_PATH = Path("data/nominees_outfield.csv")
OUT_TABLE_DIR = Path("analysis/tables")
OUT_GRAPH_DIR = Path("analysis/graphs")

TRAIN_SEASONS = ["2021-2022", "2022-2023"]
VAL_SEASON = "2023-2024"
TEST_SEASON = "2024-2025"

TARGET_COL = "Vote Share Outfield"

BOOTSTRAP_N = 800
BOOTSTRAP_SEED = 0
BOOTSTRAP_CI_Q = (0.05, 0.95)  # 90% interval for paper-friendly uncertainty

EPS_LOG_TARGET = 1e-6

TROPHY_COLS = [
    "League Winner",
    "UCL Winner",
    "Cup Winner",
    "Major International Continental Trophy Winner",
    "World Cup Winner",
]

# Fixed, small feature set: mix of performance (per90), progression/creation, and durability.
# Keep this disciplined — with ~110 rows, feature explosion will make results unstable.
CORE_FEATURES = [
    # Performance / efficiency
    "Per 90 Minutes NpxG+xAG",
    "Per 90 Minutes G+A-PK",
    "Per 90 Minutes XAG",
    "SCA SCA90",
    # Chance creation / progression
    "KP",
    "Pass Types TB",
    "Carries 1/3",
    "Progression PrgC",
    "Progression PrgP",
    # Durability / availability
    "Playing Time 90s",
    "Starts Starts",
    "Playing Time Mn/MP",
]

BASELINE_METRIC_COL = "Per 90 Minutes NpxG+xAG"

# Baseline 2 (Part 3-style): similarity to an "average past winner profile".
SIMILARITY_FEATURES = [
    "Carries 1/3",
    "Expected Np:G-xG",
    "Pass Types TB",
    "Playing Time Mn/MP",
    "Performance Fld",
    "Blocks Sh",
    "Take-Ons Att",
    "Per 90 Minutes G+A-PK",
]


def softmax_share(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert arbitrary (possibly negative) scores to a valid share distribution via softmax.
    Ordering is preserved within a season, but values become nonnegative and sum to 1.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return s
    t = float(temperature) if temperature is not None else 1.0
    # Temperature must be positive. Smaller => more peaked; larger => more uniform.
    if not np.isfinite(t) or t <= 0:
        t = 1.0
    s = np.nan_to_num(s, nan=np.nanmin(s) if np.any(np.isfinite(s)) else 0.0)
    z = (s / t) - np.max(s / t)
    ez = np.exp(z)
    denom = ez.sum()
    if denom <= 0 or not np.isfinite(denom):
        return np.zeros_like(s)
    return ez / denom


def add_pred_share_for_single_season(
    df: pd.DataFrame, score_col: str, out_col: str, temperature: float = 1.0
) -> pd.DataFrame:
    """
    For a dataframe representing one season, generate a valid vote-share prediction from scores.
    """
    df = df.copy()
    df[out_col] = softmax_share(df[score_col].to_numpy(), temperature=temperature)
    return df


def add_per90_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small engineered feature set: totals -> per90 rates using Playing Time 90s.
    This lets performance-only models avoid mixing per90 features with raw totals.
    """
    df = df.copy()
    if "Playing Time 90s" not in df.columns:
        return df

    denom = df["Playing Time 90s"].replace(0, np.nan).astype(float)
    if "KP" in df.columns:
        df["KP90"] = (df["KP"] / denom).fillna(0.0)
    if "Pass Types TB" in df.columns:
        df["TB90"] = (df["Pass Types TB"] / denom).fillna(0.0)
    if "Carries 1/3" in df.columns:
        df["Carries13_90"] = (df["Carries 1/3"] / denom).fillna(0.0)
    if "Progression PrgC" in df.columns:
        df["PrgC90"] = (df["Progression PrgC"] / denom).fillna(0.0)
    if "Progression PrgP" in df.columns:
        df["PrgP90"] = (df["Progression PrgP"] / denom).fillna(0.0)
    return df


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Rank values with average ranks for ties.
    Returns ranks starting at 1 (like scipy.stats.rankdata(method='average')).
    """
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)

    # Average ranks for ties
    # We detect ties in the sorted array.
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            avg = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation implemented via Pearson on ranks (no scipy dependency)."""
    rx = _rankdata_average_ties(np.asarray(x))
    ry = _rankdata_average_ties(np.asarray(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of pairs ordered correctly.
    Ties in either y_true or y_pred are ignored.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    correct = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            dt = y_true[i] - y_true[j]
            dp = y_pred[i] - y_pred[j]
            if dt == 0 or dp == 0:
                continue
            total += 1
            correct += int((dt > 0 and dp > 0) or (dt < 0 and dp < 0))
    return float(correct / total) if total else float("nan")


def ensure_vote_share_outfield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure TARGET_COL exists. If preprocessing hasn't been re-run yet, compute it from Voting Points.
    """
    if TARGET_COL in df.columns:
        return df

    if "Voting Points" not in df.columns:
        raise ValueError(f"Missing 'Voting Points' column; can't compute {TARGET_COL}.")

    df = df.copy()
    totals = df.groupby("Season")["Voting Points"].transform("sum").replace(0, np.nan)
    df[TARGET_COL] = (df["Voting Points"] / totals).fillna(0.0)
    return df


def add_baseline_pred_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline 1: rank by a single metric (Per 90 Minutes NpxG+xAG).
    We convert it to a within-season pseudo-share for easier comparison.
    """
    if BASELINE_METRIC_COL not in df.columns:
        return df

    df = df.copy()
    # Clip to non-negative so the normalization makes sense.
    score = df[BASELINE_METRIC_COL].clip(lower=0)
    denom = score.groupby(df["Season"]).transform("sum").replace(0, np.nan)
    df["PredScore baseline_npxg_xag90"] = score
    df["Pred baseline_npxg_xag90"] = (score / denom).fillna(0.0)
    return df


def _season_to_index(seasons: list[str]) -> dict[str, int]:
    return {s: i for i, s in enumerate(sorted(seasons))}


def add_similarity_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline 2: cosine similarity to an "average previous winner" profile using SIMILARITY_FEATURES.

    For each season:
    - build winner profile from winners in earlier seasons,
    - min-max scale features within [season nominees + winner profile],
    - compute cosine similarity and normalize to a within-season pseudo-share.
    """
    missing = [c for c in SIMILARITY_FEATURES if c not in df.columns]
    if missing:
        return df

    df = df.copy()
    df["Pred baseline_similarity"] = np.nan
    df["PredScore baseline_similarity"] = np.nan

    season_idx = _season_to_index(sorted(df["Season"].unique()))

    for season in sorted(df["Season"].unique()):
        idx = season_idx[season]
        prior_seasons = [s for s, j in season_idx.items() if j < idx]
        winners_prior = df[(df["Winner"] == 1) & (df["Season"].isin(prior_seasons))]
        season_df = df[df["Season"] == season]

        if len(winners_prior) == 0 or len(season_df) == 0:
            continue

        winner_profile = winners_prior[SIMILARITY_FEATURES].mean().to_frame().T

        # Combine season nominees + winner profile for consistent scaling.
        combo = pd.concat([season_df[SIMILARITY_FEATURES], winner_profile], axis=0, ignore_index=True)

        scaled = combo.copy()
        for col in SIMILARITY_FEATURES:
            col_min = min(0.0, float(np.nanmin(scaled[col])))
            col_max = float(np.nanmax(scaled[col]))
            if col_max - col_min == 0:
                scaled[col] = 0.05
            else:
                scaled[col] = (scaled[col] - col_min) / (col_max - col_min)
                scaled[col] = scaled[col].clip(lower=0.05)

        season_vecs = scaled.iloc[: len(season_df)].to_numpy(dtype=float)
        winner_vec = scaled.iloc[len(season_df) :].to_numpy(dtype=float)[0]

        # cosine similarity
        denom = np.linalg.norm(season_vecs, axis=1) * np.linalg.norm(winner_vec)
        sim = (season_vecs @ winner_vec) / np.where(denom == 0, np.nan, denom)
        sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to pseudo-share within season so it can be compared to vote share.
        sim_sum = float(sim.sum())
        pred_share = sim / sim_sum if sim_sum > 0 else np.zeros_like(sim)

        df.loc[df["Season"] == season, "Pred baseline_similarity"] = pred_share
        df.loc[df["Season"] == season, "PredScore baseline_similarity"] = sim

    df["Pred baseline_similarity"] = df["Pred baseline_similarity"].fillna(0.0)
    df["PredScore baseline_similarity"] = df["PredScore baseline_similarity"].fillna(0.0)
    return df


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: list[str]
    y_transform: str | None = None  # None | "log_share"


def build_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=alpha, random_state=0)),
        ]
    )


def _transform_y(y: np.ndarray, transform: str | None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if transform is None:
        return y
    if transform == "log_share":
        return np.log(y + EPS_LOG_TARGET)
    raise ValueError(f"Unknown y_transform: {transform}")


def fit_choose_alpha(
    train_df: pd.DataFrame, val_df: pd.DataFrame, spec: ModelSpec, alphas: list[float]
) -> tuple[float, pd.DataFrame]:
    """
    Pick alpha by Spearman rank correlation on the validation season.
    Returns best alpha and a small diagnostics dataframe.
    """
    rows = []
    X_train = train_df[spec.features].to_numpy()
    y_train = _transform_y(train_df[TARGET_COL].to_numpy(), spec.y_transform)
    X_val = val_df[spec.features].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()

    for a in alphas:
        pipe = build_ridge(alpha=a)
        pipe.fit(X_train, y_train)
        pred_score = pipe.predict(X_val)
        # Evaluate on valid within-season shares (monotonic in score => rank metrics unchanged,
        # but residual/top-k plots become interpretable as share deltas).
        pred = softmax_share(pred_score)
        rows.append(
            {
                "model": spec.name,
                "alpha": a,
                "val_spearman": spearmanr(y_val, pred),
                "val_pairwise_acc": pairwise_accuracy(y_val, pred),
            }
        )

    diag = pd.DataFrame(rows).sort_values(by=["val_spearman", "val_pairwise_acc"], ascending=False)
    best_alpha = float(diag.iloc[0]["alpha"])
    return best_alpha, diag


def season_metrics(season_df: pd.DataFrame, pred_col: str) -> dict:
    y_true = season_df[TARGET_COL].to_numpy()
    y_pred = season_df[pred_col].to_numpy()
    winner_rows = season_df[season_df.get("Winner", 0) == 1]
    winner_name = winner_rows["Name"].iloc[0] if len(winner_rows) else None

    # ranks: 1 is best
    pred_rank = (-season_df[pred_col]).rank(method="average")
    true_rank = (-season_df[TARGET_COL]).rank(method="average")

    winner_pred_rank = None
    if winner_name is not None:
        winner_pred_rank = float(pred_rank[season_df["Name"] == winner_name].iloc[0])

    return {
        "season": season_df["Season"].iloc[0],
        "n_nominees": int(len(season_df)),
        "spearman": spearmanr(y_true, y_pred),
        "pairwise_acc": pairwise_accuracy(y_true, y_pred),
        "winner": winner_name,
        "winner_pred_rank": winner_pred_rank,
        "top1_hit": float(winner_pred_rank == 1.0) if winner_pred_rank is not None else float("nan"),
        "top3_hit": float(winner_pred_rank <= 3.0) if winner_pred_rank is not None else float("nan"),
        "top5_hit": float(winner_pred_rank <= 5.0) if winner_pred_rank is not None else float("nan"),
    }


def scatter_plot(season_df: pd.DataFrame, pred_col: str, out_path: Path, title: str) -> None:
    x = season_df[TARGET_COL].to_numpy()
    y = season_df[pred_col].to_numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.85)
    lim = max(float(x.max()), float(y.max()), 1e-9)
    plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
    plt.xlabel("Actual Vote Share (Outfield)")
    plt.ylabel("Predicted Vote Share (Outfield)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def top10_bar_pred_vs_actual(season_df: pd.DataFrame, pred_col: str, out_path: Path, title: str) -> None:
    plot_df = season_df[["Name", TARGET_COL, pred_col]].copy()
    plot_df = plot_df.sort_values(by=pred_col, ascending=False).head(10)

    x = np.arange(len(plot_df))
    width = 0.42

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, plot_df[TARGET_COL], width=width, label="Actual")
    plt.bar(x + width / 2, plot_df[pred_col], width=width, label="Predicted")
    plt.xticks(x, plot_df["Name"], rotation=45, ha="right")
    plt.ylabel("Vote Share (Outfield)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def residuals_plot(season_df: pd.DataFrame, pred_col: str, out_path: Path, title: str, top_k: int = 10) -> None:
    plot_df = season_df[["Name", TARGET_COL, pred_col]].copy()
    plot_df["Residual"] = plot_df[pred_col] - plot_df[TARGET_COL]

    over = plot_df.sort_values("Residual", ascending=False).head(top_k)
    under = plot_df.sort_values("Residual", ascending=True).head(top_k)
    both = pd.concat([under, over], axis=0, ignore_index=True)

    colors = ["#d62728" if r < 0 else "#2ca02c" for r in both["Residual"]]
    y = np.arange(len(both))

    plt.figure(figsize=(10, 7))
    plt.barh(y, both["Residual"], color=colors)
    plt.yticks(y, both["Name"])
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Residual (Pred − Actual Vote Share)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_model_interpretability(
    model: Pipeline, features: list[str], eval_df: pd.DataFrame, out_prefix: str
) -> None:
    """
    Save:
    - standardized coefficients (Ridge on standardized features),
    - permutation importance (on eval_df) as a sanity check.
    """
    coef = model.named_steps["model"].coef_
    coef_df = pd.DataFrame({"feature": features, "coef": coef})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    coef_df.to_csv(OUT_TABLE_DIR / f"{out_prefix}_coefficients.csv", index=False)

    # Coefficient bar plot (top 20 by abs)
    top = coef_df.head(20).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["coef"])
    plt.axvline(0, color="black", linewidth=1)
    plt.title(f"Top coefficients (standardized) — {out_prefix}")
    plt.tight_layout()
    plt.savefig(OUT_GRAPH_DIR / f"{out_prefix}_coefficients.png", dpi=300)
    plt.close()

    # Permutation importance (ranking-style objective isn't directly optimized,
    # but this is still a useful robustness sanity check).
    X = eval_df[features].to_numpy()
    y = eval_df[TARGET_COL].to_numpy()
    pi = permutation_importance(model, X, y, n_repeats=50, random_state=0)
    imp = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    imp.to_csv(OUT_TABLE_DIR / f"{out_prefix}_perm_importance.csv", index=False)

    topi = imp.head(20).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(topi["feature"], topi["importance_mean"], xerr=topi["importance_std"])
    plt.axvline(0, color="black", linewidth=1)
    plt.title(f"Permutation importance (val) — {out_prefix}")
    plt.tight_layout()
    plt.savefig(OUT_GRAPH_DIR / f"{out_prefix}_perm_importance.png", dpi=300)
    plt.close()


def _stratified_bootstrap(df: pd.DataFrame, group_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Sample with replacement within each group, preserving group sizes."""
    parts: list[pd.DataFrame] = []
    for _, g in df.groupby(group_col, sort=False):
        if len(g) == 0:
            continue
        idx = rng.integers(0, len(g), size=len(g))
        parts.append(g.iloc[idx])
    return pd.concat(parts, axis=0, ignore_index=True) if parts else df.copy()


def calibrate_temperature_on_season(
    scores: np.ndarray,
    y_true_share: np.ndarray,
    grid: list[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    """
    Calibrate a softmax temperature on a labeled season.

    Motivation:
    When model scores are tightly clustered, softmax with temperature=1.0 can yield near-uniform
    shares. That preserves ranks, but makes share-level quantities (residual magnitudes, top1 gaps)
    poorly calibrated / misleading.

    We pick the temperature that minimizes MSE between predicted shares and true shares.
    This preserves the within-season ordering (temperature scaling is monotone).
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y_true_share, dtype=float)

    if grid is None:
        # Log-ish grid: smaller => more peaked; larger => more uniform.
        grid = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0]

    if s.size == 0 or y.size == 0 or s.size != y.size:
        return 1.0, pd.DataFrame([])

    # If the target is missing/degenerate, fall back.
    if not np.isfinite(y).all() or float(np.nansum(y)) <= 0:
        return 1.0, pd.DataFrame([])

    rows = []
    best_t = 1.0
    best_mse = float("inf")
    for t in grid:
        pred = softmax_share(s, temperature=float(t))
        mse = float(np.mean((pred - y) ** 2))
        rows.append({"temperature": float(t), "mse": mse})
        if mse < best_mse:
            best_mse = mse
            best_t = float(t)

    diag = pd.DataFrame(rows).sort_values(by=["mse", "temperature"], ascending=[True, True])
    return best_t, diag


def bootstrap_uncertainty_for_season(
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    spec: ModelSpec,
    alpha: float,
    temperature: float,
    n_boot: int,
    seed: int,
    ci_q: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Stratified-by-season bootstrap uncertainty for one eval season.

    Returns:
    - per-player uncertainty table
    - season-level closeness summary table
    - top1 gap samples (for plotting)
    """
    rng = np.random.default_rng(seed)
    X_eval = eval_df[spec.features].to_numpy()
    n_eval = len(eval_df)

    shares = np.zeros((n_boot, n_eval), dtype=float)
    p_rank1 = np.zeros((n_boot, n_eval), dtype=float)
    p_rank3 = np.zeros((n_boot, n_eval), dtype=float)
    p_rank5 = np.zeros((n_boot, n_eval), dtype=float)
    top1_gaps = np.zeros(n_boot, dtype=float)
    winner_gaps = np.zeros(n_boot, dtype=float)

    winner_idx = None
    if "Winner" in eval_df.columns and (eval_df["Winner"] == 1).any():
        winner_idx = int(np.flatnonzero((eval_df["Winner"] == 1).to_numpy())[0])

    for b in range(n_boot):
        boot = _stratified_bootstrap(fit_df, "Season", rng)
        X_tr = boot[spec.features].to_numpy()
        y_tr = _transform_y(boot[TARGET_COL].to_numpy(), spec.y_transform)

        model = build_ridge(alpha=alpha)
        model.fit(X_tr, y_tr)

        score = model.predict(X_eval)
        share = softmax_share(score, temperature=temperature)
        shares[b, :] = share

        # Use 'min' so ties count as top-k hits.
        ranks = pd.Series(-share).rank(method="min").to_numpy()
        p_rank1[b, :] = (ranks == 1).astype(float)
        p_rank3[b, :] = (ranks <= 3).astype(float)
        p_rank5[b, :] = (ranks <= 5).astype(float)

        s_sorted = np.sort(share)[::-1]
        top1_gaps[b] = float(s_sorted[0] - s_sorted[1]) if len(s_sorted) >= 2 else float("nan")

        if winner_idx is not None:
            win_share = float(share[winner_idx])
            best_other = float(np.max(np.delete(share, winner_idx))) if len(share) > 1 else 0.0
            winner_gaps[b] = win_share - best_other
        else:
            winner_gaps[b] = float("nan")

    qlo, qhi = ci_q
    out = eval_df[["Name", "Season", "Voting Points", "Winner", TARGET_COL]].copy()
    out = out.rename(columns={TARGET_COL: "ActualVoteShare"})
    out["Temperature"] = float(temperature)
    out["PredVoteShare_mean"] = shares.mean(axis=0)
    out[f"PredVoteShare_p{int(qlo*100):02d}"] = np.quantile(shares, qlo, axis=0)
    out[f"PredVoteShare_p{int(qhi*100):02d}"] = np.quantile(shares, qhi, axis=0)
    out["P_rank1"] = p_rank1.mean(axis=0)
    out["P_rank<=3"] = p_rank3.mean(axis=0)
    out["P_rank<=5"] = p_rank5.mean(axis=0)
    out = out.sort_values(by="PredVoteShare_mean", ascending=False).reset_index(drop=True)

    closeness = pd.DataFrame(
        [
            {
                "season": str(eval_df["Season"].iloc[0]) if len(eval_df) else None,
                "model": spec.name,
                "n_boot": int(n_boot),
                "temperature": float(temperature),
                "top1_gap_median": float(np.nanmedian(top1_gaps)),
                f"top1_gap_p{int(qlo*100):02d}": float(np.nanquantile(top1_gaps, qlo)),
                f"top1_gap_p{int(qhi*100):02d}": float(np.nanquantile(top1_gaps, qhi)),
                "p_top1_gap_gt_0p02": float(np.nanmean(top1_gaps > 0.02)),
                "winner_gap_median": float(np.nanmedian(winner_gaps)),
                f"winner_gap_p{int(qlo*100):02d}": float(np.nanquantile(winner_gaps, qlo)),
                f"winner_gap_p{int(qhi*100):02d}": float(np.nanquantile(winner_gaps, qhi)),
                "p_winner_rank1": float(np.nanmean(p_rank1[:, winner_idx])) if winner_idx is not None else float("nan"),
            }
        ]
    )

    return out, closeness, top1_gaps


def plot_predshare_ci(
    unc_df: pd.DataFrame, out_path: Path, title: str, qlo: float, qhi: float, top_k: int = 12
) -> None:
    plot_df = unc_df.head(top_k).copy()
    x = np.arange(len(plot_df))
    y = plot_df["PredVoteShare_mean"].to_numpy()
    lo = plot_df[f"PredVoteShare_p{int(qlo*100):02d}"].to_numpy()
    hi = plot_df[f"PredVoteShare_p{int(qhi*100):02d}"].to_numpy()
    yerr = np.vstack([y - lo, hi - y])

    plt.figure(figsize=(10, 5))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xticks(x, plot_df["Name"], rotation=45, ha="right")
    plt.ylabel("Predicted Vote Share (Outfield)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_top1_gap_hist(top1_gaps: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(7, 4))
    v = top1_gaps[np.isfinite(top1_gaps)]
    plt.hist(v, bins=30, alpha=0.9)
    plt.axvline(0.02, color="black", linestyle="--", linewidth=1, label="0.02 share")
    plt.xlabel("Top-1 minus Top-2 predicted vote share")
    plt.ylabel("Bootstrap count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = ensure_vote_share_outfield(df)
    df = add_per90_variants(df)
    df = add_baseline_pred_share(df)
    df = add_similarity_baseline(df)

    # Minimal hygiene: drop rows missing any required feature or target.
    # (Better long-term: explicit imputation, but with tiny n it can hide issues.)
    perf_all_per90_features = [
        # Core per90 performance
        "Per 90 Minutes NpxG+xAG",
        "Per 90 Minutes G+A-PK",
        "Per 90 Minutes XAG",
        "SCA SCA90",
        # Engineered per90 creation/progression (from totals)
        "KP90",
        "TB90",
        "Carries13_90",
        "PrgC90",
        "PrgP90",
        # Durability / availability (explicit)
        "Playing Time 90s",
        "Starts Starts",
        "Playing Time Mn/MP",
    ]
    perf_min_durability_features = [
        "Per 90 Minutes NpxG+xAG",
        "Per 90 Minutes G+A-PK",
        "Per 90 Minutes XAG",
        "SCA SCA90",
        "KP90",
        "TB90",
        "Carries13_90",
        "PrgC90",
        "PrgP90",
        "Playing Time 90s",
    ]

    model_specs = [
        # Original draft (kept for comparison; mixes per90 + totals)
        ModelSpec(name="performance_only", features=CORE_FEATURES),
        # Ablations to resolve performance-only story
        ModelSpec(name="performance_only_all_per90_durability", features=perf_all_per90_features),
        ModelSpec(name="performance_only_min_durability", features=perf_min_durability_features),
        ModelSpec(name="performance_only_logtarget", features=perf_all_per90_features, y_transform="log_share"),
        ModelSpec(
            name="performance_only_simple_metric_minutes",
            features=["Per 90 Minutes NpxG+xAG", "Playing Time 90s"],
        ),
        # Voter-behavior model
        ModelSpec(name="with_trophies", features=CORE_FEATURES + TROPHY_COLS),
    ]

    results_metrics = []
    alpha_grid = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    train_df = df[df["Season"].isin(TRAIN_SEASONS)].copy()
    val_df = df[df["Season"] == VAL_SEASON].copy()
    test_df = df[df["Season"] == TEST_SEASON].copy()

    # Baseline metrics for context (no training).
    if "Pred baseline_npxg_xag90" in df.columns:
        for season in [VAL_SEASON, TEST_SEASON]:
            season_df = df[df["Season"] == season].copy()
            results_metrics.append(
                {
                    **season_metrics(season_df, "Pred baseline_npxg_xag90"),
                    "model": "baseline_npxg_xag90",
                    "alpha": np.nan,
                    "split": "val" if season == VAL_SEASON else "test",
                }
            )

        # Save a simple 24-25 baseline table
        season_df = df[df["Season"] == TEST_SEASON][
            [
                "Name",
                "Season",
                "Voting Points",
                "Winner",
                TARGET_COL,
                "Pred baseline_npxg_xag90",
                "PredScore baseline_npxg_xag90",
                BASELINE_METRIC_COL,
            ]
        ].copy()
        season_df = season_df.rename(
            columns={
                TARGET_COL: "ActualVoteShare",
                "Pred baseline_npxg_xag90": "PredVoteShare",
                "PredScore baseline_npxg_xag90": "PredScore",
            }
        )
        season_df["ActualRank"] = (-season_df["ActualVoteShare"]).rank(method="average")
        season_df["PredRank"] = (-season_df["PredVoteShare"]).rank(method="average")
        season_df["Residual"] = season_df["PredVoteShare"] - season_df["ActualVoteShare"]
        season_df = season_df.sort_values(by="PredRank", ascending=True)
        season_df.to_csv(OUT_TABLE_DIR / "part4_2024_25_predictions_baseline_npxg_xag90.csv", index=False)

    # Baseline 2 metrics (similarity score, no training).
    if "Pred baseline_similarity" in df.columns:
        for season in [VAL_SEASON, TEST_SEASON]:
            season_df = df[df["Season"] == season].copy()
            results_metrics.append(
                {
                    **season_metrics(season_df, "Pred baseline_similarity"),
                    "model": "baseline_similarity",
                    "alpha": np.nan,
                    "split": "val" if season == VAL_SEASON else "test",
                }
            )

        season_df = df[df["Season"] == TEST_SEASON][
            ["Name", "Season", "Voting Points", "Winner", TARGET_COL, "Pred baseline_similarity", "PredScore baseline_similarity"]
        ].copy()
        season_df = season_df.rename(
            columns={
                TARGET_COL: "ActualVoteShare",
                "Pred baseline_similarity": "PredVoteShare",
                "PredScore baseline_similarity": "PredScore",
            }
        )
        season_df["ActualRank"] = (-season_df["ActualVoteShare"]).rank(method="average")
        season_df["PredRank"] = (-season_df["PredVoteShare"]).rank(method="average")
        season_df["Residual"] = season_df["PredVoteShare"] - season_df["ActualVoteShare"]
        season_df = season_df.sort_values(by="PredRank", ascending=True)
        season_df.to_csv(OUT_TABLE_DIR / "part4_2024_25_predictions_baseline_similarity.csv", index=False)

    best_alpha_by_model: dict[str, float] = {}
    best_temp_by_model: dict[str, float] = {}

    for spec in model_specs:
        _require_cols(df, ["Name", "Season", "Voting Points", TARGET_COL] + spec.features)

        use_cols = ["Name", "Season", "Winner", "Voting Points", TARGET_COL] + spec.features
        train_use = train_df[use_cols].dropna()
        val_use = val_df[use_cols].dropna()
        test_use = test_df[use_cols].dropna()

        best_alpha, diag = fit_choose_alpha(train_use, val_use, spec, alpha_grid)
        best_alpha_by_model[spec.name] = float(best_alpha)
        diag.to_csv(OUT_TABLE_DIR / f"part4_alpha_search_{spec.name}.csv", index=False)

        # True validation evaluation: train on 21-23, evaluate on 23-24 (no leakage).
        train_model = build_ridge(alpha=best_alpha)
        train_model.fit(
            train_use[spec.features].to_numpy(),
            _transform_y(train_use[TARGET_COL].to_numpy(), spec.y_transform),
        )
        val_score = train_model.predict(val_use[spec.features].to_numpy())
        best_temp, temp_diag = calibrate_temperature_on_season(
            scores=val_score,
            y_true_share=val_use[TARGET_COL].to_numpy(),
        )
        best_temp_by_model[spec.name] = float(best_temp)
        if len(temp_diag):
            temp_diag.insert(0, "model", spec.name)
            temp_diag.to_csv(OUT_TABLE_DIR / f"part4_temperature_search_{spec.name}.csv", index=False)

        val_eval = val_use.copy()
        val_eval[f"PredScore {spec.name}"] = val_score
        val_eval = add_pred_share_for_single_season(
            val_eval,
            f"PredScore {spec.name}",
            f"PredVoteShare {spec.name}",
            temperature=best_temp,
        )
        results_metrics.append(
            {
                **season_metrics(val_eval.assign(Season=VAL_SEASON), f"PredVoteShare {spec.name}"),
                "model": spec.name,
                "alpha": best_alpha,
                "temperature": float(best_temp),
                "split": "val",
            }
        )

        # Lock alpha based on 23-24; then refit on train+val and evaluate 24-25 once.
        fit_df = pd.concat([train_use, val_use], axis=0, ignore_index=True)
        final_model = build_ridge(alpha=best_alpha)
        final_model.fit(
            fit_df[spec.features].to_numpy(),
            _transform_y(fit_df[TARGET_COL].to_numpy(), spec.y_transform),
        )

        test_score = final_model.predict(test_use[spec.features].to_numpy())
        test_eval = test_use.copy()
        test_eval[f"PredScore {spec.name}"] = test_score
        test_eval = add_pred_share_for_single_season(
            test_eval,
            f"PredScore {spec.name}",
            f"PredVoteShare {spec.name}",
            temperature=best_temp,
        )
        results_metrics.append(
            {
                **season_metrics(test_eval.assign(Season=TEST_SEASON), f"PredVoteShare {spec.name}"),
                "model": spec.name,
                "alpha": best_alpha,
                "temperature": float(best_temp),
                "split": "test",
            }
        )

        # Plots (val plot uses train-only model; test plot uses final model)
        scatter_plot(
            val_eval.assign(Season=VAL_SEASON),
            pred_col=f"PredVoteShare {spec.name}",
            out_path=OUT_GRAPH_DIR / f"part4_pred_vs_actual_vote_share_2324_{spec.name}.png",
            title=f"Predicted vs Actual Vote Share (Outfield) — {VAL_SEASON} ({spec.name})",
        )
        scatter_plot(
            test_eval.assign(Season=TEST_SEASON),
            pred_col=f"PredVoteShare {spec.name}",
            out_path=OUT_GRAPH_DIR / f"part4_pred_vs_actual_vote_share_2425_{spec.name}.png",
            title=f"Predicted vs Actual Vote Share (Outfield) — {TEST_SEASON} ({spec.name})",
        )

        top10_bar_pred_vs_actual(
            test_eval.assign(Season=TEST_SEASON),
            pred_col=f"PredVoteShare {spec.name}",
            out_path=OUT_GRAPH_DIR / f"part4_top10_pred_vs_actual_2425_{spec.name}.png",
            title=f"Top-10 predicted vs actual vote share — {TEST_SEASON} ({spec.name})",
        )
        residuals_plot(
            test_eval.assign(Season=TEST_SEASON),
            pred_col=f"PredVoteShare {spec.name}",
            out_path=OUT_GRAPH_DIR / f"part4_residuals_over_under_2425_{spec.name}.png",
            title=f"Most over/under-rated (Pred − Actual) — {TEST_SEASON} ({spec.name})",
        )

        # Interpretability artifacts (coefficients + permutation importance on validation season).
        if spec.name in {"performance_only", "with_trophies"}:
            write_model_interpretability(
                model=train_model,
                features=spec.features,
                eval_df=val_use,
                out_prefix=f"part4_{spec.name}",
            )

        # Save 24-25 prediction table (one per model spec)
        out = test_eval[
            [
                "Name",
                "Season",
                "Voting Points",
                "Winner",
                TARGET_COL,
                f"PredScore {spec.name}",
                f"PredVoteShare {spec.name}",
            ]
        ].copy()
        out = out.rename(
            columns={
                TARGET_COL: "ActualVoteShare",
                f"PredScore {spec.name}": "PredScore",
                f"PredVoteShare {spec.name}": "PredVoteShare",
            }
        )
        out["Temperature"] = float(best_temp)
        out["ActualRank"] = (-out["ActualVoteShare"]).rank(method="average")
        out["PredRank"] = (-out["PredVoteShare"]).rank(method="average")
        out["Residual"] = out["PredVoteShare"] - out["ActualVoteShare"]
        out = out.sort_values(by="PredRank", ascending=True)
        out.to_csv(OUT_TABLE_DIR / f"part4_2024_25_predictions_{spec.name}.csv", index=False)

    # Compact “appendix credibility” metric: leave-one-season-out (LOSO) on 4 seasons.
    # Note: we keep it simple and only evaluate the disciplined performance-only spec.
    loso_spec = model_specs[0]
    _require_cols(df, ["Name", "Season", "Voting Points", TARGET_COL] + loso_spec.features)

    for holdout in sorted(df["Season"].unique()):
        train_loso = df[df["Season"] != holdout][["Name", "Season", "Winner", "Voting Points", TARGET_COL] + loso_spec.features].dropna()
        holdout_df = df[df["Season"] == holdout][["Name", "Season", "Winner", "Voting Points", TARGET_COL] + loso_spec.features].dropna()
        if len(holdout_df) == 0 or len(train_loso) == 0:
            continue
        X_tr = train_loso[loso_spec.features].to_numpy()
        y_tr = _transform_y(train_loso[TARGET_COL].to_numpy(), loso_spec.y_transform)
        model = build_ridge(alpha=1.0)  # fixed for appendix stability
        model.fit(X_tr, y_tr)
        score = model.predict(holdout_df[loso_spec.features].to_numpy())
        holdout_df = holdout_df.copy()
        holdout_df["PredScore loso_performance_only"] = score
        holdout_df = add_pred_share_for_single_season(
            holdout_df, "PredScore loso_performance_only", "Pred loso_performance_only", temperature=1.0
        )
        results_metrics.append(
            {
                **season_metrics(holdout_df, "Pred loso_performance_only"),
                "model": "loso_performance_only",
                "alpha": 1.0,
                "temperature": 1.0,
                "split": "loso",
            }
        )

    metrics_df = pd.DataFrame(results_metrics)
    metrics_df.to_csv(OUT_TABLE_DIR / "part4_metrics_by_season.csv", index=False)

    # Report-friendly model comparison (val/test only)
    comp = metrics_df[metrics_df["split"].isin(["val", "test"])].copy()
    comp = comp.sort_values(by=["split", "spearman"], ascending=[True, False])
    comp.to_csv(OUT_TABLE_DIR / "part4_model_comparison.csv", index=False)

    # Bootstrap uncertainty for 2024-25:
    # always include with_trophies, plus the best performance-only variant by validation Spearman.
    val_perf = metrics_df[(metrics_df["split"] == "val") & (metrics_df["model"].str.startswith("performance_only"))].copy()
    best_perf_name = (
        str(val_perf.sort_values("spearman", ascending=False).iloc[0]["model"]) if len(val_perf) else "performance_only"
    )

    for model_name in [best_perf_name, "with_trophies"]:
        spec = next((s for s in model_specs if s.name == model_name), None)
        if spec is None:
            continue
        alpha = float(best_alpha_by_model.get(spec.name, 1.0))
        temperature = float(best_temp_by_model.get(spec.name, 1.0))

        fit_df = df[df["Season"].isin(TRAIN_SEASONS + [VAL_SEASON])][
            ["Name", "Season", "Winner", "Voting Points", TARGET_COL] + spec.features
        ].dropna()
        eval_df = df[df["Season"] == TEST_SEASON][
            ["Name", "Season", "Winner", "Voting Points", TARGET_COL] + spec.features
        ].dropna()

        unc, closeness, top1_gaps = bootstrap_uncertainty_for_season(
            fit_df=fit_df,
            eval_df=eval_df,
            spec=spec,
            alpha=alpha,
            temperature=temperature,
            n_boot=BOOTSTRAP_N,
            seed=BOOTSTRAP_SEED,
            ci_q=BOOTSTRAP_CI_Q,
        )
        unc.to_csv(OUT_TABLE_DIR / f"part4_2024_25_uncertainty_{spec.name}.csv", index=False)
        closeness.to_csv(OUT_TABLE_DIR / f"part4_2024_25_closeness_{spec.name}.csv", index=False)

        qlo, qhi = BOOTSTRAP_CI_Q
        plot_predshare_ci(
            unc_df=unc,
            out_path=OUT_GRAPH_DIR / f"part4_2024_25_predshare_ci_{spec.name}.png",
            title=f"Predicted vote share with {int((qhi-qlo)*100)}% CI — {TEST_SEASON} ({spec.name})",
            qlo=qlo,
            qhi=qhi,
        )
        plot_top1_gap_hist(
            top1_gaps=top1_gaps,
            out_path=OUT_GRAPH_DIR / f"part4_2024_25_top1_gap_bootstrap_{spec.name}.png",
            title=f"How close was it? Top1−Top2 gap bootstrap — {TEST_SEASON} ({spec.name})",
        )


if __name__ == "__main__":
    main()


