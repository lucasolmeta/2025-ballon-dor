"""
Optional helper: regenerate `data/ballon_dor_votes.csv` from Wikipedia.

Why optional?
- Wikipedia page structure can change.
- For reproducibility, this repo treats `data/ballon_dor_votes.csv` as a frozen snapshot
  (the source of truth). This script is only a convenience if you want to refresh it.

Usage (from repo root):
  python -m setup_scripts.scrape_votes_wikipedia --out data/ballon_dor_votes.csv

Notes:
- This script intentionally stays conservative and may require small tweaks if Wikipedia
  table headers differ (e.g., "Points" vs "Voting points").
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from unidecode import unidecode


DEFAULT_OUT = Path("data/ballon_dor_votes.csv")

# These are placeholders; adjust if we want to use this to scrape different pages/years.
WIKI_PAGES = [
    # Example pages (verify titles if we want to use this):
    # ("2021-2022", "https://en.wikipedia.org/wiki/2022_Ballon_d%27Or"),
    # ("2022-2023", "https://en.wikipedia.org/wiki/2023_Ballon_d%27Or"),
    # ("2023-2024", "https://en.wikipedia.org/wiki/2024_Ballon_d%27Or"),
    # ("2024-2025", "https://en.wikipedia.org/wiki/2025_Ballon_d%27Or"),
]


def normalize_name(name: str) -> str:
    name = unidecode(str(name))
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^\w\s-]", "", name)
    return name.strip()


def find_votes_table(soup: BeautifulSoup) -> pd.DataFrame:
    """
    Find the first wikitable that looks like a vote/points table and return it as a dataframe.
    This is heuristic; we may need to adjust for specific pages.
    """
    tables = soup.select("table.wikitable")
    if not tables:
        raise RuntimeError("No wikitable found on page.")

    for tbl in tables:
        df_list = pd.read_html(str(tbl))
        if not df_list:
            continue
        df = df_list[0]
        cols = {str(c).strip().lower() for c in df.columns}
        if any("point" in c for c in cols) and any("player" in c or "name" in c for c in cols):
            return df

    # fallback: just try the first
    return pd.read_html(str(tables[0]))[0]


def extract_votes(season: str, url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    raw = find_votes_table(soup)

    # Try to identify name + points columns.
    cols = {str(c).strip().lower(): c for c in raw.columns}
    name_col = None
    points_col = None
    for k, orig in cols.items():
        if name_col is None and ("player" in k or "name" in k):
            name_col = orig
        if points_col is None and ("point" in k):
            points_col = orig

    if name_col is None or points_col is None:
        raise RuntimeError(f"Could not find name/points columns in table columns: {list(raw.columns)}")

    out = pd.DataFrame(
        {
            "Name": raw[name_col].astype(str).map(normalize_name),
            "Season": season,
            "Finalist": 1,
            "Voting Points": pd.to_numeric(raw[points_col], errors="coerce").fillna(0).astype(int),
        }
    )

    # Winner flag from max points
    out["Winner"] = 0
    if len(out):
        out.loc[out["Voting Points"] == out["Voting Points"].max(), "Winner"] = 1

    # Trophy flags are not reliably derivable from Wikipedia here; leave as 0.
    out["League Winner"] = 0
    out["UCL Winner"] = 0
    out["Cup Winner"] = 0
    out["Major International Continental Trophy Winner"] = 0
    out["World Cup Winner"] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_path = Path(args.out)

    if not WIKI_PAGES:
        raise SystemExit(
            "WIKI_PAGES is empty in setup_scripts/scrape_votes_wikipedia.py. "
            "Populate it with (season, url) pairs before running."
        )

    frames = [extract_votes(season, url) for season, url in WIKI_PAGES]
    df = pd.concat(frames, ignore_index=True)

    # Stable column order (matches preprocess expectations)
    cols = [
        "Name",
        "Season",
        "Finalist",
        "Voting Points",
        "Winner",
        "League Winner",
        "UCL Winner",
        "Cup Winner",
        "Major International Continental Trophy Winner",
        "World Cup Winner",
    ]
    df = df[cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()


