#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import os
import re
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd


MSI_TAR_URL = "http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz"


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[OK] Found existing: {out_path}")
        return
    print(f"[DL] {url}")
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())
    print(f"[OK] Saved: {out_path} ({out_path.stat().st_size} bytes)")


def extract(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted_ok"
    if marker.exists():
        print(f"[OK] Already extracted: {out_dir}")
        return
    print(f"[EXTRACT] {tar_path} -> {out_dir}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=out_dir)
    marker.write_text("ok\n", encoding="utf-8")
    print(f"[OK] Extracted.")


def sniff_delimiter(sample: str) -> str:
    # simple heuristic: prefer tab if present, else comma
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    return "\t"


def try_read_table(path: Path) -> Optional[pd.DataFrame]:
    # Try tsv/csv with header; if fails, try no-header.
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:2000]
        sep = sniff_delimiter(head)
        df = pd.read_csv(path, sep=sep)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass

    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:2000]
        sep = sniff_delimiter(head)
        df = pd.read_csv(path, sep=sep, header=None)
        if df.shape[1] >= 2:
            return df
    except Exception:
        return None

    return None


def find_approved_pairs_file(extracted_root: Path) -> Path:
    """
    MSI tarball structure can change. We do:
    1) search by filename keywords
    2) if multiple candidates, pick the smallest "reasonable" table that has >=2 columns
    """
    patterns = [
        re.compile(r"approved", re.I),
        re.compile(r"treat", re.I),
        re.compile(r"drug.*disease", re.I),
        re.compile(r"disease.*drug", re.I),
        re.compile(r"indication", re.I),
        re.compile(r"drug_disease", re.I),
    ]

    candidates: List[Path] = []
    for p in extracted_root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if p.suffix.lower() not in {".tsv", ".csv", ".txt"}:
            continue
        score = sum(1 for pat in patterns if pat.search(name))
        if score >= 2:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "Could not locate an 'approved drug–disease pairs' file automatically. "
            "Please pass --pairs_file to point to the dataset #6 file inside the extracted MSI folder."
        )

    # choose best candidate by: has >=2 cols and contains drug/disease-ish column names if any
    best: Tuple[int, int, Path] = (-1, 10**18, candidates[0])  # (name_score, file_size, path)
    for p in candidates:
        name_score = 0
        if "approved" in p.name.lower():
            name_score += 3
        if "drug" in p.name.lower() and "disease" in p.name.lower():
            name_score += 3
        if "treat" in p.name.lower() or "indication" in p.name.lower():
            name_score += 2

        df = try_read_table(p)
        if df is None or df.shape[1] < 2:
            continue

        size = p.stat().st_size
        # prefer higher name_score; if tie, prefer smaller file to avoid huge matrices
        if (name_score > best[0]) or (name_score == best[0] and size < best[1]):
            best = (name_score, size, p)

    print(f"[OK] Approved-pairs candidate: {best[2]}")
    return best[2]


def normalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to normalize into columns: disease, drug
    If headers exist, try to pick by name; else take first two columns.
    """
    cols = [c.lower() for c in df.columns.astype(str)]
    disease_col = None
    drug_col = None

    for i, c in enumerate(cols):
        if disease_col is None and ("disease" in c or "indication" in c):
            disease_col = df.columns[i]
        if drug_col is None and ("drug" in c):
            drug_col = df.columns[i]

    if disease_col is None or drug_col is None:
        # fallback: first two columns
        disease_col = df.columns[0]
        drug_col = df.columns[1]

    out = df[[disease_col, drug_col]].copy()
    out.columns = ["disease", "drug"]
    out = out.dropna()
    out["disease"] = out["disease"].astype(str)
    out["drug"] = out["drug"].astype(str)
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def split_train_val_test(pairs: pd.DataFrame, seed: int, val_ratio: float, test_ratio: float):
    rng = pd.Series(range(len(pairs))).sample(frac=1.0, random_state=seed).values
    n = len(pairs)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_idx = rng[:n_test]
    val_idx = rng[n_test:n_test + n_val]
    train_idx = rng[n_test + n_val:]

    return pairs.iloc[train_idx].reset_index(drop=True), pairs.iloc[val_idx].reset_index(drop=True), pairs.iloc[test_idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msi_url", default=MSI_TAR_URL)
    ap.add_argument("--raw_dir", default="data/raw/msi")
    ap.add_argument("--out_dir", default="data/processed/msi")
    ap.add_argument("--pairs_file", default=None, help="Optional: path to the dataset #6 file inside extracted MSI folder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    tar_path = raw_dir / "data.tar.gz"
    extracted_root = raw_dir / "extracted"

    download(args.msi_url, tar_path)
    extract(tar_path, extracted_root)

    if args.pairs_file:
        pairs_path = Path(args.pairs_file)
        if not pairs_path.is_absolute():
            pairs_path = extracted_root / pairs_path
    else:
        pairs_path = find_approved_pairs_file(extracted_root)

    df = try_read_table(pairs_path)
    if df is None:
        raise RuntimeError(f"Failed to read table: {pairs_path}")

    pairs = normalize_pairs(df)
    print(f"[INFO] Loaded approved pairs: {len(pairs)}")

    train, val, test = split_train_val_test(pairs, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train.tsv", sep="\t", index=False)
    val.to_csv(split_dir / "val.tsv", sep="\t", index=False)
    test.to_csv(split_dir / "test.tsv", sep="\t", index=False)

    # Also write a short note for provenance
    (split_dir / "SOURCE.txt").write_text(
        f"MSI source: https://github.com/snap-stanford/multiscale-interactome\n"
        f"Downloaded from: {args.msi_url}\n"
        f"Label set: dataset #6 (approved drug–disease pairs)\n"
        f"Seed={args.seed}, val_ratio={args.val_ratio}, test_ratio={args.test_ratio}\n"
        f"Pairs file used: {pairs_path}\n",
        encoding="utf-8",
    )

    print(f"[OK] Wrote splits to: {split_dir}")


if __name__ == "__main__":
    main()
