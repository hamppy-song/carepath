#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

MSI_TAR_URL = "http://snap.stanford.edu/multiscale-interactome/data/data.tar.gz"
MSI_REPO = "https://github.com/snap-stanford/multiscale-interactome"

EXTS = {".tsv", ".csv", ".txt"}


def dl(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"[OK] {out} exists")
        return
    print(f"[DL] {url}")
    with urllib.request.urlopen(url) as r, open(out, "wb") as f:
        f.write(r.read())
    print(f"[OK] saved {out}")


def extract(tar_gz: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted_ok"
    if marker.exists():
        return
    with tarfile.open(tar_gz, "r:gz") as tf:
        tf.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def score(name: str, keys: Iterable[str]) -> int:
    s = name.lower()
    return sum(1 for k in keys if k in s)


def find_best(root: Path, keys: List[str], min_score: int = 2) -> Path:
    best = None
    best_score = -1
    best_size = -1
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        sc = score(p.name, keys)
        if sc < min_score:
            continue
        sz = p.stat().st_size
        if (sc > best_score) or (sc == best_score and sz > best_size):
            best, best_score, best_size = p, sc, sz
    if best is None:
        raise FileNotFoundError(f"Could not find file for keys={keys} under {root}")
    return best


def sniff_sep(p: Path) -> str:
    head = p.read_text(encoding="utf-8", errors="ignore")[:3000]
    if "\t" in head:
        return "\t"
    if "," in head:
        return ","
    return "\t"


def read_table(p: Path) -> pd.DataFrame:
    sep = sniff_sep(p)
    try:
        df = pd.read_csv(p, sep=sep)
        if df.shape[1] >= 2:
            return df
    except Exception:
        pass
    df = pd.read_csv(p, sep=sep, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Not a 2+ col table: {p}")
    return df


def pick_two_cols(df: pd.DataFrame, want_left: List[str], want_right: List[str]) -> Tuple[pd.Series, pd.Series]:
    cols = [str(c).lower() for c in df.columns]
    li = ri = None
    for i, c in enumerate(cols):
        if li is None and any(k in c for k in want_left):
            li = i
        if ri is None and any(k in c for k in want_right):
            ri = i
    if li is None or ri is None or li == ri:
        return df.iloc[:, 0], df.iloc[:, 1]
    return df.iloc[:, li], df.iloc[:, ri]


def edges_from(df: pd.DataFrame, etype: int, directed_flag: int = 0, weight: float = 1.0) -> List[Tuple[str, str, int, float, int]]:
    a, b = df.iloc[:, 0], df.iloc[:, 1]
    out = []
    for u, v in zip(a, b):
        u = str(u).strip()
        v = str(v).strip()
        if not u or not v or u == "nan" or v == "nan":
            continue
        out.append((u, v, etype, float(weight), int(directed_flag)))
    return out


def normalize_pairs_6(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).lower() for c in df.columns]
    drug_col = None
    dis_col = None
    for i, c in enumerate(cols):
        if drug_col is None and c == "drug":
            drug_col = df.columns[i]
        if dis_col is None and (c == "disease" or "disease" in c):
            dis_col = df.columns[i]
    if drug_col is None or dis_col is None:
        if df.shape[1] >= 3:
            drug_col, dis_col = df.columns[0], df.columns[2]
        else:
            drug_col, dis_col = df.columns[0], df.columns[1]

    pairs = df[[drug_col, dis_col]].copy()
    pairs.columns = ["drug", "disease"]
    pairs = pairs.dropna()
    pairs["drug"] = pairs["drug"].astype(str).str.strip()
    pairs["disease"] = pairs["disease"].astype(str).str.strip()
    pairs = pairs[(pairs["drug"] != "") & (pairs["disease"] != "")]
    pairs = pairs.drop_duplicates().reset_index(drop=True)
    return pairs


def sample_negatives(pos_pairs: Set[Tuple[str, str]], drugs: List[str], diseases: List[str], ratio: float, seed: int) -> Set[Tuple[str, str]]:
    target = int(round(len(pos_pairs) * ratio))
    rng = random.Random(seed)
    neg = set()
    tries = 0
    max_tries = max(10000, target * 50)
    while len(neg) < target and tries < max_tries:
        tries += 1
        d = rng.choice(drugs)
        s = rng.choice(diseases)
        if (d, s) in pos_pairs:
            continue
        neg.add((d, s))
    return neg


def write_graph(path: Path, rows: List[Tuple[str, str, int, float, int]]) -> None:
    # 모델의 read_graph(..., delimiter=' ')에 맞춰 공백 구분으로 저장
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for u, v, t, w, dr in rows:
            f.write(f"{u} {v} {t} {w} {dr}\n")


def write_nodetypes(path: Path, mapping: Dict[str, str]) -> None:
    # 모델의 load_node_types(... header=0)에 맞춰 헤더 추가
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("node\ttype\n")
        for n, ty in mapping.items():
            f.write(f"{n}\t{ty}\n")


def write_atc(path: Path, atc_src: Path) -> None:
    # 모델이 읽는 컬럼명(db_id, atc_code)으로 정규화해서 ATC(#7) 출력
    path.parent.mkdir(parents=True, exist_ok=True)
    df = read_table(atc_src)
    cols = [str(c).lower() for c in df.columns]

    id_i = next((i for i, c in enumerate(cols) if "db" in c or "drug" in c or "id" in c), 0)
    atc_i = next((i for i, c in enumerate(cols) if "atc" in c), 1)

    out = df.iloc[:, [id_i, atc_i]].copy()
    out.columns = ["db_id", "atc_code"]
    out["db_id"] = out["db_id"].astype(str).str.strip()
    out["atc_code"] = out["atc_code"].astype(str).str.strip()
    out = out.replace({"db_id": {"nan": None}, "atc_code": {"nan": None}})
    out = out.dropna().drop_duplicates().reset_index(drop=True)
    out.to_csv(path, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msi_url", default=MSI_TAR_URL)
    ap.add_argument("--raw_dir", default="data/raw/msi")
    ap.add_argument("--out_dir", default="dataset")
    ap.add_argument("--download", action="store_true", help="Download MSI tarball (default: assumes already present)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg_ratio", type=float, default=1.0, help="negatives per positive (random sampling)")
    ap.add_argument("--files_json", default=None, help="Optional: JSON mapping to pin exact files inside extracted/")

    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    tar_path = raw_dir / "data.tar.gz"
    ex_dir = raw_dir / "extracted"

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        dl(args.msi_url, tar_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing {tar_path}. Use --download or place it manually. Source: {MSI_REPO}")

    extract(tar_path, ex_dir)

    if args.files_json:
        spec = json.loads(Path(args.files_json).read_text(encoding="utf-8"))
        f1 = ex_dir / spec["drug_protein"]
        f2 = ex_dir / spec["disease_protein"]
        f3 = ex_dir / spec["protein_protein"]
        f4 = ex_dir / spec["protein_function"]
        f5 = ex_dir / spec["function_function"]
        f6 = ex_dir / spec["approved_pairs"]
        f7 = ex_dir / spec.get("atc", "") if spec.get("atc") else None
    else:
        f1 = find_best(ex_dir, ["drug", "protein"])
        f2 = find_best(ex_dir, ["disease", "protein"])
        f3 = find_best(ex_dir, ["protein", "protein"])
        f4 = find_best(ex_dir, ["protein", "function"])
        f5 = find_best(ex_dir, ["function", "function"])
        f6 = find_best(ex_dir, ["approved", "drug", "disease"], min_score=3)
        # ATC (optional)
        try:
            f7 = find_best(ex_dir, ["atc"], min_score=1)
        except Exception:
            f7 = None

    edges: List[Tuple[str, str, int, float, int]] = []
    drugs: Set[str] = set()
    diseases: Set[str] = set()
    proteins: Set[str] = set()
    functions: Set[str] = set()

    df1 = read_table(f1); edges += edges_from(df1, 1)
    df2 = read_table(f2); edges += edges_from(df2, 2)
    df3 = read_table(f3); edges += edges_from(df3, 3)
    df4 = read_table(f4); edges += edges_from(df4, 4)
    df5 = read_table(f5); edges += edges_from(df5, 5)

    for u, v, t, _, _ in edges:
        if t == 1:
            drugs.add(u); proteins.add(v)
        elif t == 2:
            diseases.add(u); proteins.add(v)
        elif t == 3:
            proteins.add(u); proteins.add(v)
        elif t == 4:
            proteins.add(u); functions.add(v)
        elif t == 5:
            functions.add(u); functions.add(v)

    write_graph(out_dir / "graph.txt", edges)

    nodemap: Dict[str, str] = {}
    for x in drugs:
        nodemap[x] = "drug"
    for x in diseases:
        nodemap[x] = "disease"
    for x in proteins:
        nodemap.setdefault(x, "gene")
    for x in functions:
        nodemap.setdefault(x, "function")
    write_nodetypes(out_dir / "nodetypes.tsv", nodemap)

    if f7 is not None:
        write_atc(out_dir / "7_drug_classification_df.tsv", f7)

    df6 = read_table(f6)
    pos = normalize_pairs_6(df6)
    pos_set = set((r["drug"], r["disease"]) for _, r in pos.iterrows())

    d_list = sorted(list(drugs if drugs else set(pos["drug"].tolist())))
    s_list = sorted(list(diseases if diseases else set(pos["disease"].tolist())))
    neg_set = sample_negatives(pos_set, d_list, s_list, ratio=args.neg_ratio, seed=args.seed)

    lab = []
    for d, s in pos_set:
        lab.append((d, s, 1))
    for d, s in neg_set:
        lab.append((d, s, 0))

    lab_df = pd.DataFrame(lab, columns=["drug", "disease", "label"]).drop_duplicates()
    (out_dir / "dda_labels.tsv").parent.mkdir(parents=True, exist_ok=True)
    lab_df.to_csv(out_dir / "dda_labels.tsv", sep="\t", index=False)

    (out_dir / "SOURCE.txt").write_text(
        "MSI source:\n"
        f"{MSI_REPO}\n\n"
        "Downloaded tarball:\n"
        f"{args.msi_url}\n\n"
        "Used supplementary datasets #1-#7 (1-5 graph, 6 labels, 7 ATC).\n"
        "Outputs: graph.txt, nodetypes.tsv, dda_labels.tsv, 7_drug_classification_df.tsv\n",
        encoding="utf-8",
    )

    print("[OK] Done.")
    print(f"  graph:      {out_dir / 'graph.txt'}")
    print(f"  nodetypes:  {out_dir / 'nodetypes.tsv'}")
    print(f"  labels:     {out_dir / 'dda_labels.tsv'}")
    if f7 is not None:
        print(f"  atc:        {out_dir / '7_drug_classification_df.tsv'}")


if __name__ == "__main__":
    main()
