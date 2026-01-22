# prediction/predict.py
import argparse
import pickle
import numpy as np
import pandas as pd

from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()

    # required
    p.add_argument("--model_file", type=str, required=True,
                   help="Pickle file containing a trained sklearn model (e.g., StackingClassifier).")
    p.add_argument("--embedding_file", type=str, required=True,
                   help="Pickle file: dict[key='disease__drug'] -> np.ndarray feature vector.")
    p.add_argument("--pair_file", type=str, required=True,
                   help="TSV file. Header expected. Columns: drug, disease, (optional) label")

    # optional
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output_file", type=str, default="predictions.tsv")
    p.add_argument("--skip_missing", action="store_true",
                   help="If set, skip pairs not found in embedding_dict. Otherwise raise error.")
    p.add_argument("--max_rows", type=int, default=None,
                   help="For quick test, limit number of rows read from pair_file.")
    return p.parse_args()


def load_embedding_dict(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def read_pairs(pair_file: str, max_rows=None):
    """
    Supports:
      - TSV with header containing columns drug, disease (and optional label)
      - Or first 3 columns are [drug, disease, label] if header differs
    """
    df = pd.read_csv(pair_file, sep="\t", dtype=str)

    # normalize column names
    cols_lower = {c.lower(): c for c in df.columns}

    # find drug/disease/label columns
    if "drug" in cols_lower and "disease" in cols_lower:
        drug_col = cols_lower["drug"]
        dis_col = cols_lower["disease"]
        label_col = cols_lower.get("label", None)
    else:
        # fallback: assume first two columns are drug/disease
        drug_col = df.columns[0]
        dis_col = df.columns[1]
        label_col = df.columns[2] if len(df.columns) >= 3 else None

    # strip
    df[drug_col] = df[drug_col].astype(str).str.strip()
    df[dis_col] = df[dis_col].astype(str).str.strip()

    if label_col is not None:
        # try convert to int if possible
        df[label_col] = df[label_col].astype(str).str.strip()

    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    return df, drug_col, dis_col, label_col


def main():
    args = parse_args()
    set_seed(args.seed)

    model = load_model(args.model_file)
    emb_dict = load_embedding_dict(args.embedding_file)

    df, drug_col, dis_col, label_col = read_pairs(args.pair_file, max_rows=args.max_rows)

    X = []
    keep_rows = []
    missing = 0

    for i, row in df.iterrows():
        drug = row[drug_col]
        dis = row[dis_col]
        key = f"{dis}__{drug}"

        if key not in emb_dict:
            missing += 1
            if args.skip_missing:
                continue
            else:
                raise KeyError(f"Missing embedding for key={key} (row={i}). "
                               f"Use --skip_missing to ignore.")
        X.append(emb_dict[key])
        keep_rows.append(i)

    if len(X) == 0:
        raise RuntimeError("No usable pairs found (all missing embeddings?).")

    X = np.array(X)
    df_used = df.loc[keep_rows].copy().reset_index(drop=True)

    # predict prob
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[:, 1]
    else:
        # fallback: decision_function -> sigmoid
        if not hasattr(model, "decision_function"):
            raise ValueError("Model has neither predict_proba nor decision_function.")
        logits = model.decision_function(X)
        prob = 1.0 / (1.0 + np.exp(-logits))

    # logit
    eps = 1e-9
    prob_clip = np.clip(prob, eps, 1 - eps)
    logit = np.log(prob_clip / (1 - prob_clip))

    pred = (prob >= args.threshold).astype(int)

    df_used["entity_key"] = [f"{d}__{dr}" for d, dr in zip(df_used[dis_col], df_used[drug_col])]
    df_used["prob"] = prob.astype(float)
    df_used["logit"] = logit.astype(float)
    df_used["y_pred"] = pred.astype(int)

    # if label exists, compute correctness (optional)
    if label_col is not None:
        # if label column is non-numeric, coerce safely
        try:
            y_true = df_used[label_col].astype(int).values
            df_used["y_true"] = y_true.astype(int)
            df_used["correct"] = (df_used["y_true"].values == df_used["y_pred"].values).astype(int)
        except Exception:
            # leave label as-is
            pass

    df_used.to_csv(args.output_file, sep="\t", index=False)

    print("✅ Saved predictions:", args.output_file)
    print(f"Pairs used: {len(df_used)} | Missing embeddings: {missing} | threshold={args.threshold}")


if __name__ == "__main__":
    main()
