# prediction/config.py
import argparse
import json
from pathlib import Path

DEFAULT_BASE_MODELS = [
    {
        "name": "xgb1",
        "params": dict(
            n_estimators=689, max_depth=5, learning_rate=0.1510,
            subsample=0.9946, colsample_bytree=0.7067,
            tree_method="hist",
        ),
        "seed_offset": 0,
    },
    {
        "name": "xgb2",
        "params": dict(
            n_estimators=800, max_depth=5, learning_rate=0.1422,
            subsample=0.8117, colsample_bytree=0.9988,
            tree_method="hist",
        ),
        "seed_offset": 1,
    },
    {
        "name": "xgb3",
        "params": dict(
            n_estimators=763, max_depth=9, learning_rate=0.0892,
            subsample=0.9915, colsample_bytree=0.8584, min_child_weight=6,
            tree_method="hist",
        ),
        "seed_offset": 2,
    },
]

def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--embedding_file", type=str, required=True)
    p.add_argument("--pair_file", type=str, required=True)

    # run
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--splits", type=str, default="random,drug,disease",
                   help="Comma-separated: random,drug,disease")

    # outputs
    p.add_argument("--output_file", type=str, default="cv_results.tsv")
    p.add_argument("--pred_detail_file", type=str, default="cv_pred_details.tsv")

    # model config (독자가 마음대로 바꾸게)
    p.add_argument("--xgb_config", type=str, default=None,
                   help="Optional JSON file for base models. If omitted, uses DEFAULT_BASE_MODELS.")
    p.add_argument("--meta_C", type=float, default=1.0,
                   help="LogisticRegression C for meta-learner (default=1.0).")
    p.add_argument("--stack_cv", type=int, default=5,
                   help="Internal CV for stacking (default=5).")

    args = p.parse_args()

    # load base models config
    base_models = DEFAULT_BASE_MODELS
    if args.xgb_config is not None:
        cfg_path = Path(args.xgb_config)
        with cfg_path.open("r") as f:
            base_models = json.load(f)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    return {
        "embeddingf": args.embedding_file,
        "pairf": args.pair_file,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "splits": splits,
        "output_file": args.output_file,
        "pred_detail_file": args.pred_detail_file,
        "base_models": base_models,
        "meta_C": args.meta_C,
        "stack_cv": args.stack_cv,
    }
