# prediction/train.py
import pickle
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from utils import set_seed  # 너 utils.py 그대로 사용
from config import parse_args


def load_dataset(embeddingf: str, pairf: str):
    with open(embeddingf, "rb") as fin:
        embedding_dict = pickle.load(fin)

    xs, ys, drugs, diseases = [], [], [], []

    with open(pairf, "r") as fin:
        lines = fin.readlines()

    for line in lines[1:]:
        drug, dis, label = line.strip().split("\t")
        drug = drug.strip()
        dis = dis.strip()
        label = int(label)

        key = f"{dis}__{drug}"
        if key not in embedding_dict:
            continue

        xs.append(embedding_dict[key])
        ys.append(label)
        drugs.append(drug)
        diseases.append(dis)

    return np.array(xs), np.array(ys), np.array(drugs), np.array(diseases)


def return_scores(y_true, y_prob):
    scores = []
    for metric in [accuracy_score, roc_auc_score, average_precision_score, f1_score]:
        if metric in [roc_auc_score, average_precision_score]:
            scores.append(metric(y_true, y_prob))
        else:
            scores.append(metric(y_true, np.round(y_prob)))
    return scores  # [ACC, AUROC, AUPRC, F1]


def make_stacking_clf(base_models_cfg, seed: int, meta_C: float, stack_cv: int):
    base_models = []
    for m in base_models_cfg:
        name = m["name"]
        params = dict(m["params"])
        params["random_state"] = seed + int(m.get("seed_offset", 0))
        base_models.append((name, XGBClassifier(**params)))

    clf = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(C=meta_C, max_iter=1000),
        cv=stack_cv,
        n_jobs=-1,
    )
    return clf


def get_splitter(split_type: str, xs, ys, drugs, diseases, n_splits: int, seed: int):
    if split_type == "random":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return splitter.split(xs, ys)

    if split_type == "disease":
        groups = diseases
    elif split_type == "drug":
        groups = drugs
    else:
        raise ValueError(f"Unknown split_type: {split_type}")

    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(xs, ys, groups=groups)


def run_cv(args: dict, split_type: str):
    set_seed(args["seed"])
    xs, ys, drugs, diseases = load_dataset(args["embeddingf"], args["pairf"])

    pred_rows = []
    fold_metric_rows = []
    fold_scores = []

    split_iter = get_splitter(
        split_type=split_type,
        xs=xs, ys=ys,
        drugs=drugs, diseases=diseases,
        n_splits=args["n_splits"],
        seed=args["seed"],
    )

    for fold, (tr_idx, te_idx) in enumerate(split_iter, start=1):
        X_train, X_test = xs[tr_idx], xs[te_idx]
        y_train, y_test = ys[tr_idx], ys[te_idx]

        clf = make_stacking_clf(
            base_models_cfg=args["base_models"],
            seed=args["seed"],
            meta_C=args["meta_C"],
            stack_cv=args["stack_cv"],
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]

        # logit (optional debug/analysis)
        eps = 1e-9
        y_prob_clip = np.clip(y_prob, eps, 1 - eps)
        y_logit = np.log(y_prob_clip / (1 - y_prob_clip))

        drug_test = drugs[te_idx]
        dis_test = diseases[te_idx]
        y_pred = (y_prob >= 0.5).astype(int)

        # ====== fold-level prints (원래 너 로그 유지) ======
        print("\n[LOGIT SAMPLE] (first 20 in this fold)")
        for i in range(min(20, len(y_test))):
            print(
                f"{dis_test[i]}__{drug_test[i]}"
                f"\ty={int(y_test[i])}\tprob={y_prob[i]:.6f}\tlogit={y_logit[i]:.6f}"
            )

        wrong = np.where(y_pred != y_test)[0]
        if len(wrong) > 0:
            print(f"\n[WRONG CASES] n={len(wrong)} (show up to 50)")
            for idx in wrong[:50]:
                print(
                    f"{dis_test[idx]}__{drug_test[idx]}"
                    f"\ty={int(y_test[idx])}\tpred={int(y_pred[idx])}"
                    f"\tprob={y_prob[idx]:.6f}\tlogit={y_logit[idx]:.6f}"
                )
        else:
            print("\n[WRONG CASES] n=0")

        df_fold = pd.DataFrame({
            "split_type": split_type,
            "fold": fold,
            "entity_key": [f"{d}__{dr}" for d, dr in zip(dis_test, drug_test)],
            "disease": dis_test,
            "drug": drug_test,
            "y_true": y_test.astype(int),
            "y_pred": y_pred.astype(int),
            "prob": y_prob.astype(float),
            "logit": y_logit.astype(float),
            "correct": (y_pred == y_test).astype(int),
        })
        pred_rows.append(df_fold)

        scores = return_scores(y_test, y_prob)
        print(f"▶ Fold {fold} | ACC: {scores[0]*100:.2f}% | AUROC: {scores[1]:.4f} | AUPRC: {scores[2]:.4f} | F1: {scores[3]:.4f}")
        print(
            f"FOLD-{fold} TEST-{split_type} "
            f"Acc: {scores[0]*100:.2f}% | AUROC: {scores[1]:.4f} | AUPR: {scores[2]:.4f} | F1: {scores[3]:.4f}"
        )

        fold_metric_rows.append({"fold": fold, "ACC": scores[0], "AUROC": scores[1], "AUPR": scores[2], "F1": scores[3]})
        fold_scores.append(scores)

    fold_scores = np.array(fold_scores)
    fold_metric_df = pd.DataFrame(fold_metric_rows)
    print("\n📌 Fold-wise metrics (summary)")
    print(fold_metric_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    pred_df = pd.concat(pred_rows, ignore_index=True)

    pred_out = args["pred_detail_file"].replace(".tsv", f"_{split_type}.tsv")
    pred_df.to_csv(pred_out, sep="\t", index=False)
    print("✅ Saved per-pair predictions to:", pred_out)

    wrong_out = args["pred_detail_file"].replace(".tsv", f"_{split_type}_WRONG.tsv")
    pred_df[pred_df["correct"] == 0].to_csv(wrong_out, sep="\t", index=False)
    print("✅ Saved WRONG cases to:", wrong_out)

    result = {
        "split_type": split_type,
        "AUC_mean": fold_scores[:, 1].mean(),   "AUC_std": fold_scores[:, 1].std(),
        "AUPRC_mean": fold_scores[:, 2].mean(), "AUPRC_std": fold_scores[:, 2].std(),
        "ACC_mean": fold_scores[:, 0].mean(),   "ACC_std": fold_scores[:, 0].std(),
        "F1_mean": fold_scores[:, 3].mean(),    "F1_std": fold_scores[:, 3].std(),
    }
    print(f"\n📊 {split_type} CV Results:")
    for k, v in result.items():
        if k != "split_type":
            print(f"  {k}: {v:.4f}")

    return result


def main():
    args = parse_args()

    results = []
    for split in args["splits"]:
        print(f"\n===== {args['n_splits']}-fold CV: split_type = {split} =====")
        results.append(run_cv(args, split))

    df_result = pd.DataFrame(results)
    cols = ["split_type", "AUC_mean", "AUC_std", "AUPRC_mean", "AUPRC_std", "ACC_mean", "ACC_std", "F1_mean", "F1_std"]
    df_result = df_result[cols]
    df_result.to_csv(args["output_file"], sep="\t", index=False)
    print("\n✅ All Done. Results saved to", args["output_file"])

    for _, res in df_result.iterrows():
        print(
            f"TEST-{res.split_type} - "
            f"Acc:  {res.ACC_mean*100:.2f}% STD: {res.ACC_std*100:.2f}% | "
            f"AUROC: {res.AUC_mean:.4f} STD: {res.AUC_std:.4f} | "
            f"AUPR:  {res.AUPRC_mean:.4f} STD: {res.AUPRC_std:.4f} | "
            f"F1:    {res.F1_mean:.4f} STD: {res.F1_std:.4f}"
        )


if __name__ == "__main__":
    main()
