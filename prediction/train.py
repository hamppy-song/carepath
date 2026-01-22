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


def get_splitter(split_type: str, xs, ys, drugs, diseases, n_splits: int, seed: i_
