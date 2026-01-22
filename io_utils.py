# carepath/io_utils.py
import os
import pickle
from typing import Dict, List, Tuple, Optional

import pandas as pd


def load_node_types(node_type_file: str) -> Dict[str, str]:
    df = pd.read_csv(node_type_file, sep="\t", header=0)
    node2type = {}
    for _, row in df.iterrows():
        node = str(row.iloc[0]).strip()
        ntype = str(row.iloc[1]).strip().lower()
        node2type[node] = ntype
    return node2type


def classify_nodes_from_types(node_type_file: str) -> Tuple[List[str], List[str]]:
    node2type = load_node_types(node_type_file)
    disease_nodes = [n for n, t in node2type.items() if t == "disease"]
    drug_nodes = [n for n, t in node2type.items() if t == "drug"]
    return disease_nodes, drug_nodes


def load_disease_drug_pairs(pair_file_path: str) -> List[Tuple[str, str, int]]:
    """
    pair TSV columns: drug, disease, label
    returns list of (disease, drug, label)
    """
    df = pd.read_csv(pair_file_path, sep="\t", header=0)
    pairs = []
    for _, row in df.iterrows():
        drug = str(row["drug"]).strip()
        disease = str(row["disease"]).strip()
        label = int(row["label"])
        pairs.append((disease, drug, label))
    return pairs


def build_id_to_name_mapping(
    drug2prot_tsv: Optional[str],
    dis2prot_tsv: Optional[str],
    ppi_tsv: Optional[str],
) -> Dict[str, str]:
    """
    Optional: TSVs expected columns: node_1,node_1_name,node_2,node_2_name
    """
    id2name: Dict[str, str] = {}

    def _ingest(tsv_path: str) -> None:
        if not tsv_path:
            return
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Mapping TSV not found: {tsv_path}")
        df = pd.read_csv(tsv_path, sep="\t")
        for _, row in df.iterrows():
            n1 = str(row["node_1"]).strip()
            n1n = str(row["node_1_name"]).strip()
            n2 = str(row["node_2"]).strip()
            n2n = str(row["node_2_name"]).strip()
            if n1:
                id2name[n1] = n1n
            if n2:
                id2name[n2] = n2n

    _ingest(drug2prot_tsv)
    _ingest(dis2prot_tsv)
    _ingest(ppi_tsv)

    return id2name


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
