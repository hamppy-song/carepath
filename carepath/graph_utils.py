# carepath/graph_utils.py
import random
from typing import Dict, List, Tuple, Optional

import networkx as nx
import pandas as pd


def load_node_types(node_type_file: str) -> Dict[str, str]:
    """Load node types from nodetypes.tsv (node, type)."""
    df = pd.read_csv(node_type_file, sep="\t", header=0)
    node2type = {}
    for _, row in df.iterrows():
        node = str(row[0]).strip()
        ntype = str(row[1]).strip().lower()
        node2type[node] = ntype
    return node2type


def classify_nodes_from_types(node_type_file: str) -> Tuple[List[str], List[str]]:
    """Return (disease_nodes, drug_nodes) based on nodetypes file."""
    node2type = load_node_types(node_type_file)
    drug_nodes = [node for node, t in node2type.items() if t == "drug"]
    disease_nodes = [node for node, t in node2type.items() if t == "disease"]
    return disease_nodes, drug_nodes


def clean_graph_nodes(G: nx.Graph) -> nx.Graph:
    """Strip whitespace from node ids."""
    mapping = {node: str(node).strip() for node in G.nodes}
    return nx.relabel_nodes(G, mapping)


def find_fixed_paths(G: nx.Graph, disease: str, drug: str, max_genes: int = 5) -> List[List[str]]:
    """
    Enumerate simple paths disease->drug with cutoff=3, keeping only paths
    with <= max_genes gene nodes (prefix 'G').
    """
    all_paths = []
    for path in nx.all_simple_paths(G, source=disease, target=drug, cutoff=3):
        genes = [n for n in path if str(n).startswith("G")]
        if len(genes) <= max_genes:
            all_paths.append(path)
    return all_paths


def load_drug_atc_dict(atc_path: str, level: int = 3) -> Dict[str, str]:
    """
    Build {db_id -> atc_prefix[:level]} mapping.
    """
    atc_df = pd.read_csv(atc_path, sep="\t")
    atc_df = atc_df.dropna(subset=["db_id", "atc_code"])
    atc_df["atc_group"] = atc_df["atc_code"].astype(str).str.strip().str[:level]
    return dict(zip(atc_df["db_id"].astype(str).str.strip(), atc_df["atc_group"]))


def build_drug_similarity_graph(drug_atc_dict: Dict[str, str], G: nx.Graph) -> nx.Graph:
    """
    Build drug-drug similarity graph where edges connect drugs that share same ATC group.
    NOTE: This is O(n^2). Keep only if you really need it.
    """
    G_sim = nx.Graph()
    drugs = [d for d in drug_atc_dict if d in G.nodes()]
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            d1, d2 = drugs[i], drugs[j]
            if drug_atc_dict[d1] == drug_atc_dict[d2]:
                G_sim.add_edge(d1, d2, weight=1.0)
    return G_sim


def teleport_operation(cur: str, G_sim: nx.Graph) -> Optional[str]:
    """Weighted random neighbor choice based on edge weight."""
    if cur not in G_sim:
        return None
    nbrs = list(G_sim.neighbors(cur))
    if not nbrs:
        return None

    weights = [G_sim[cur][nbr].get("weight", 0.0) for nbr in nbrs]
    total = sum(weights)
    if total <= 0:
        return None

    r = random.uniform(0, total)
    c = 0.0
    for nbr, w in zip(nbrs, weights):
        c += w
        if c >= r:
            return nbr
    return None


def teleport_drug(current_drug: str, drug_atc_dict: Dict[str, str], G: nx.Graph) -> str:
    """Teleport within same ATC group if possible (random choice), else return itself."""
    group = drug_atc_dict.get(current_drug)
    if not group:
        return current_drug

    same_group = [
        d for d, g in drug_atc_dict.items()
        if g == group and d != current_drug and d in G.nodes()
    ]
    return random.choice(same_group) if same_group else current_drug
