# carepath/graph_utils.py
import random
from typing import Dict, List
import networkx as nx

from .utils import read_graph, set_seed


def find_fixed_paths(G: nx.Graph, disease: str, drug: str, max_genes: int = 5) -> List[List[str]]:
    """
    Enumerate simple paths disease->drug with cutoff=3
    Filter by number of gene nodes (prefix 'G') <= max_genes
    """
    all_paths: List[List[str]] = []
    try:
        for path in nx.all_simple_paths(G, source=disease, target=drug, cutoff=3):
            genes = [n for n in path if str(n).startswith("G")]
            if len(genes) <= max_genes:
                all_paths.append([str(x).strip() for x in path])
    except nx.NodeNotFound:
        return []
    return all_paths


def teleport_drug(current_drug: str, drug_atc_dict: Dict[str, str], G: nx.Graph) -> str:
    g = drug_atc_dict.get(current_drug)
    if not g:
        return current_drug

    same_group = [
        d for d, gg in drug_atc_dict.items()
        if gg == g and d != current_drug and d in G.nodes()
    ]
    return random.choice(same_group) if same_group else current_drug
