# carepath/mech_context.py
import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from biolinkbert_embeddings import get_biolinkbert_cls_embedding


# ---------- BioLinkBERT caching ----------
_emb_cache: Dict[str, np.ndarray] = {}


def embed_cached(text: str) -> np.ndarray:
    if text in _emb_cache:
        return _emb_cache[text]
    e = np.asarray(get_biolinkbert_cls_embedding(text))
    _emb_cache[text] = e
    return e


def is_gene_or_protein(node: str, node2type: Dict[str, str]) -> bool:
    t = node2type.get(node, "")
    if t in ("gene", "protein"):
        return True
    s = str(node)
    return s.startswith(("G", "P"))


# ---------- Context texts (leakage-safe) ----------
def build_entity_contexts_safe(
    G: nx.Graph,
    node2type: Dict[str, str],
    id2name: Dict[str, str],
    max_neighbors: Optional[int] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    drug/disease context uses only gene/protein neighbors (avoid drug<->disease leakage)
    """
    drug_ctx = defaultdict(list)
    dis_ctx = defaultdict(list)

    for node in G.nodes():
        t = node2type.get(node)
        if t not in ("drug", "disease"):
            continue

        nbrs = list(G.neighbors(node))
        if max_neighbors is not None and len(nbrs) > max_neighbors:
            nbrs = random.sample(nbrs, max_neighbors)

        node_name = id2name.get(node, node)

        for nb in nbrs:
            if not is_gene_or_protein(nb, node2type):
                continue
            nb_name = id2name.get(nb, nb)

            if t == "drug":
                drug_ctx[node].append(f"Drug {node_name} is connected to {nb_name}.")
            else:
                dis_ctx[node].append(f"Disease {node_name} is connected to {nb_name}.")

    return drug_ctx, dis_ctx


def mech_emb_from_ctx_texts(ctx_texts: List[str], topM: int = 30) -> Optional[np.ndarray]:
    if not ctx_texts:
        return None
    texts = ctx_texts[:topM]
    embs = [embed_cached(t) for t in texts]
    return np.mean(embs, axis=0)


# ---------- Drug pooling (ATC) ----------
def build_neighbor_pool_from_atc(
    drug_list: List[str],
    drug_atc_dict: Dict[str, str],
    mech_emb_dict: Dict[str, np.ndarray],
    alpha: float = 0.5,
    k: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Deterministic ATC-group pooling:
      mech_new = alpha*orig + (1-alpha)*mean(neighbors)
    """
    group2drugs = defaultdict(list)
    for d in drug_list:
        g = drug_atc_dict.get(d)
        if g:
            group2drugs[g].append(d)

    for g in group2drugs:
        group2drugs[g] = sorted(group2drugs[g])

    out = {}
    for d in drug_list:
        orig = mech_emb_dict.get(d, np.zeros(768, dtype=np.float32))
        g = drug_atc_dict.get(d)

        if not g or g not in group2drugs:
            out[d] = orig
            continue

        nbrs = [x for x in group2drugs[g] if x != d and x in mech_emb_dict]
        if k is not None and len(nbrs) > k:
            nbrs = nbrs[:k]

        if not nbrs:
            pool = np.zeros(768, dtype=np.float32)
        else:
            pool = np.stack([mech_emb_dict[n] for n in nbrs], axis=0).mean(axis=0)

        out[d] = alpha * orig + (1.0 - alpha) * pool

    return out


# ---------- Disease pooling (gene-vector kNN) ----------
def build_weighted_gene_vectors(
    G: nx.Graph,
    node2type: Dict[str, str],
    nodes: List[str],
    max_1hop: int = 500,
    use_idf: bool = True,
) -> Tuple[csr_matrix, List[str], List[str]]:
    """
    For each node, build sparse weighted vector over 1-hop gene/protein neighbors.
    """
    node_order = list(nodes)

    node_gene_lists = []
    df_counter = Counter()

    for n in node_order:
        genes = []
        for nb in G.neighbors(n):
            if is_gene_or_protein(nb, node2type):
                genes.append(nb)

        if max_1hop is not None and len(genes) > max_1hop:
            genes = random.sample(genes, max_1hop)

        uniq = set(genes)
        for g in uniq:
            df_counter[g] += 1
        node_gene_lists.append(list(uniq))

    genes_all = list(df_counter.keys())
    gene2col = {g: i for i, g in enumerate(genes_all)}
    N = len(node_order)

    if use_idf:
        idf = {g: math.log((N + 1) / (df_counter[g] + 1)) + 1.0 for g in genes_all}
    else:
        idf = {g: 1.0 for g in genes_all}

    rows, cols, data = [], [], []
    for r, gene_list in enumerate(node_gene_lists):
        for g in gene_list:
            rows.append(r)
            cols.append(gene2col[g])
            data.append(1.0 * idf[g])

    X = csr_matrix((data, (rows, cols)), shape=(N, len(genes_all)), dtype=np.float32)
    return X, node_order, genes_all


def build_knn_neighbor_pool_from_sparse(
    X: csr_matrix,
    node_order: List[str],
    mech_emb_dict: Dict[str, np.ndarray],
    k: int = 10,
    alpha: float = 0.5,
    weighted_by_sim: bool = True,
) -> Dict[str, np.ndarray]:
    """
    kNN over cosine distance on weighted gene vectors.
    """
    row_norm = np.sqrt(X.multiply(X).sum(axis=1)).A1 + 1e-9
    Xn = X.multiply(1.0 / row_norm[:, None])

    nn = NearestNeighbors(n_neighbors=min(k + 1, Xn.shape[0]), metric="cosine", algorithm="brute")
    nn.fit(Xn)
    dists, idxs = nn.kneighbors(Xn)

    out = {}
    for i, node in enumerate(node_order):
        nbr_idxs = idxs[i][1:]
        nbr_dists = dists[i][1:]

        if len(nbr_idxs) == 0:
            pool = np.zeros(768, dtype=np.float32)
        else:
            nbr_nodes = [node_order[j] for j in nbr_idxs]
            nbr_embs = np.stack([mech_emb_dict.get(n, np.zeros(768)) for n in nbr_nodes], axis=0)

            if weighted_by_sim:
                sim = 1.0 - nbr_dists
                w = sim / (sim.sum() + 1e-9)
                pool = (nbr_embs * w[:, None]).sum(axis=0)
            else:
                pool = nbr_embs.mean(axis=0)

        orig = mech_emb_dict.get(node, np.zeros(768))
        out[node] = alpha * orig + (1.0 - alpha) * pool

    return out
