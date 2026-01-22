# carepath/mech_context.py
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from biolinkbert_embeddings import get_biolinkbert_cls_embedding


# global cache (원본과 동일)
emb_cache: Dict[str, np.ndarray] = {}


def embed_cached(text: str) -> np.ndarray:
    """CLS embedding with simple in-memory cache."""
    if text in emb_cache:
        return emb_cache[text]
    e = get_biolinkbert_cls_embedding(text)
    emb_cache[text] = e
    return e


def is_gene_or_protein(node: str, node2type: Dict[str, str]) -> bool:
    t = node2type.get(node, "")
    if t in ("gene", "protein"):
        return True
    s = str(node)
    return s.startswith("G") or s.startswith("P")


def build_entity_contexts_safe(G, node2type: Dict[str, str], id2name: Dict[str, str], max_neighbors: int = 30):
    """
    Leak-safe entity contexts:
    - Only gene/protein neighbors (gene/protein or prefix G/P).
    """
    drug_ctx = defaultdict(list)
    dis_ctx = defaultdict(list)

    def _is_gp(n, t):
        if t in ("gene", "protein"):
            return True
        s = str(n)
        return s.startswith("G") or s.startswith("P")

    for node in G.nodes():
        t = node2type.get(node)
        if t not in ("drug", "disease"):
            continue

        nbrs = list(G.neighbors(node))
        # 원본도 sampling은 주석처리: 그대로 유지
        node_name = id2name.get(node, node)

        for nb in nbrs:
            nb_t = node2type.get(nb, "unknown")
            if not _is_gp(nb, nb_t):
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
    embs = [embed_cached(t) for t in ctx_texts[:topM]]
    return np.mean(embs, axis=0)


def build_weighted_gene_vectors(
    G,
    node2type: Dict[str, str],
    nodes: List[str],
    max_1hop: int = 500,
    use_idf: bool = True,
) -> Tuple[csr_matrix, List[str], List[str]]:
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
    row_norm = np.sqrt(X.multiply(X).sum(axis=1)).A1 + 1e-9
    Xn = X.multiply(1.0 / row_norm[:, None])

    nn = NearestNeighbors(n_neighbors=min(k + 1, Xn.shape[0]), metric="cosine", algorithm="brute")
    nn.fit(Xn)
    dists, idxs = nn.kneighbors(Xn)

    mech_new_dict = {}
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
        mech_new_dict[node] = alpha * orig + (1.0 - alpha) * pool

    return mech_new_dict


def build_neighbor_pool_from_atc(
    drug_list: List[str],
    drug_atc_dict: Dict[str, str],
    mech_emb_dict: Dict[str, np.ndarray],
    k: Optional[int] = 10,
    alpha: float = 0.5,
    sample: bool = False,  # 호환성 유지용(현재 미사용)
) -> Dict[str, np.ndarray]:
    group2drugs = defaultdict(list)
    for d in drug_list:
        g = drug_atc_dict.get(d)
        if g:
            group2drugs[g].append(d)

    for g in group2drugs:
        group2drugs[g] = sorted(group2drugs[g])

    mech_new_dict = {}
    for d in drug_list:
        orig = mech_emb_dict.get(d, np.zeros(768, dtype=np.float32))
        g = drug_atc_dict.get(d)

        if not g or g not in group2drugs:
            mech_new_dict[d] = orig
            continue

        nbrs = [x for x in group2drugs[g] if x != d and x in mech_emb_dict]
        if not nbrs:
            pool = np.zeros(768, dtype=np.float32)
        else:
            if k is not None and len(nbrs) > k:
                nbrs = nbrs[:k]
            nbr_embs = np.stack([mech_emb_dict[n] for n in nbrs], axis=0)
            pool = nbr_embs.mean(axis=0)

        mech_new_dict[d] = alpha * orig + (1.0 - alpha) * pool

    return mech_new_dict
