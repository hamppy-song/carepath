# carepath/utils.py
import os
import random
import numpy as np
import networkx as nx
import torch


def read_graph(edgeList: str, weighted: bool = True, directed: bool = False, delimiter: str = " "):
    """
    Read graph using NetworkX read_edgelist.

    Expected format (weighted=True):
      node1 <delim> node2 <delim> type(int) <delim> weight(float) <delim> id(int)

    If weighted=False:
      node1 <delim> node2 <delim> type(int)
      and weight will be set to 1.0 for all multi-edges.

    Returns:
      nx.Graph or nx.DiGraph (converted from MultiDiGraph / MultiGraph)
    """
    if weighted:
        G = nx.read_edgelist(
            edgeList,
            nodetype=str,
            data=(("type", int), ("weight", float), ("id", int)),
            create_using=nx.MultiDiGraph(),
            delimiter=delimiter,
        )
    else:
        G = nx.read_edgelist(
            edgeList,
            nodetype=str,
            data=(("type", int),),
            create_using=nx.MultiDiGraph(),
            delimiter=delimiter,
        )
        # set weight=1.0 for each multi-edge
        for u, v in G.edges():
            edge_dict = G[u][v]
            for k in edge_dict:
                edge_dict[k]["weight"] = 1.0

    if not directed:
        G = G.to_undirected()

    # strip node ids (important if file has trailing spaces)
    G = nx.relabel_nodes(G, {n: str(n).strip() for n in G.nodes()})
    return G


def set_seed(seed: int = 42, verbose: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if verbose:
        print(f"[seed] {seed}")
