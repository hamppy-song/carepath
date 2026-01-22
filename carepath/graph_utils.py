import random
import networkx as nx
import pandas as pd

def load_node_types(node_type_file):
    df = pd.read_csv(node_type_file, sep='\t', header=0)
    node2type = {}
    for _, row in df.iterrows():
        node = str(row[0]).strip()
        ntype = str(row[1]).strip().lower()
        node2type[node] = ntype
    return node2type


def classify_nodes_from_types(node_type_file):
    node2type = load_node_types(node_type_file)
    drug_nodes = [node for node, t in node2type.items() if t == 'drug']
    disease_nodes = [node for node, t in node2type.items() if t == 'disease']
    return disease_nodes, drug_nodes


def clean_graph_nodes(G):
    mapping = {node: str(node).strip() for node in G.nodes}
    return nx.relabel_nodes(G, mapping)


def find_fixed_paths(G, disease, drug, max_genes=5):
    all_paths = []
    for path in nx.all_simple_paths(G, source=disease, target=drug, cutoff=3):
        genes = [node for node in path if str(node).startswith("G")]
        if len(genes) <= max_genes:
            all_paths.append(path)
    return all_paths


def load_drug_atc_dict(atc_path: str, level: int = 3):
    atc_df = pd.read_csv(atc_path, sep="\t")
    atc_df = atc_df.dropna(subset=["db_id", "atc_code"])
    atc_df["atc_group"] = atc_df["atc_code"].astype(str).str.strip().str[:level]
    drug_atc_dict = dict(zip(atc_df["db_id"].astype(str).str.strip(), atc_df["atc_group"]))
    return drug_atc_dict


def build_drug_similarity_graph(drug_atc_dict, G):
    # (현재 파이프라인에서는 G_sim을 실제로 사용하지 않지만, 원본 코드에 있었던 함수라 유지)
    G_sim = nx.Graph()
    drugs = [drug for drug in drug_atc_dict if drug in G.nodes()]
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            drug1, drug2 = drugs[i], drugs[j]
            if drug_atc_dict[drug1] == drug_atc_dict[drug2]:
                G_sim.add_edge(drug1, drug2, weight=1.0)
    return G_sim


def teleport_operation(cur, G_sim):
    if cur not in G_sim:
        return None
    cur_nbrs = list(G_sim.neighbors(cur))
    if not cur_nbrs:
        return None

    weights = [G_sim[cur][nbr]['weight'] for nbr in cur_nbrs]
    total_weight = sum(weights)
    if total_weight == 0:
        return None

    rand_val = random.uniform(0, total_weight)
    cumulative_weight = 0
    for nbr, weight in zip(cur_nbrs, weights):
        cumulative_weight += weight
        if cumulative_weight >= rand_val:
            return nbr
    return None


def teleport_drug(current_drug, drug_atc_dict, G):
    current_atc_group = drug_atc_dict.get(current_drug)
    if not current_atc_group:
        return current_drug

    same_group_drugs = [
        drug for drug, atc_group in drug_atc_dict.items()
        if atc_group == current_atc_group and drug != current_drug and drug in G.nodes()
    ]
    if same_group_drugs:
        return random.choice(same_group_drugs)
    return current_drug
