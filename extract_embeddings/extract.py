# extract_embeddings/extract.py
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from node2vec import Node2Vec
from biolinkbert_embeddings import get_biolinkbert_cls_embedding

from carepath.utils import read_graph, set_seed
from carepath.graph_utils import (
    load_node_types,
    classify_nodes_from_types,
    clean_graph_nodes,
    find_fixed_paths,
    teleport_drug,
)
from carepath.prompts import (
    build_id_to_name_mapping,
    path_to_prompt,
)
from carepath.mech_context import (
    build_entity_contexts_safe,
    mech_emb_from_ctx_texts,
    build_neighbor_pool_from_atc,
    build_weighted_gene_vectors,
    build_knn_neighbor_pool_from_sparse,
)


def load_disease_drug_pairs(pair_file_path: str):
    df = pd.read_csv(pair_file_path, sep="\t", header=0)
    pairs = []
    for _, row in df.iterrows():
        drug = str(row["drug"]).strip()
        disease = str(row["disease"]).strip()
        label = int(row["label"])
        pairs.append((disease, drug, label))
    return pairs


def save_embedding_files(
    netf: str,
    outputf: str,
    nodetypef: str,
    tp_factor: float = 0.5,  # kept for compatibility; not used in current pipeline
    max_genes: int = 5,
    seed: int = 42,
    directed: bool = False,
    weighted: bool = True,
    workers: int = 5,
    net_delimiter: str = " ",
    pairf: str = None,
    dataset_dir: str = None,
    atc_file: str = None,
):


    set_seed(seed)

    if pairf is None:
        raise ValueError("pairf is required (path to the labeled drug-disease pair TSV).")

    dpath = Path(dataset_dir) if dataset_dir is not None else None


    if atc_file is None:
        if dpath is None:
            raise ValueError("atc_file is None but dataset_dir was not provided.")
        atc_file = str(dpath / "7_drug_classification_df.tsv")


    print("Reading network file preview...")
    with open(netf, "r") as f:
        for _ in range(5):
            print(repr(f.readline()))

    atc_df = pd.read_csv(atc_file, sep="\t")
    atc_df["db_id"] = atc_df["db_id"].astype(str).str.strip()
    atc_df["atc_code"] = atc_df["atc_code"].astype(str).str.strip()
    atc_df = atc_df.dropna(subset=["db_id", "atc_code"])


    atc_df["atc_group"] = atc_df["atc_code"].str[:3]
    drug_atc_dict = dict(zip(atc_df["db_id"], atc_df["atc_group"]))


    drug_atc_dict_3 = dict(zip(atc_df["db_id"], atc_df["atc_code"].str[:3]))

    G = read_graph(netf, weighted=weighted, directed=directed, delimiter=net_delimiter)
    print(f"# of nodes: {len(G.nodes())}")
    print(f"# of edges: {len(G.edges())}")


    G = clean_graph_nodes(G)

    print("Training Node2Vec...")
    node2vec = Node2Vec(G, dimensions=128, walk_length=10, num_walks=100, workers=workers)
    n2v_model = node2vec.fit(window=10, min_count=1, seed=seed, workers=workers)
    print("Node2Vec training done.")

    disease_list, drug_list = classify_nodes_from_types(nodetypef)
    print(f"Found {len(disease_list)} diseases and {len(drug_list)} drugs")

    node2type = load_node_types(nodetypef)

    if dpath is None:
        # Falls back to prompts.py defaults (hard-coded paths)
        id2name = build_id_to_name_mapping()
    else:
        id2name = build_id_to_name_mapping(
            drug_tsv=str(dpath / "1_drug_to_protein.tsv"),
            disease_tsv=str(dpath / "2_indication_to_protein.tsv"),
            ppi_tsv=str(dpath / "3_protein_to_protein.tsv"),
        )

    drug_ctx, dis_ctx = build_entity_contexts_safe(G, node2type, id2name, max_neighbors=30)

    drug_mech_emb = {}
    for dr in drug_list:
        e = mech_emb_from_ctx_texts(drug_ctx.get(dr, []), topM=30)
        drug_mech_emb[dr] = e if e is not None else np.zeros(768, dtype=np.float32)

    disease_mech_emb = {}
    for ds in disease_list:
        e = mech_emb_from_ctx_texts(dis_ctx.get(ds, []), topM=30)
        disease_mech_emb[ds] = e if e is not None else np.zeros(768, dtype=np.float32)

    K_SIM = 10
    ALPHA = 0.5

    drug_mech_emb_mixed = build_neighbor_pool_from_atc(
        drug_list=drug_list,
        drug_atc_dict=drug_atc_dict_3,
        mech_emb_dict=drug_mech_emb,
        k=None,          # use all neighbors in the same group
        alpha=ALPHA,
        sample=False,    # kept for API compatibility; current implementation is deterministic
    )


    X_dis, dis_order, _ = build_weighted_gene_vectors(
        G, node2type, disease_list, max_1hop=500, use_idf=True
    )
    disease_mech_emb_mixed = build_knn_neighbor_pool_from_sparse(
        X_dis, dis_order, disease_mech_emb, k=K_SIM, alpha=ALPHA, weighted_by_sim=True
    )

    print("Generating Disease-Gene-Drug paths and building pair embeddings...")
    grouped_embeddings = {}

    total_pairs = sorted(set(load_disease_drug_pairs(pairf)))

    for disease, drug, label in tqdm(total_pairs, desc="Processing disease-drug pairs"):
        paths = find_fixed_paths(G, disease, drug, max_genes)

        if not paths:
            tele_drug = teleport_drug(drug, drug_atc_dict, G)
            paths = find_fixed_paths(G, disease, tele_drug, max_genes)

        rep_paths = paths if len(paths) > 0 else []

        drug_vec = n2v_model.wv[drug] if drug in n2v_model.wv else np.zeros(128, dtype=np.float32)
        disease_vec = n2v_model.wv[disease] if disease in n2v_model.wv else np.zeros(128, dtype=np.float32)

        emb_list = []
        if rep_paths:
            for path in rep_paths:
                prompt = path_to_prompt(path, id2name, node2type)

                if random.random() < 0.002:
                    print("\n[DEBUG PROMPT SAMPLE]")
                    print(prompt)
                    print("-" * 60)

                emb_list.append(get_biolinkbert_cls_embedding(prompt))
        else:
            dname = id2name.get(drug, drug)
            disname = id2name.get(disease, disease)
            fallback_prompt = (
                f"Premise: {disname} involves genes none.\n"
                f"Hypothesis: {dname} can be repurposed to treat {disname}.\n"
                f"Label:"
            )
            emb_list.append(get_biolinkbert_cls_embedding(fallback_prompt, max_length=512))

        if emb_list:
            emb_array = np.array(emb_list)
            bio_emb = np.max(emb_array, axis=0)

            d_me = drug_mech_emb_mixed.get(drug, np.zeros(768, dtype=np.float32))
            s_me = disease_mech_emb_mixed.get(disease, np.zeros(768, dtype=np.float32))

            final_emb = np.concatenate([drug_vec, disease_vec, bio_emb, d_me, s_me])

            key = f"{disease}__{drug}"
            grouped_embeddings[key] = final_emb

    print(f"Saving grouped embeddings to: {outputf}")
    with open(outputf, "wb") as fw:
        pickle.dump(grouped_embeddings, fw)

    print(f"Saved: {outputf}")
