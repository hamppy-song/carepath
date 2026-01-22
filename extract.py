# carepath/extract.py
from typing import Dict

import numpy as np
from tqdm import tqdm
from node2vec import Node2Vec

from biolinkbert_embeddings import get_biolinkbert_cls_embedding

from .io_utils import (
    load_node_types,
    classify_nodes_from_types,
    load_disease_drug_pairs,
    build_id_to_name_mapping,
    save_pickle,
)
from .graph_utils import (
    set_seed,
    read_graph,
    find_fixed_paths,
    load_drug_atc_dict,
    teleport_drug,
)
from .prompts import path_to_prompt, fallback_prompt
from .mech_context import (
    build_entity_contexts_safe,
    mech_emb_from_ctx_texts,
    build_neighbor_pool_from_atc,
    build_weighted_gene_vectors,
    build_knn_neighbor_pool_from_sparse,
)


def run_embedding_extraction(args) -> str:
    """
    Main entry:
      - builds embeddings dict and saves to args.output_file
      - returns saved path
    """
    final_seed = int(args.seed) + int(args.run_id)
    set_seed(final_seed)

    # ---------- Load graph ----------
    G = read_graph(
        args.network_file,
        weighted=args.weighted,
        directed=args.directed,
        delimiter=args.net_delimiter,
    )
    print(f"#nodes={G.number_of_nodes():,}, #edges={G.number_of_edges():,}")

    # ---------- Node types ----------
    node2type = load_node_types(args.node_type_file)
    disease_list, drug_list = classify_nodes_from_types(args.node_type_file)
    print(f"Found {len(disease_list):,} diseases & {len(drug_list):,} drugs")

    # ---------- Optional ID->name mapping ----------
    id2name = build_id_to_name_mapping(args.drug2prot_tsv, args.dis2prot_tsv, args.ppi_tsv)

    # ---------- Node2Vec ----------
    print("Training Node2Vec...")
    node2vec = Node2Vec(
        G,
        dimensions=args.n2v_dim,
        walk_length=args.n2v_walk_length,
        num_walks=args.n2v_num_walks,
        workers=args.workers,
        weight_key="weight",
    )
    n2v_model = node2vec.fit(
        window=args.n2v_window,
        min_count=1,
        seed=final_seed,
        workers=args.workers,
    )
    print("Node2Vec done.")

    # ---------- ATC dict ----------
    drug_atc_tele = {}
    drug_atc_pool = {}
    if args.atc_tsv:
        drug_atc_tele = load_drug_atc_dict(args.atc_tsv, level=args.atc_level_teleport)
        drug_atc_pool = load_drug_atc_dict(args.atc_tsv, level=args.atc_level_pool)

    # ---------- Build mech contexts (base) ----------
    print("Building leakage-safe entity contexts...")
    drug_ctx, dis_ctx = build_entity_contexts_safe(
        G, node2type=node2type, id2name=id2name, max_neighbors=args.ctx_max_neighbors
    )

    print("Encoding base mechanism embeddings...")
    drug_mech = {}
    for d in tqdm(drug_list, desc="drug mech (base)"):
        e = mech_emb_from_ctx_texts(drug_ctx.get(d, []), topM=args.ctx_topM)
        drug_mech[d] = e if e is not None else np.zeros(args.bert_dim, dtype=np.float32)

    disease_mech = {}
    for s in tqdm(disease_list, desc="disease mech (base)"):
        e = mech_emb_from_ctx_texts(dis_ctx.get(s, []), topM=args.ctx_topM)
        disease_mech[s] = e if e is not None else np.zeros(args.bert_dim, dtype=np.float32)

    # ---------- Pooling + residual mixing ----------
    print("Pooling mechanism embeddings...")
    # drug: ATC pooling
    if drug_atc_pool:
        drug_mech_mixed = build_neighbor_pool_from_atc(
            drug_list=drug_list,
            drug_atc_dict=drug_atc_pool,
            mech_emb_dict=drug_mech,
            alpha=args.alpha,
            k=args.atc_pool_k,
        )
    else:
        drug_mech_mixed = drug_mech

    # disease: gene-vector kNN pooling
    X_dis, dis_order, _ = build_weighted_gene_vectors(
        G,
        node2type=node2type,
        nodes=disease_list,
        max_1hop=args.max_1hop_genes,
        use_idf=(not args.no_idf),
    )
    disease_mech_mixed = build_knn_neighbor_pool_from_sparse(
        X_dis,
        dis_order,
        disease_mech,
        k=args.k_sim,
        alpha=args.alpha,
        weighted_by_sim=(not args.knn_unweighted),
    )

    # ---------- Pair loop ----------
    pairs_raw = load_disease_drug_pairs(args.pair_file)
    # deduplicate by (disease, drug)
    uniq = {}
    for s, d, y in pairs_raw:
        uniq[(s, d)] = y
    pairs = [(s, d, uniq[(s, d)]) for (s, d) in uniq.keys()]

    grouped_embeddings: Dict[str, np.ndarray] = {}
    print(f"Processing {len(pairs):,} disease-drug pairs...")

    for disease, drug, _label in tqdm(pairs, desc="pairs"):
        # 1) enumerate constrained paths
        paths = find_fixed_paths(G, disease, drug, max_genes=args.max_genes)

        # 2) teleport if no paths
        if not paths and drug_atc_tele:
            tele_drug = teleport_drug(drug, drug_atc_tele, G)
            if tele_drug != drug:
                paths = find_fixed_paths(G, disease, tele_drug, max_genes=args.max_genes)

        # 3) BioLinkBERT prompt embeddings (max pool)
        emb_list = []
        if paths:
            for path in paths:
                prompt = path_to_prompt(path, id2name, node2type)
                if args.debug_prompt_prob > 0 and np.random.rand() < args.debug_prompt_prob:
                    print("\n[DEBUG PROMPT SAMPLE]\n", prompt, "\n" + "-" * 60)
                emb_list.append(np.asarray(get_biolinkbert_cls_embedding(prompt)))
        else:
            fp = fallback_prompt(disease, drug, id2name)
            emb_list.append(np.asarray(get_biolinkbert_cls_embedding(fp, max_length=args.prompt_max_length)))

        bio_emb = np.max(np.stack(emb_list, axis=0), axis=0).astype(np.float32)

        # 4) node2vec
        drug_vec = n2v_model.wv[drug] if drug in n2v_model.wv else np.zeros(args.n2v_dim, dtype=np.float32)
        dis_vec = n2v_model.wv[disease] if disease in n2v_model.wv else np.zeros(args.n2v_dim, dtype=np.float32)

        # 5) mech context
        d_me = drug_mech_mixed.get(drug, np.zeros(args.bert_dim, dtype=np.float32))
        s_me = disease_mech_mixed.get(disease, np.zeros(args.bert_dim, dtype=np.float32))

        final_emb = np.concatenate([drug_vec, dis_vec, bio_emb, d_me, s_me]).astype(np.float32)
        grouped_embeddings[f"{disease}__{drug}"] = final_emb

    # ---------- Save ----------
    save_pickle(grouped_embeddings, args.output_file)
    print(f"✅ Saved: {args.output_file} (#pairs={len(grouped_embeddings):,})")
    return args.output_file
