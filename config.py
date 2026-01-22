# carepath/config.py
import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("CAREPath: extract embeddings")

    # Core inputs
    p.add_argument("--network_file", type=str, required=True, help="KG edge list (u\\tv[\\tw])")
    p.add_argument("--node_type_file", type=str, required=True, help="nodetypes.tsv (node\\ttype)")
    p.add_argument("--pair_file", type=str, required=True, help="pairs.tsv with columns: drug, disease, label")
    p.add_argument("--output_file", type=str, required=True, help="output .pkl path")

    # Optional mapping TSVs (ID -> name)
    p.add_argument("--drug2prot_tsv", type=str, default=None, help="1_drug_to_protein.tsv (optional)")
    p.add_argument("--dis2prot_tsv", type=str, default=None, help="2_indication_to_protein.tsv (optional)")
    p.add_argument("--ppi_tsv", type=str, default=None, help="3_protein_to_protein.tsv (optional)")

    # ATC
    p.add_argument("--atc_tsv", type=str, default=None, help="ATC TSV with columns: db_id, atc_code (optional)")
    p.add_argument("--atc_level_teleport", type=int, default=3, help="ATC prefix length for teleport (default=3)")
    p.add_argument("--atc_level_pool", type=int, default=3, help="ATC prefix length for drug pooling (default=3)")
    p.add_argument("--atc_pool_k", type=int, default=None, help="Limit #neighbors within ATC group (optional)")

    # Graph read options
    p.add_argument("--net_delimiter", type=str, default="\t")
    p.add_argument("--weighted", action="store_true")
    p.add_argument("--directed", action="store_true")

    # Seeds / run replication
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_id", type=int, default=0, help="final_seed = seed + run_id (for multiple runs)")

    # Path constraints
    p.add_argument("--max_genes", type=int, default=5, help="max genes allowed per path (Disease->Gene(s)->Drug)")
    p.add_argument("--prompt_max_length", type=int, default=512)
    p.add_argument("--debug_prompt_prob", type=float, default=0.0, help="e.g., 0.002 to print some prompts")

    # Node2Vec
    p.add_argument("--n2v_dim", type=int, default=128)
    p.add_argument("--n2v_walk_length", type=int, default=10)
    p.add_argument("--n2v_num_walks", type=int, default=100)
    p.add_argument("--n2v_window", type=int, default=10)
    p.add_argument("--workers", type=int, default=5)

    # Mechanism context
    p.add_argument("--bert_dim", type=int, default=768)
    p.add_argument("--ctx_topM", type=int, default=30, help="max neighbor sentences to embed per entity")
    p.add_argument("--ctx_max_neighbors", type=int, default=None, help="optional cap on 1-hop neighbors")
    p.add_argument("--alpha", type=float, default=0.5, help="residual mixing alpha (orig vs pooled)")

    # Disease pooling (gene-vector kNN)
    p.add_argument("--k_sim", type=int, default=10, help="kNN size for disease pooling")
    p.add_argument("--max_1hop_genes", type=int, default=500, help="cap genes/proteins sampled per disease 1-hop")
    p.add_argument("--no_idf", action="store_true", help="disable IDF downweighting for hub genes")
    p.add_argument("--knn_unweighted", action="store_true", help="use mean instead of similarity-weighted pooling")

    return p


def parse_args():
    return build_parser().parse_args()
