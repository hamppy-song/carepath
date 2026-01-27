# extract_embeddings/config.py
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", type=str, required=True,
                   help="Dataset folder path (e.g., ./MSI dataset)")

    p.add_argument("--network_file", type=str, default=None)
    p.add_argument("--node_type_file", type=str, default=None)
    p.add_argument("--pair_file", type=str, default=None)
    p.add_argument("--atc_file", type=str, default=None)

    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weighted", type=bool, default=True)
    p.add_argument("--directed", type=bool, default=False)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--net_delimiter", type=str, default="\t")
    p.add_argument("--max_genes", type=int, default=5)
    p.add_argument("--tp_factor", type=float, default=0.5)

    args, _ = p.parse_known_args()
    d = Path(args.dataset_dir)

    netf      = args.network_file or str(d / "graph.txt")
    nodetypef = args.node_type_file or str(d / "nodetypes.tsv")   
    pairf     = args.pair_file or str(d / "dda_labels.tsv")
    atcf      = args.atc_file or str(d / "7_drug_classification_df.tsv")

    return {
        "dataset_dir": str(d),
        "netf": netf,
        "nodetypef": nodetypef,
        "pairf": pairf,
        "atc_file": atcf,
        "outputf": args.output_file,
        "seed": args.seed,
        "tp_factor": args.tp_factor,
        "weighted": args.weighted,
        "directed": args.directed,
        "workers": args.workers,
        "net_delimiter": args.net_delimiter,
        "max_genes": args.max_genes,
    }



def parse_run_id():
    extra_parser = argparse.ArgumentParser()
    extra_parser.add_argument('--run_id', type=int, required=True, help='Run index from 0 to 4')
    extra_args, _ = extra_parser.parse_known_args()
    return extra_args.run_id
