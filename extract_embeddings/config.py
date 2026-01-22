import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--network_file', type=str,
                        default='/data/project/haerin/DREAMwalk/data/extracted/data/graph.txt')
    parser.add_argument('--node_type_file', type=str,
                        default='/data/project/haerin/DREAMwalk/data/extracted/data/nodetypes.tsv')
    parser.add_argument('--output_file', type=str,
                        default='/data/project/haerin/DREAMwalk/DREAMwalk/embedding_output/0107_embeddings_prompt5_biolinkbert_drugdisease_sim_drug1_atc6_all.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tp_factor', type=float, default=0.5)
    parser.add_argument('--weighted', type=bool, default=True)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--net_delimiter', type=str, default='\t')  # (원래 코드에서도 실사용은 안 됨)
    parser.add_argument('--max_genes', type=int, default=5)
    parser.add_argument('--pair_file', type=str,
                        default='/data/project/haerin/DREAMwalk/DREAMwalk/dda_all.tsv')

    # 기존처럼 Jupyter에서도 돌아가게 parse_known_args 유지
    args, _ = parser.parse_known_args()

    return {
        'netf': args.network_file,
        'outputf': args.output_file,
        'nodetypef': args.node_type_file,
        'tp_factor': args.tp_factor,
        'seed': args.seed,
        'weighted': args.weighted,
        'directed': args.directed,
        'workers': args.workers,
        'net_delimiter': args.net_delimiter,
        'max_genes': args.max_genes,
        'pairf': args.pair_file
    }


def parse_run_id():
    extra_parser = argparse.ArgumentParser()
    extra_parser.add_argument('--run_id', type=int, required=True, help='Run index from 0 to 4')
    extra_args, _ = extra_parser.parse_known_args()
    return extra_args.run_id
