import os

from .config import parse_args, parse_run_id
from .extract import save_embedding_files

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    base_args = parse_args()

    run_id = parse_run_id()
    seed = base_args['seed'] + run_id
    output_name = base_args['outputf'].replace('.pkl', f'_seed{seed}.pkl')

    print(f"\n🚀 [RUN {run_id+1}] seed={seed}, output={output_name}")

    save_embedding_files(
        netf=base_args['netf'],
        outputf=output_name,
        nodetypef=base_args['nodetypef'],
        tp_factor=base_args['tp_factor'],
        seed=seed,
        weighted=base_args['weighted'],
        directed=base_args['directed'],
        workers=base_args['workers'],
        net_delimiter=base_args['net_delimiter'],
        max_genes=base_args['max_genes'],
        pairf=base_args['pairf'],
        dataset_dir=base_args["dataset_dir"],   
        atc_file=base_args.get("atc_file"),    
)
