# CAREPath

**CAREPath (Context-Aware REasoning Path)** is a KG–LLM framework for **drug repurposing** that predicts disease–drug associations by combining:

- **DFS-like constrained semantic path encoding** over short disease–gene–drug paths  
- **BFS-like mechanism context augmentation** from 1-hop gene neighborhoods  

It then fuses these signals with **Node2Vec topology features** and scores pairs using an **XGBoost-based stacking ensemble**.

This repository includes code to:
1) **Extract per-pair embeddings** (semantic path + mechanism context + Node2Vec)  
2) **Run prediction and evaluation** (CV with random/drug/disease splits)

---

## What CAREPath does (high-level)

Given a disease–drug pair *(s, d)*:

### 1) Constrained semantic path encoding (DFS-like)
- Enumerate short simple paths **s → gene(s) → d** with constraints (e.g., max hop=3, limited number of genes).
- Convert each path into an NLI-style prompt:
  - `Premise: {disease} involves genes {g1, ..., gk}.`
  - `Hypothesis: {drug} can be repurposed to treat {disease}.`
  - `Label:`
- Encode each prompt with **BioLinkBERT (CLS)** and aggregate via **max pooling** to obtain a pair-specific semantic path embedding **Z_path(s,d)**.
- If no path exists, use a fallback prompt with `genes none`.

### 2) Mechanism context augmentation (BFS-like)
- Build entity-level context from **1-hop gene/protein neighbors only** (to reduce direct disease–drug leakage).
- Encode neighborhood sentences with BioLinkBERT and mean-pool into initial context embeddings.
- Apply similarity-guided pooling + residual mixing:
  - **Drugs:** pool within ATC-prefix–related drugs
  - **Diseases:** pool via gene-signature similarity (cosine kNN on weighted gene vectors)
- Produces robust context embeddings **Z_ctx^drug(d)** and **Z_ctx^dis(s)**, especially when paths are sparse/noisy.

### 3) Feature fusion + prediction
- Concatenate features:
  - `Node2Vec(drug)`, `Node2Vec(disease)`, `Z_path(s,d)`, `Z_ctx^drug(d)`, `Z_ctx^dis(s)`
- Score with an **XGBoost stacking ensemble** for final association probability.

---

## Repository structure

```text
.
├── MSI dataset/
│   ├── graph.txt
│   ├── nodetypes.tsv                      # node -> type (drug/disease/gene/protein/...)
│   ├── 1_drug_to_protein.tsv
│   ├── 2_indication_to_protein.tsv
│   ├── 3_protein_to_protein.tsv
│   ├── 4_protein_to_biological_function.tsv
│   ├── 5_biological_function_to_biological_function.tsv
│   ├── 7_drug_classification_df.tsv       # ATC codes (for drug pooling / teleport)
│   └── dda_labels.tsv                     # columns: drug, disease, label
│
├── carepath/
│   ├── __init__.py
│   ├── graph_utils.py                     # graph/path utilities + ATC teleport
│   ├── mech_context.py                    # mechanism context + pooling
│   ├── prompts.py                         # id->name mapping + prompt builder
│   └── utils.py                           # read_graph + set_seed
│
├── extract_embeddings/
│   ├── __init__.py
│   ├── config.py                          # CLI args for embedding extraction
│   ├── extract.py                         # main embedding extraction pipeline
│   └── main.py                            # entry point (supports run_id)
│
├── prediction/
│   ├── __init__.py
│   ├── config.py                          # CLI args + optional JSON model config
│   └── train_and_prediction.py            # CV training + evaluation + per-pair outputs
│
└── README.md

Usage

Below are minimal runnable commands you can copy/paste.
Replace paths with your dataset locations.

1) Extract embeddings

This step creates a per-pair embedding dictionary (.pkl) keyed by "{disease}__{drug}".

Example (MSI)
python -m extract_embeddings.main \
  --network_file "MSI dataset/graph.txt" \
  --node_type_file "MSI dataset/nodetypes.tsv" \
  --pair_file "MSI dataset/dda_labels.tsv" \
  --output_file "outputs/msi_embeddings.pkl" \
  --seed 42 \
  --max_genes 5 \
  --workers 5 \
  --run_id 0

2) Train + predict (cross-validation)

This step loads the embedding .pkl and runs CV evaluation.
It saves cv_results.tsv and per-pair prediction files like cv_pred_details_{split}.tsv.

python -m prediction.train_and_prediction \
  --embedding_file "outputs/msi_embeddings.pkl" \
  --pair_file "MSI dataset/dda_labels.tsv" \
  --seed 42 \
  --n_splits 5 \
  --splits "random,drug,disease" \
  --output_file "outputs/cv_results.tsv" \
  --pred_detail_file "outputs/cv_pred_details.tsv"
