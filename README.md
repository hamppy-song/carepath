# CAREPath

**CAREPath (Context-Aware REasoning Path)** is a KG–LLM framework for **drug repurposing** that predicts disease–drug associations by combining:

- **DFS-like constrained semantic path encoding** over short disease–gene–drug paths, and  
- **BFS-like mechanism context augmentation** from 1-hop gene neighborhoods,

then fusing them with **Node2Vec topology** features and scoring with an XGBoost-based ensemble.

This repository contains the code to (1) extract embeddings and (2) run prediction/evaluation for CAREPath.

---

## What CAREPath does (high-level)

Given a disease–drug pair (s, d):

1. **Constrained semantic path encoding (DFS-like)**
   - Enumerate simple paths from s → … → d under a strict constraint on intermediate genes (short, mechanistic paths).
   - Convert each path into an NLI-style prompt:
     - `Premise: {disease} involves genes {g1, ..., gk}.`
     - `Hypothesis: {drug} can be repurposed to treat {disease}.`
     - `Label:`
   - Encode prompts with **BioLinkBERT** and aggregate by max pooling to obtain a pair-specific semantic path embedding.

2. **Mechanism context augmentation (BFS-like)**
   - Build entity context from 1-hop gene neighborhoods (gene/protein neighbors only; excludes direct disease–drug leakage).
   - Encode neighborhood sentences with BioLinkBERT and mean-pool to form initial context embeddings.
   - Apply similarity-guided pooling + residual mixing:
     - Drugs: pool within ATC-prefix–related drugs
     - Diseases: pool over gene-signature similarity (cosine kNN over weighted gene vectors)
   - Improves robustness when paths are sparse/noisy.

3. **Feature fusion + prediction**
   - Concatenate features:
     - `Node2Vec(drug)`, `Node2Vec(disease)`, `Z_path(s,d)`, `Z_ctx^drug(d)`, `Z_ctx^dis(s)`
   - Score with an XGBoost ensemble for final association probability.


---

## Repository structure


---

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `torch`
- `transformers`
- `xgboost`
- `networkx`
- `node2vec`
- `tqdm`

Install:
```bash
pip install -r requirements.txt
