# carepath/prompts.py
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd


def _find_msi_file(keys: List[str], search_root: str) -> Optional[str]:
    root = Path(search_root)
    if not root.exists():
        return None
    best, best_score = None, -1
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".tsv", ".csv", ".txt"}:
            continue
        name = p.name.lower()
        score = sum(1 for k in keys if k in name)
        if score >= len(keys) and score > best_score:
            best, best_score = str(p), score
    return best


def build_id_to_name_mapping(
    drug_tsv: str = None,
    disease_tsv: str = None,
    ppi_tsv: str = None,
    search_root: str = "data/raw/msi/extracted",
) -> Dict[str, str]:
    if drug_tsv is None:
        drug_tsv = _find_msi_file(["drug", "protein"], search_root)
    if disease_tsv is None:
        disease_tsv = _find_msi_file(["indication", "protein"], search_root)
    if ppi_tsv is None:
        ppi_tsv = _find_msi_file(["protein", "protein"], search_root)

    id2name: Dict[str, str] = {}

    for path in (drug_tsv, disease_tsv, ppi_tsv):
        if path is None:
            continue
        df = pd.read_csv(path, sep="\t")
        for _, row in df.iterrows():
            id2name[str(row["node_1"]).strip()] = str(row["node_1_name"]).strip()
            id2name[str(row["node_2"]).strip()] = str(row["node_2_name"]).strip()

    return id2name


def path_to_prompt(path: List[str], id2name: Dict[str, str], node2type: Dict[str, str]) -> str:
    drug_nodes = [n for n in path if node2type.get(n) == "drug"]
    disease_nodes = [n for n in path if node2type.get(n) == "disease"]
    gene_nodes = [n for n in path if (node2type.get(n) == "gene") or str(n).startswith("G")]
    nm = lambda x: id2name.get(x, x)
    disease_name = nm(disease_nodes[0]) if disease_nodes else nm(path[0])
    drug_name = nm(drug_nodes[-1]) if drug_nodes else nm(path[-1])
    genes = sorted({nm(g) for g in gene_nodes}, key=lambda s: str(s).lower())
    genes_text = ", ".join(genes) if genes else "none"
    return (
        f"Premise: {disease_name} involves genes {genes_text}.\n"
        f"Hypothesis: {drug_name} can be repurposed to treat {disease_name}.\n"
        f"Label:"
    )
