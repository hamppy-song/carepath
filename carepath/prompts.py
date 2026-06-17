# carepath/prompts.py
from typing import Dict, List
import pandas as pd


def build_id_to_name_mapping(
    drug_tsv: str = "data/raw/msi/extracted/data/1_drug_to_protein.tsv",
    disease_tsv: str = "data/raw/msi/extracted/data/2_indication_to_protein.tsv",
    ppi_tsv: str = "data/raw/msi/extracted/data/3_protein_to_protein.tsv",
) -> Dict[str, str]:
    id2name = {}
    drug_df = pd.read_csv(drug_tsv, sep="\t")
    for _, row in drug_df.iterrows():
        id2name[str(row["node_1"]).strip()] = str(row["node_1_name"]).strip()
        id2name[str(row["node_2"]).strip()] = str(row["node_2_name"]).strip()
    disease_df = pd.read_csv(disease_tsv, sep="\t")
    for _, row in disease_df.iterrows():
        id2name[str(row["node_1"]).strip()] = str(row["node_1_name"]).strip()
        id2name[str(row["node_2"]).strip()] = str(row["node_2_name"]).strip()
    ppi_df = pd.read_csv(ppi_tsv, sep="\t")
    for _, row in ppi_df.iterrows():
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
