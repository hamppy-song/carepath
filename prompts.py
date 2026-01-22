# carepath/prompts.py
from typing import Dict, List


def path_to_prompt(path: List[str], id2name: Dict[str, str], node2type: Dict[str, str]) -> str:
    """
    NLI prompt:
      Premise: {disease} involves genes {g1, ..., gk}.
      Hypothesis: {drug} can be repurposed to treat {disease}.
      Label:
    """
    drug_nodes = [n for n in path if node2type.get(n) == "drug"]
    disease_nodes = [n for n in path if node2type.get(n) == "disease"]
    gene_nodes = [n for n in path if (node2type.get(n) in ("gene", "protein")) or str(n).startswith(("G", "P"))]

    nm = lambda x: id2name.get(x, x)

    disease_name = nm(disease_nodes[0]) if disease_nodes else nm(path[0])
    drug_name = nm(drug_nodes[-1]) if drug_nodes else nm(path[-1])

    genes = sorted({nm(g) for g in gene_nodes}, key=lambda s: s.lower())
    genes_text = ", ".join(genes) if genes else "none"

    prompt = (
        f"Premise: {disease_name} involves genes {genes_text}.\n"
        f"Hypothesis: {drug_name} can be repurposed to treat {disease_name}.\n"
        f"Label:"
    )
    return prompt


def fallback_prompt(disease: str, drug: str, id2name: Dict[str, str]) -> str:
    dname = id2name.get(drug, drug)
    sname = id2name.get(disease, disease)
    return (
        f"Premise: {sname} involves genes none.\n"
        f"Hypothesis: {dname} can be repurposed to treat {sname}.\n"
        f"Label:"
    )
