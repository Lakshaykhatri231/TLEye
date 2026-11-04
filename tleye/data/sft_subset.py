"""Fixed test subset for explanation generation: at most 40 rows per problem plus ALL usable test-split certificate-pilot rows; 3,000 of them (in_B) get counterfactuals.
    python -I -m tleye.data.sft_subset      -> data/processed/kaggle_sidecar/sft_test_subset.parquet"""
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    d = pd.read_parquet("data/processed/kaggle_pack/tleye_data.parquet", columns=["submission_id", "problem_id", "split"]); te = d[d.split == "test"]
    capped = te.groupby("problem_id", group_keys=False).sample(frac=1, random_state=0).groupby("problem_id").head(40)
    pil = pd.concat([pd.DataFrame([json.loads(x) for x in open(f"data/processed/{f}.jsonl")]) for f in ["cert_pilot", "cert_pilot_confirm", "cert_pilot3", "cert_pilot4", "cert_pilot5", "cert_pilot6"]])
    pil = pil[~pil.status.isin(["compile_error", "missing_dependency", "infra_error"]) & pil.submission_id.isin(te.submission_id)]
    s = pd.DataFrame({"submission_id": sorted(set(capped.submission_id) | set(pil.submission_id))}).sample(frac=1, random_state=0).reset_index(drop=True)
    s["in_B"] = False; s.loc[:2999, "in_B"] = True; s["is_pilot"] = s.submission_id.isin(pil.submission_id)
    s.to_parquet(Path("data/processed/kaggle_sidecar/sft_test_subset.parquet"), index=False)
    print("subset rows", len(s), "| B", int(s.in_B.sum()), "| pilot rows", int(s.is_pilot.sum()))


if __name__ == "__main__":
    main()
