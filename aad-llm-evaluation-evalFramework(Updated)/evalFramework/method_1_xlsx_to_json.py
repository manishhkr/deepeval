
import pandas as pd
import json
import os

ID_COL = "No."
PROMPT_COL = "Prompt"

def run(xlsx_path: str, out_json: str, sheet: str, reference_col: str):
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    scenarios = []
    for i, row in df.iterrows():
        sid = f"MCP_{row.get(ID_COL, i+1)}"
        scenarios.append({
            "id": sid,
            "prompt": str(row.get(PROMPT_COL, "")).strip(),
            "reference": str(row.get(reference_col, "")).strip(),
            "metadata": {
                "sheet": sheet,
                "row": int(i)
            }
        })

    payload = {
        "dataset_type": "mcp_eval",
        "scenarios": scenarios
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return out_json
