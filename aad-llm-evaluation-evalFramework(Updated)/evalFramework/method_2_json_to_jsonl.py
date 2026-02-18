
import json

def run(in_json: str, out_jsonl: str):
    payload = json.load(open(in_json, "r", encoding="utf-8"))
    scenarios = payload["scenarios"]

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps({
                "id": s["id"],
                "input": s["prompt"],
                "reference": s.get("reference"),
                "metadata": s.get("metadata", {})
            }, ensure_ascii=False) + "\n")

    return out_jsonl
