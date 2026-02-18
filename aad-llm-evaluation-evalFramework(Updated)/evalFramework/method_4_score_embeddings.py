
import json, math, time
from typing import List, Dict
from openai import OpenAI

import os

def _load_env_file(env_file: str = ".env") -> None:
    """
    Best-effort .env loader.
    - If python-dotenv is installed, use it.
    - Otherwise, parse KEY=VALUE lines manually.
    Does NOT override existing environment variables.
    """
    if not env_file:
        return

    # Try python-dotenv if available
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_file, override=False)
        return
    except Exception:
        pass

    # Manual fallback (no dependency)
    if not os.path.exists(env_file):
        return

    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb) if na and nb else 0.0

def run(canonical_json: str,
        answers_jsonl: str,
        out_results_jsonl: str,
        out_results_json: str,
        threshold: float = 0.8):
    if not os.getenv("OPENAI_API_KEY"):
        _load_env_file(".env")

    client = OpenAI()

    client = OpenAI()
    with open(canonical_json, "r", encoding="utf-8") as fh:
        scenarios = {s["id"]: s for s in json.load(fh)["scenarios"]}

    results = []

    with open(answers_jsonl, "r", encoding="utf-8") as f_in,          open(out_results_jsonl, "w", encoding="utf-8") as f_out:

        for line in f_in:
            a = json.loads(line)
            ref = scenarios[a["id"]].get("reference")

            sim = None
            passed = None
            if ref:
                e1 = client.embeddings.create(model="text-embedding-3-small", input=a["answer"]).data[0].embedding
                e2 = client.embeddings.create(model="text-embedding-3-small", input=ref).data[0].embedding
                sim = cosine(e1, e2)
                passed = sim >= threshold
            prompt = scenarios[a["id"]].get("prompt", "")
            row = {
                **a,
                "prompt": prompt,
                "reference": ref,
                "similarity": sim,
                "threshold": threshold,
                "passed": passed
            }
            results.append(row)
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return out_results_json
