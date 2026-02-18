import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI


SYSTEM = (
    "You are an evaluation judge. "
    "Assess whether the assistant answer is grounded in the provided reference. "
    "Return STRICT JSON only."
)


def _load_env_file(path_or_folder: Optional[str]) -> None:
    """
    Loads KEY=VALUE pairs into os.environ if not already set.
    - If given a folder, looks for .env inside it.
    - Does NOT override existing env vars.
    """
    if not path_or_folder:
        return

    env_path = path_or_folder
    if os.path.isdir(env_path):
        env_path = os.path.join(env_path, ".env")

    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and (k not in os.environ or os.environ.get(k) == ""):
                os.environ[k] = v


def _as_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        t = x.strip().lower()
        if t in ("true", "yes", "y", "1"):
            return True
        if t in ("false", "no", "n", "0"):
            return False
    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def judge_hallucination_traceability(
    client: OpenAI,
    prompt: str,
    expected: str,
    answer: str,
    judge_model: str,
) -> Dict[str, Any]:
    user = f"""
PROMPT:
{prompt}

EXPECTED RESPONSE:
{expected}

MODEL ANSWER:
{answer}

Return JSON only with:
- hallucination_success (boolean)
- hallucination_reason (string)
- traceability_geval_success (boolean)
- traceability_geval_reason (string)
"""

    t0 = time.time()
    resp = client.responses.create(
        model=judge_model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    latency_ms = int((time.time() - t0) * 1000)

    text = getattr(resp, "output_text", "") or ""
    data = _extract_json(text) or {}

    return {
        "hallucination_success": _as_bool(data.get("hallucination_success")),
        "hallucination_reason": (data.get("hallucination_reason") or "").strip(),
        "traceability_geval_success": _as_bool(data.get("traceability_geval_success")),
        "traceability_geval_reason": (data.get("traceability_geval_reason") or "").strip(),
        "judge_latency_ms": latency_ms,
        "judge_model": judge_model,
    }


def run(
    results_json: str,
    out_results_json: str,
    out_results_jsonl: str,
    judge_model: str = "gpt-4o-mini",
    env_file: Optional[str] = None,
    project_folder: Optional[str] = None,
):
    # load env from project folder first (./configs/MCP/.env), then optional explicit env_file
    _load_env_file(project_folder)
    _load_env_file(env_file)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Provide it via environment, --env-file, or --project-folder"
        )

    client = OpenAI()

    rows: List[Dict[str, Any]] = json.load(open(results_json, "r", encoding="utf-8"))

    updated = 0
    for r in rows:
        prompt = r.get("prompt", "")
        expected = r.get("reference", "")  # NOTE: uses 'reference' field from results
        answer = r.get("answer", "")

        if not expected or not answer:
            continue

        j = judge_hallucination_traceability(client, prompt, expected, answer, judge_model)
        r.update(j)
        updated += 1

    with open(out_results_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(out_results_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Hallucination + Traceability scored for {updated}/{len(rows)} prompts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--out-results-json", required=True)
    ap.add_argument("--out-results-jsonl", required=True)
    ap.add_argument("--judge-model", default="gpt-4o-mini")
    ap.add_argument("--env-file", default=None, help="Path to .env (optional)")
    ap.add_argument(
        "--project-folder",
        default=None,
        help="Folder containing .env (e.g. ./configs/MCP)",
    )
    args = ap.parse_args()

    run(
        results_json=args.results_json,
        out_results_json=args.out_results_json,
        out_results_jsonl=args.out_results_jsonl,
        judge_model=args.judge_model,
        env_file=args.env_file,
        project_folder=args.project_folder,
    )
