# method_3_run_model.py (replace or add)
import json
import os
import time
from typing import List, Dict, Any, Optional

import requests  # add to requirements if not there

# try python-dotenv but provide fallback
def _load_env_file(env_dir: str):
    env_path = os.path.join(env_dir, ".env")
    if not os.path.exists(env_path):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    # minimal manual loader
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

# === provider-specific invocations ===

def call_openai(prompt: str, model: str) -> str:
    # keep existing behavior, but use env if present
    from openai import OpenAI
    client = OpenAI()
    resp = client.responses.create(model=model, input=prompt)
    return getattr(resp, "output_text", "") or ""

import os
import json
import requests

def call_dataiku(prompt: str, api_url: str, api_key: str, extra=None) -> str:
    import os
    import json
    import requests

    auth_mode = os.getenv("DATAIKU_AUTH_MODE", "bearer").lower()

    headers = {"Content-Type": "application/json"}
    if auth_mode == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
    elif auth_mode == "authorization":
        headers["Authorization"] = api_key
    elif auth_mode == "x-api-key":
        headers["X-API-Key"] = api_key
    else:
        raise ValueError(f"Unsupported DATAIKU_AUTH_MODE: {auth_mode}")

    # --- payload mapping ---
    prompt_param = os.getenv("DATAIKU_PROMPT_PARAM", "input")
    threshold_param = os.getenv("DATAIKU_THRESHOLD_PARAM")
    threshold_value = os.getenv("DATAIKU_THRESHOLD_VALUE")

    body = {
        prompt_param: prompt
    }

    if threshold_param and threshold_value is not None:
        try:
            body[threshold_param] = float(threshold_value)
        except ValueError:
            body[threshold_param] = threshold_value

    if extra:
        body.update(extra)

    r = requests.post(api_url, headers=headers, json=body, timeout=60)

    if r.status_code == 403:
        raise RuntimeError(
            f"403 Forbidden from DataIKU\n"
            f"AUTH_MODE={auth_mode}\n"
            f"URL={api_url}\n"
            f"BODY={json.dumps(body)}\n"
            f"RESPONSE={r.text[:300]}"
        )

    r.raise_for_status()
    j = r.json()

    # flexible response parsing
    for key in ("output", "answer", "response", "result"):
        if key in j:
            return j[key]

    return json.dumps(j, ensure_ascii=False)
def _extract_dataiku_text(resp_obj) -> str:
    """
    DataIKU returns: {"code":200,"body":"\"{\\\"...\\\"}\"",...}
    We want a readable text string for scoring (embeddings/judge/report).
    """
    import json

    if resp_obj is None:
        return ""
    if isinstance(resp_obj, str):
        return resp_obj

    if isinstance(resp_obj, dict):
        body = resp_obj.get("body", "")
        # body is often a JSON string inside quotes, with escapes
        if isinstance(body, str):
            try:
                # First decode outer JSON string (removes the wrapping quotes)
                unquoted = json.loads(body)
                # Now unquoted is a JSON string; parse into dict
                j = json.loads(unquoted) if isinstance(unquoted, str) else unquoted
            except Exception:
                # If parsing fails, just return raw body
                return body

            # If it has matches, join the top documents as "answer text"
            matches = j.get("matches")
            if isinstance(matches, list) and matches:
                docs = []
                for m in matches[:5]:
                    doc = (m.get("document") or "").strip()
                    if doc:
                        docs.append(doc)
                return "\n\n".join(docs) if docs else json.dumps(j, ensure_ascii=False)

            return json.dumps(j, ensure_ascii=False)

        return json.dumps(resp_obj, ensure_ascii=False)

    # fallback
    return str(resp_obj)


def run(canonical_json: str,
        provider: str,
        out_answers_jsonl: str,
        out_answers_json: str,
        model: str = "gpt-5-mini",
        project_folder: Optional[str] = None,
        **kwargs):
    """
    provider: legacy value 'openai' or 'project' (when using project_folder)
    project_folder: path to config folder containing .env (recommended)
    """
    # load scenarios
    payload = json.load(open(canonical_json, "r", encoding="utf-8"))
    scenarios = payload.get("scenarios", payload)  # support old format

    # if project folder provided, load its .env (but don't override existing env)
    if project_folder:
        _load_env_file(project_folder)

    # decide provider
    prov = os.getenv("PROVIDER", provider or "openai").lower()

    # make sure provider-specific vars exist
    if prov == "openai":
        # ensure OPENAI_API_KEY present or will raise when OpenAI() called
        pass
    elif prov == "dataiku":
        api_url = os.getenv("DATAIKU_API_URL")
        api_key = os.getenv("DATAIKU_API_KEY")
        if not api_url or not api_key:
            raise RuntimeError("DataIKU provider selected but DATAIKU_API_URL/DATAIKU_API_KEY not found in environment or project folder.")
    else:
        raise ValueError(f"Unsupported provider: {prov}")

    rows: List[Dict[str, Any]] = []
    with open(out_answers_jsonl, "w", encoding="utf-8") as f:
        for s in scenarios:
            sid = s.get("id")
            prompt = s.get("prompt") or s.get("input") or ""
            t0 = time.time()
            if prov == "openai":
                answer = call_openai(prompt, model)
            elif prov == "dataiku":
                api_url = os.getenv("DATAIKU_API_URL")
                api_key = os.getenv("DATAIKU_API_KEY")
                # optionally allow per-scenario override
                extra = s.get("metadata", {}).get("dataiku_params")
                raw = call_dataiku(prompt, api_url, api_key, extra=extra)
                answer = _extract_dataiku_text(raw)

            else:
                raise ValueError("Unsupported provider")

            dt_ms = int((time.time() - t0) * 1000)
            row = {
                "id": sid,
                "prompt": prompt,
                "answer": answer,
                "timing_ms": {"generation": dt_ms},
                "metadata": s.get("metadata", {}),
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_answers_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    return out_answers_jsonl, out_answers_json
