# make_offline_report.py
# Offline HTML report generator (no JS/CDNs). Safe string building (no nested f-strings).
# Supports:
# - Embedding similarity KPIs (similarity/passed)
# - Latency KPIs (gen_latency_ms, emb_latency_ms)
# - Behavior KPIs (deflection/clarifying heuristic)
# - DeepEval KPIs (deepeval_score, deepeval_passed)
# - Grounding KPIs (hallucination_success, traceability_geval_success)
#
# Expected row keys (per prompt) in results.json:
# id, prompt, reference, answer
# similarity (float), passed (bool)
# gen_latency_ms (int) optional, emb_latency_ms (int) optional
# deepeval_score (float) optional, deepeval_passed (bool) optional
# hallucination_success (bool) optional, traceability_geval_success (bool) optional
# hallucination_reason/traceability_geval_reason optional

import html
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    v = sorted(values)
    idx = int(math.ceil(0.95 * len(v))) - 1
    idx = max(0, min(idx, len(v) - 1))
    return float(v[idx])


def fmt3(v: Any) -> str:
    return "—" if v is None else f"{float(v):.3f}"


def fmt_ms(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{int(v)} ms"
    except Exception:
        return "—"


def pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _is_bool(x: Any) -> bool:
    return x in (True, False)


def _rate(trues: int, denom: int) -> Optional[float]:
    if denom <= 0:
        return None
    return trues / denom


# ------------------------
# Charts (pure HTML/CSS)
# ------------------------

def vbar_chart(labels: List[str], values: List[float], mode: str = "rate") -> str:
    """
    Offline vertical bar chart using only HTML/CSS.
    mode="rate": values are 0..1
    mode="ms": values are milliseconds
    """
    if not labels or not values:
        return "<div class='muted'>No data</div>"

    vals = [float(v or 0) for v in values]
    if mode == "ms":
        max_v = max(vals) if vals else 0.0
        scale = max_v if max_v > 0 else 1.0
        fmt = lambda x: f"{int(round(x))} ms"
    else:
        scale = 1.0
        fmt = lambda x: f"{x * 100:.1f}%"

    bars = []
    for lab, v in zip(labels, vals):
        h = 0 if scale == 0 else int(round(100 * (v / scale)))
        h = max(0, min(100, h))
        bars.append(
            "<div class='vbar'>"
            f"  <div class='vbar-col'><div class='vbar-fill' style='height:{h}%'></div></div>"
            f"  <div class='vbar-lab'>{esc(lab)}</div>"
            f"  <div class='vbar-val'>{esc(fmt(v))}</div>"
            "</div>"
        )

    return "<div class='vbar-chart'>" + "".join(bars) + "</div>"


def details_block(title: str, body_html: str) -> str:
    if not body_html.strip():
        return ""
    return (
        "<details class='details'>"
        f"<summary>{esc(title)}</summary>"
        f"{body_html}"
        "</details>"
    )


# ------------------------
# KPI aggregations
# ------------------------

def kpi_embeddings(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sims = [r.get("similarity") for r in rows if isinstance(r.get("similarity"), (int, float))]
    passed = [r for r in rows if r.get("passed") is True]
    denom = [r for r in rows if _is_bool(r.get("passed"))]
    return {
        "avg": statistics.mean(sims) if sims else None,
        "p95": _p95([float(x) for x in sims]) if sims else None,
        "max": max([float(x) for x in sims]) if sims else None,
        "pass_rate": _rate(len(passed), len(denom)),
        "scored": len(sims),
    }

def _get_gen_latency(r):
    if isinstance(r.get("gen_latency_ms"), (int, float)):
        return r.get("gen_latency_ms")
    if isinstance(r.get("timing_ms", {}).get("generation"), (int, float)):
        return r["timing_ms"]["generation"]
    return None

def _get_emb_latency(r):
    if isinstance(r.get("emb_latency_ms"), (int, float)):
        return r.get("emb_latency_ms")
    if isinstance(r.get("timing_ms", {}).get("embedding"), (int, float)):
        return r["timing_ms"]["embedding"]
    return None
    
def kpi_latency(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gen = [_get_gen_latency(r) for r in rows if _get_gen_latency(r) is not None]
    emb = [_get_emb_latency(r) for r in rows if _get_emb_latency(r) is not None]
    gen_f = [float(x) for x in gen]
    emb_f = [float(x) for x in emb]
    return {
        "avg_gen": statistics.mean(gen_f) if gen_f else None,
        "p95_gen": _p95(gen_f) if gen_f else None,
        "max_gen": max(gen_f) if gen_f else None,
        "avg_emb": statistics.mean(emb_f) if emb_f else None,
        "p95_emb": _p95(emb_f) if emb_f else None,
        "max_emb": max(emb_f) if emb_f else None,
    }


def _looks_like_clarifying(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return False
    # heuristic: question marks + common clarification phrasing
    if "?" in a and any(x in a for x in ["could you", "can you", "which", "what", "please clarify", "do you mean", "to confirm"]):
        return True
    return False


def _looks_like_deflection(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return False
    # heuristic: refusal/deflection patterns
    patterns = [
        "i can’t help", "i can't help", "i cannot help",
        "i’m not able", "i'm not able", "unable to",
        "i can’t assist", "i can't assist", "i cannot assist",
        "as an ai", "i don't have access", "i do not have access",
        "i cannot provide", "i can't provide",
    ]
    return any(p in a for p in patterns)


def kpi_behavior(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = 0
    clar = 0
    defl = 0
    for r in rows:
        ans = r.get("answer") or ""
        if not ans.strip():
            continue
        scored += 1
        if _looks_like_clarifying(ans):
            clar += 1
        if _looks_like_deflection(ans):
            defl += 1
    return {
        "scored": scored,
        "clar_rate": _rate(clar, scored),
        "defl_rate": _rate(defl, scored),
    }


def kpi_deepeval(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r.get("deepeval_score") for r in rows if isinstance(r.get("deepeval_score"), (int, float))]
    denom = [r for r in rows if _is_bool(r.get("deepeval_passed"))]
    passed = [r for r in denom if r.get("deepeval_passed") is True]
    if not scores and not denom:
        return {"available": False}
    return {
        "available": True,
        "avg": statistics.mean([float(x) for x in scores]) if scores else None,
        "p95": _p95([float(x) for x in scores]) if scores else None,
        "max": max([float(x) for x in scores]) if scores else None,
        "pass_rate": _rate(len(passed), len(denom)),
        "scored": len(scores),
    }


def kpi_grounding(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    denom_h = [r for r in rows if _is_bool(r.get("hallucination_success"))]
    denom_t = [r for r in rows if _is_bool(r.get("traceability_geval_success"))]
    h_pass = [r for r in denom_h if r.get("hallucination_success") is True]
    t_pass = [r for r in denom_t if r.get("traceability_geval_success") is True]
    if not denom_h and not denom_t:
        return {"available": False}
    return {
        "available": True,
        "hallucination_rate": _rate(len(h_pass), len(denom_h)),
        "traceability_rate": _rate(len(t_pass), len(denom_t)),
        "hallucination_scored": len(denom_h),
        "traceability_scored": len(denom_t),
    }


def worst_rows(rows: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
    # Sort by similarity ascending (missing similarity treated as 1.0)
    def key(r):
        s = r.get("similarity")
        return float(s) if isinstance(s, (int, float)) else 1.0
    return sorted(rows, key=key)[:n]


# ------------------------
# Main report generator
# ------------------------

def generate_offline_dashboard(rows: List[Dict[str, Any]], out_html: str) -> None:
    emb = kpi_embeddings(rows)
    lat = kpi_latency(rows)
    beh = kpi_behavior(rows)
    dev = kpi_deepeval(rows)
    grd = kpi_grounding(rows)

    worst = worst_rows(rows, 10)

    # --- Build cards safely (no nested f-strings) ---

    # Embedding KPIs card
    card_embedding = f"""
    <div class="card">
      <h2>Embedding Similarity KPIs</h2>
      <div class="kpis">
        <div class="kpi"><div class="v">{fmt3(emb["avg"])}</div><div class="l">Avg Similarity</div></div>
        <div class="kpi"><div class="v">{fmt3(emb["p95"])}</div><div class="l">P95 Similarity</div></div>
        <div class="kpi"><div class="v">{fmt3(emb["max"])}</div><div class="l">Max Similarity</div></div>
        <div class="kpi"><div class="v">{pct(emb["pass_rate"])}</div><div class="l">Pass Rate</div></div>
      </div>
      <div class="muted" style="margin-top:10px">Scored prompts: {emb["scored"]}</div>
      <div style="margin-top:12px">
        {vbar_chart(["Pass Rate"], [emb["pass_rate"] or 0.0], mode="rate")}
      </div>
    </div>
    """

    # Behavior KPIs card
    card_behavior = f"""
    <div class="card">
      <h2>Behavior KPIs</h2>
      {vbar_chart(["Deflection", "Clarifying Q"], [beh["defl_rate"] or 0.0, beh["clar_rate"] or 0.0], mode="rate")}
      <div class="muted" style="margin-top:10px">Scored prompts: {beh["scored"]}</div>
    </div>
    """

    # Latency KPIs card
    card_latency = f"""
    <div class="card">
      <h2>Latency KPIs</h2>
      <div class="kpis">
        <div class="kpi"><div class="v">{fmt_ms(lat["avg_gen"])}</div><div class="l">Avg Gen</div></div>
        <div class="kpi"><div class="v">{fmt_ms(lat["p95_gen"])}</div><div class="l">P95 Gen</div></div>
        <div class="kpi"><div class="v">{fmt_ms(lat["max_gen"])}</div><div class="l">Max Gen</div></div>
        <div class="kpi"><div class="v">{fmt_ms(lat["avg_emb"])}</div><div class="l">Avg Emb</div></div>
      </div>
      <div style="margin-top:12px">
        {vbar_chart(["Avg Gen", "P95 Gen", "Max Gen"], [lat["avg_gen"] or 0.0, lat["p95_gen"] or 0.0, lat["max_gen"] or 0.0], mode="ms")}
      </div>
    </div>
    """

    # DeepEval card (optional)
    card_deepeval = ""
    if dev.get("available"):
        card_deepeval = f"""
        <div class="card">
          <h2>DeepEval KPIs</h2>
          <div class="kpis">
            <div class="kpi"><div class="v">{fmt3(dev["avg"])}</div><div class="l">Avg Score</div></div>
            <div class="kpi"><div class="v">{fmt3(dev["p95"])}</div><div class="l">P95 Score</div></div>
            <div class="kpi"><div class="v">{fmt3(dev["max"])}</div><div class="l">Max Score</div></div>
            <div class="kpi"><div class="v">{pct(dev["pass_rate"])}</div><div class="l">Pass Rate</div></div>
          </div>
          <div class="muted" style="margin-top:10px">Scored prompts: {dev["scored"]}</div>
          <div style="margin-top:12px">
            {vbar_chart(["Pass Rate"], [dev["pass_rate"] or 0.0], mode="rate")}
          </div>
        </div>
        """

    # Grounding card (optional; requires Method 7 outputs)
    card_grounding = ""
    if grd.get("available"):
        card_grounding = f"""
        <div class="card">
          <h2>Grounding KPIs</h2>
          {vbar_chart(
              ["Hallucination (No extra claims)", "Traceability (Grounded in expected)"],
              [grd["hallucination_rate"] or 0.0, grd["traceability_rate"] or 0.0],
              mode="rate"
          )}
          <div class="muted" style="margin-top:10px">
            Scored: Hallucination {grd["hallucination_scored"]} • Traceability {grd["traceability_scored"]}
          </div>
        </div>
        """

    # Worst prompts table (keep it familiar + add judge flags if present)
    worst_rows_html = []
    for r in worst:
        hallu = r.get("hallucination_success")
        trace = r.get("traceability_geval_success")
        hallu_icon = "—" if not _is_bool(hallu) else ("✅" if hallu else "❌")
        trace_icon = "—" if not _is_bool(trace) else ("✅" if trace else "❌")

        reason_html = ""
        if isinstance(r.get("hallucination_reason"), str) and r["hallucination_reason"].strip():
            reason_html += f"<div><b>Hallucination:</b> {esc(r['hallucination_reason'])}</div>"
        if isinstance(r.get("traceability_geval_reason"), str) and r["traceability_geval_reason"].strip():
            reason_html += f"<div><b>Traceability:</b> {esc(r['traceability_geval_reason'])}</div>"

        reasons = details_block("Judge reasons", reason_html)

        worst_rows_html.append(f"""
          <tr>
            <td>{esc(r.get("id",""))}</td>
            <td class="mono">{fmt3(r.get("similarity"))}</td>
            <td>{'✅' if r.get("passed") is True else ('❌' if r.get("passed") is False else '—')}</td>
            <td class="mono">{fmt3(r.get("deepeval_score"))}</td>
            <td>{'✅' if r.get("deepeval_passed") is True else ('❌' if r.get("deepeval_passed") is False else '—')}</td>
            <td>{hallu_icon}</td>
            <td>{trace_icon}</td>
            <td class="wrap">{esc(r.get("prompt",""))}</td>
            <td class="wrap">{esc(r.get("reference",""))}</td>
            <td class="wrap">{esc(r.get("answer",""))}{reasons}</td>
          </tr>
        """)

    card_worst = f"""
    <div class="card">
      <h2>Worst Prompts (Lowest Similarity)</h2>
      <div style="overflow-x:auto; margin-top:10px">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Sim</th>
              <th>Sim Pass</th>
              <th>DeepEval</th>
              <th>DE Pass</th>
              <th>Halluc.</th>
              <th>Trace</th>
              <th>Prompt</th>
              <th>Expected</th>
              <th>Answer</th>
            </tr>
          </thead>
          <tbody>
            {''.join(worst_rows_html)}
          </tbody>
        </table>
      </div>
      <div class="muted" style="margin-top:10px">
        Note: Hallucination/Traceability appear only if you ran Method 7 (judge scorer).
      </div>
    </div>
    """

    # KPI definitions (business-friendly)
    defs = """
    <div class="card">
      <h2>KPI Definitions</h2>
      <div class="muted">
        <b>Embedding Similarity</b>: semantic closeness between expected and model answer using embeddings + cosine similarity.<br/>
        <b>Pass Rate</b>: percent of prompts meeting the configured threshold.<br/><br/>

        <b>Behavior KPIs</b>: heuristic detection based on response text patterns:<br/>
        • <b>Deflection</b>: refusal/decline language (e.g., “I can’t help with that”).<br/>
        • <b>Clarifying Q</b>: response asks clarifying question(s).<br/><br/>

        <b>Latency KPIs</b>:<br/>
        • <b>Avg Gen / P95 Gen / Max Gen</b>: generation latency from the model invocation step (ms).<br/>
        • <b>Avg Emb</b>: embedding latency (ms).<br/><br/>

        <b>DeepEval KPIs</b>: optional LLM-judge score if present in results.json.<br/><br/>

        <b>Grounding KPIs</b> (optional, requires Method 7):<br/>
        • <b>Hallucination (No extra claims)</b>: TRUE if answer does not introduce claims beyond expected reference.<br/>
        • <b>Traceability</b>: TRUE if key claims in answer can be traced back to expected reference.
      </div>
    </div>
    """

    # --- Final HTML (CSS uses doubled braces because we use f-string here) ---
    css = """
    body { font-family: Arial, sans-serif; background: #fafafa; margin: 0; padding: 18px; }
    h1 { margin: 0 0 14px 0; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
    .card { background: #fff; border: 2px solid #444; border-radius: 14px; padding: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
    .kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
    .kpi { border: 2px solid #777; border-radius: 12px; padding: 10px; background: #f7f7f7; text-align: center; }
    .kpi .v { font-size: 20px; font-weight: 800; }
    .kpi .l { font-size: 12px; color: #555; margin-top: 2px; }
    .muted { color: #666; font-size: 12px; line-height: 1.45; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border: 1px solid #ddd; padding: 8px; font-size: 12px; vertical-align: top; }
    th { background: #f0f0f0; text-align: left; position: sticky; top: 0; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; white-space: nowrap; }
    .wrap { max-width: 380px; white-space: pre-wrap; word-break: break-word; }
    .details summary { cursor: pointer; font-weight: 700; margin-top: 8px; }
    .details div { margin-top: 8px; }

    .vbar-chart { display:flex; align-items:flex-end; gap: 14px; height: 220px; padding: 10px 4px; overflow-x: auto; }
    .vbar { width: 110px; text-align: center; }
    .vbar-col { height: 170px; border: 1px solid #ddd; border-radius: 12px; background: #f7f7f7; display:flex; align-items:flex-end; overflow:hidden; }
    .vbar-fill { width:100%; background: #4a90e2; }
    .vbar-lab { margin-top: 6px; font-size: 11px; color: #333; }
    .vbar-val { font-size: 11px; color: #666; }
    """

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>AI Evaluation Report (Offline)</title>
  <style>{css}</style>
</head>
<body>
  <h1>AI Evaluation Report (Offline)</h1>

  <div class="grid">
    {card_embedding}
    {card_behavior}
    {card_latency}
    {card_deepeval}
    {card_grounding}
  </div>

  {card_worst}

  {defs}

</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_doc)
