import pandas as pd
import json
import re

xlsx = "EmailTransferData.xlsx"

def norm(s):
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# 1) Print sheets so you can pick the right one
xl = pd.ExcelFile(xlsx)
print("Sheets:", xl.sheet_names)

# 2) Read first sheet with header=0 (default) and show what pandas sees
df = pd.read_excel(xlsx, sheet_name=xl.sheet_names[0], header=0)
print("\n--- Columns pandas sees ---")
print([repr(c) for c in df.columns])
print("\n--- First 5 rows ---")
print(df.head(5))

# 3) Build normalized column map
col_map = {norm(c): c for c in df.columns}

def pick_by_synonyms(synonyms):
    # exact normalized match
    for s in synonyms:
        key = norm(s)
        if key in col_map:
            return col_map[key]
    # contains match (handles weird headers)
    for s in synonyms:
        key = norm(s)
        for k, original in col_map.items():
            if key in k:
                return original
    return None

# Broad synonyms — works even if your headers are different
prompt_col = pick_by_synonyms(["prompt", "question", "query", "user question", "input"])
expected_col = pick_by_synonyms(["expected response", "expected", "reference", "gold", "ground truth"])
answer_col = pick_by_synonyms(["mcp", "ai response", "model response", "response", "answer", "output"])

if not prompt_col or not expected_col or not answer_col:
    raise RuntimeError(
        "Could not auto-detect columns.\n"
        f"Detected prompt_col={prompt_col}, expected_col={expected_col}, answer_col={answer_col}\n"
        "Look at the printed Columns list and update synonyms in the script accordingly."
    )

print("\nUsing columns:")
print("  Prompt   :", prompt_col)
print("  Expected :", expected_col)
print("  Answer   :", answer_col)

# 4) Generate scenarios.json + responses.jsonl
scenarios = []
responses = []

for i, row in df.iterrows():
    prompt = str(row.get(prompt_col, "")).strip()
    expected = str(row.get(expected_col, "")).strip()
    answer = str(row.get(answer_col, "")).strip()

    if not prompt:
        continue

    sid = f"Q_{len(scenarios)+1}"
    scenarios.append({"id": sid, "prompt": prompt, "reference": expected})
    responses.append({"id": sid, "answer": answer})

with open("scenarios.json", "w", encoding="utf-8") as f:
    json.dump({"scenarios": scenarios}, f, indent=2, ensure_ascii=False)

with open("responses.jsonl", "w", encoding="utf-8") as f:
    for r in responses:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\n✔ Created scenarios.json and responses.jsonl")
