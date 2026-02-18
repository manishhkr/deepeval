## RAG Evaluation – Short Guide

### 1. Install dependencies

From the project root:

```bash
python -m pip install --upgrade pip
pip install openai deepeval utils
```

### 2. Set environment variables

In the same terminal (PowerShell example):

```powershell
$env:OPENAI_API_KEY = "sk-REPLACE_WITH_YOUR_KEY"
$env:CONFIDENT_METRIC_LOGGING_VERBOSE = "0"   # optional: less logging
```

### 3. Verify the setup

```bash
python test_changes.py
```

All checks should show **OK**.

### 4. Run the sample RAG evaluation

```bash
python -m test.test_rag_only
```

You will see three test cases (from `test/rag_testcases.jsonl`) and their scores:

- Answer Relevancy
- Faithfulness
- Contextual Precision
- Contextual Recall
- Contextual Relevancy

All scores are between **0.0 and 1.0** (higher is better).

### 5. Use your own test cases

Edit `test/rag_testcases.jsonl` and add one JSON object per line with:

- `input`
- `actual_output`
- `retrieval_context` (list of strings)
- `expected_output`

Then rerun:

```bash
python -m test.test_rag_only
```
*** End ***
