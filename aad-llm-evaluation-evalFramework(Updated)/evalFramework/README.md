## Setup

1. Copy env templates:
   - cp .env.example .env
   - cp configs/MCP/.env.example configs/MCP/.env
   - cp configs/DataIKU/.env.example configs/DataIKU/.env

2. Fill in API keys.

3. Run:
   python orchestrator.py --xlsx <file.xlsx> --sheet <sheet> --project-folder ./configs/MCP


Scenarios -
Input provided with NO AI Response
	python orchestrator.py --xlsx NewPrompts_MCP.xlsx --sheet "SVA-Mini" --project-folder "./configs/MCP"

Input provided with AI Response 
python from_expected_and_answer.py

Embeddings similarity
python run_embeddings_cli.py --canonical-json .\scenarios.json --answers-jsonl .\responses.jsonl --out-results-json .\results.json --out-results-jsonl .\results.jsonl --threshold 0.8

DeepEval
python method_6_score.py --canonical-json scenarios.json --results-json results.json --out-results-json results.json --out-results-jsonl results.jsonl --judge-model gpt-4o-mini

Hallucination + Traceability
python method_7_hallucination_tracebility.py --results-json results.json --out-results-json results.json --out-results-jsonl results.jsonl --project-folder "./configs/MCP" --judge-model gpt-4o-mini

Offline report
python -c "from method_5_offline_report import run; run('results.json', 'report_offline.html')"

