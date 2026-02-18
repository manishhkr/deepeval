import json
from make_offline_report import generate_offline_dashboard

def run(results_json: str, out_html: str):
    rows = json.load(open(results_json, "r", encoding="utf-8"))
    generate_offline_dashboard(rows, out_html)
