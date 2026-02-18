from openai import OpenAI
from typing import Optional, List, Dict, Any
import json


class Geval:
    """
    DeepEval-style GEval metric (LLM-as-Judge).
    """
    @staticmethod
    def get_client(api_key: Optional[str] = None):
        if api_key:
            return OpenAI(api_key=api_key)
        return OpenAI()  
    def __init__(
        self,
        name: str,
        criteria: Optional[str] = None,
        evaluation_steps: Optional[List[str]] = None,
        evaluation_params: Optional[List[str]] = None,
        threshold: float = 0.5,
        judge_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        if criteria and evaluation_steps:
            raise ValueError("You can only provide either criteria OR evaluation_steps")

        if not criteria and not evaluation_steps:
            raise ValueError("You must provide either criteria OR evaluation_steps")

        self.name = name
        self.criteria = criteria
        self.evaluation_steps = evaluation_steps
        self.evaluation_params = evaluation_params
        self.threshold = threshold
        self.judge_model = judge_model

        self.client = Geval.get_client(api_key)

        # outputs
        self.score = 0
        self.reason = ""

    # -------------------------------------------------------
    # Build prompt
    # -------------------------------------------------------
    def build_prompt(self, params: Dict[str, Any]) -> str:
        if self.criteria:
            rule_section = f"## Evaluation Criteria\n{self.criteria}\n\n"
        else:
            steps = "\n".join(f"- {s}" for s in self.evaluation_steps)
            rule_section = f"## Evaluation Steps\n{steps}\n\n"

        params_section = "## Evaluation Inputs\n"
        for key, value in params.items():
            params_section += f"- {key}: {value}\n"

        json_instruction = """
## Output Format
Return ONLY JSON:
{
  "score": <float 0-1>,
  "reason": "<short explanation>"
}
"""

        return rule_section + params_section + json_instruction

    # -------------------------------------------------------
    # Single metric evaluation
    # -------------------------------------------------------
    def measure(self, test_case: Dict[str, Any]):
        prompt = self.build_prompt(test_case)

        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()

        # Extract JSON safely
        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end == -1:
            self.score = 0
            self.reason = "Invalid JSON from judge"
            return

        json_str = raw[start:end + 1]

        try:
            parsed = json.loads(json_str)
        except Exception:
            self.score = 0
            self.reason = "Could not parse judge response JSON"
            return

        self.score = parsed.get("score", 0)
        self.reason = parsed.get("reason", "")

    # -------------------------------------------------------
    # For convenience (DeepEval style)
    # -------------------------------------------------------
    def evaluate(self, **params):
        self.measure(params)
        return {
            "score": self.score,
            "reason": self.reason,
            "pass": self.score >= self.threshold
        }

class Runner:
    """
    Runs multiple metrics against multiple test cases.
    DeepEval-style evaluation engine.
    """

    def __init__(self, test_cases=None, metrics=None):
        self.test_cases = test_cases or []
        self.metrics = metrics or []
        self.results = []

    def add_test_case(self, case):
        self.test_cases.append(case)

    def add_metric(self, metric):
        self.metrics.append(metric)

    def run(self):
        self.results = []

        for case in self.test_cases:
            metric_results = {}

            for metric in self.metrics:
                metric.measure(case)
                metric_results[metric.name] = {
                    "score": metric.score,
                    "reason": metric.reason,
                    "pass": metric.score >= metric.threshold
                }

            self.results.append({
                "input": case.get("input"),
                "actual_output": case.get("actual_output"),
                "expected_output": case.get("expected_output"),
                "metrics": metric_results
            })

        return self.results

    def summary(self):
        """Print a pretty summary like DeepEval."""

        for idx, result in enumerate(self.results, 1):
            print(f"Test Case {idx}: {result['input']}")
            for name, m in result["metrics"].items():
                status = "PASS" if m["pass"] else "FAIL"
                print(f"  - {name}: {m['score']} ({status})")

            print()
