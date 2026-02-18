from typing import Optional, Tuple, Any
from .test_case import RAGTestCase
import json,re


class MetricBase:
    def __init__(
        self,
        name: str,
        threshold: float = 0.5,
        strict_mode: bool = False,
        include_reason: bool = True,
        verbose: bool = False,
        evaluation_template: Any = None,
    ):
        self.name = name
        self.threshold = threshold
        self.strict_mode = strict_mode
        self.include_reason = include_reason
        self.verbose = verbose
        self.template = evaluation_template

        self.score: Optional[float] = None
        self.reason: Optional[str] = None

    def evaluate(self, test_case: RAGTestCase) -> Tuple[float, Optional[str]]:
        raise NotImplementedError

    def _apply_strict_mode(self, score: float) -> float:
        """
        If strict_mode is enabled, convert score to 0 or 1
        based on the metric's threshold instead of requiring 1.0.
        """
        if not self.strict_mode:
            return score
        return 1.0 if score >= self.threshold else 0.0



    def _parse_score_json(self, raw: str, default_reason: str):
        if not raw:
            return 0.0, default_reason

        text = raw.strip()
        text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
        try:
            data = json.loads(text)
            return float(data.get("score", 0.0)), data.get("reason", default_reason)
        except:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            json_part = match.group(0)
            try:
                data = json.loads(json_part)
                return float(data.get("score", 0.0)), data.get("reason", default_reason)
            except:
                pass

        cleaned = re.sub(r",\s*}", "}", text)
        cleaned = re.sub(r",\s*]", "]", cleaned)

        try:
            data = json.loads(cleaned)
            return float(data.get("score", 0.0)), data.get("reason", default_reason)
        except:
            pass

        return 0.0, default_reason


    def measure(self, test_case: RAGTestCase) -> float:
        raw_score, reason = self.evaluate(test_case)

        score = float(raw_score)
        score = max(0.0, min(1.0, score)) 
        score = self._apply_strict_mode(score)

        self.score = score
        self.reason = reason if self.include_reason else None

        if self.verbose:
            print(f"[{self.name}] score: {self.score}")
            if self.reason:
                print(f"[{self.name}] reason: {self.reason}")

        return self.score
