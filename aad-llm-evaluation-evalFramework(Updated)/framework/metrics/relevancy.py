import json 
from openai import OpenAI
from .metric_base import MetricBase
from .test_case import RAGTestCase
from .templates.answer_relevancy_template import AnswerRelevancyTemplate


class AnswerRelevancyMetric(MetricBase):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        threshold: float = 0.5,
        strict_mode: bool = False,
        include_reason: bool = True,
        verbose: bool = False,
        evaluation_template=AnswerRelevancyTemplate,
    ):
        super().__init__(
            name="answer_relevancy",
            threshold=threshold,
            strict_mode=strict_mode,
            include_reason=include_reason,
            verbose=verbose,
            evaluation_template=evaluation_template,
        )
        self.client = OpenAI()
        self.model = model

    def evaluate(self, test_case: RAGTestCase):
        prompt = self.template.build_prompt(
            input_text=test_case.input,
            actual_output=test_case.actual_output,
        )

        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        raw = resp.output_text.strip()
        score, reason = self._parse_score_json(
            raw,
            default_reason="Failed to parse answer relevancy JSON.",
        )
        return score, reason
