from openai import OpenAI
from .metric_base import MetricBase
from .test_case import RAGTestCase
from .templates.contextual_recall_template import ContextualRecallTemplate


class ContextualRecallMetric(MetricBase):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        threshold: float = 0.5,
        strict_mode: bool = False,
        include_reason: bool = True,
        verbose: bool = False,
        evaluation_template=ContextualRecallTemplate,
    ):
        super().__init__(
            name="contextual_recall",
            threshold=threshold,
            strict_mode=strict_mode,
            include_reason=include_reason,
            verbose=verbose,
            evaluation_template=evaluation_template,
        )
        self.client = OpenAI()
        self.model = model

    def evaluate(self, test_case: RAGTestCase):
        if not test_case.expected_output:
            return (
                0.0,
                "expected_output was not provided; cannot compute contextual recall properly.",
            )

        ctx_text = "\n\n".join(test_case.retrieval_context or [])
        prompt = self.template.build_prompt(
            input_text=test_case.input,
            expected_output=test_case.expected_output,
            retrieval_context=ctx_text,
        )

        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        raw = resp.output_text.strip()

        score, reason = self._parse_score_json(
            raw,
            default_reason="Failed to parse contextual recall JSON.",
        )
        return score, reason
