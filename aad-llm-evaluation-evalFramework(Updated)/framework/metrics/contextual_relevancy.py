from openai import OpenAI
from .metric_base import MetricBase
from .test_case import RAGTestCase
from .templates.contextual_relevancy_template import ContextualRelevancyTemplate


class ContextualRelevancyMetric(MetricBase):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        threshold: float = 0.5,
        strict_mode: bool = False,
        include_reason: bool = True,
        verbose: bool = False,
        evaluation_template=ContextualRelevancyTemplate,
    ):
        super().__init__(
            name="contextual_relevancy",
            threshold=threshold,
            strict_mode=strict_mode,
            include_reason=include_reason,
            verbose=verbose,
            evaluation_template=evaluation_template,
        )
        self.client = OpenAI()
        self.model = model

    def evaluate(self, test_case: RAGTestCase):
        ctx_text = "\n\n".join(test_case.retrieval_context or [])
        prompt = self.template.build_prompt(
            input_text=test_case.input,
            retrieval_context=ctx_text,
        )

        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        raw = resp.output_text.strip()

        score, reason = self._parse_score_json(
            raw,
            default_reason="Failed to parse contextual relevancy JSON.",
        )
        return score, reason
