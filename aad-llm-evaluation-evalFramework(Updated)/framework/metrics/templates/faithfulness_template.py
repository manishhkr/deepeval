class FaithfulnessTemplate:

    @staticmethod
    def build_prompt(actual_output: str, retrieval_context: str) -> str:
        return f"""
You are an evaluator for a RAG system.

Definitions:
- The answer is FAITHFUL if all its factual statements are supported by the context
  and it does not contradict the context or introduce unsupported facts.
- The faithfulness score is in [0, 1] and equals:
    (# of supported / non-contradictory statements) / (total statements in the answer).

Context (retrieval_context):
{retrieval_context}

Answer (LLM output):
{actual_output}

Steps:
1. Break the answer into factual statements.
2. For each statement, decide: supported by context, unsupported, or contradicted.
3. Compute the faithfulness score as described.
4. Briefly explain your reasoning.

Return ONLY valid JSON:
{{
  "score": <float between 0 and 1>,
  "reason": "<short explanation>"
}}
"""

