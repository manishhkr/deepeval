class ContextualPrecisionTemplate:
    @staticmethod
    def build_prompt(
        input_text: str,
        expected_output: str,
        retrieval_context: str
    ) -> str:
        return f"""
You are evaluating the precision of a RAG retriever with respect to ranking.

Definitions:
- The user query is called "input".
- The "expected_output" is the ideal answer the system should produce.
- 'retrieval_context' is a ranked list of text chunks (from most relevant at the top
  to least relevant at the bottom).
- Contextual precision measures whether the most relevant chunks (for producing
  the expected_output) are ranked near the top of the list.

Input (user query):
{input_text}

Expected output (ideal answer):
{expected_output}

Ranked retrieval_context (from index 0 = highest rank):
{retrieval_context}

Steps:
1. Identify which chunks in retrieval_context are truly relevant
   for producing the expected_output.
2. Assess how well the relevant chunks are ranked compared to irrelevant ones.
3. Output a precision-like score between 0 and 1, where:
   - 1.0 means all relevant chunks are ranked above all irrelevant ones.
   - 0.0 means ranking is very poor.
4. Provide a short explanation.

Return ONLY valid JSON:
{{
  "score": <float between 0 and 1>,
  "reason": "<short explanation>"
}}
"""
