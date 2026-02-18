class ContextualRecallTemplate:
    @staticmethod
    def build_prompt(
        input_text: str,
        expected_output: str,
        retrieval_context: str
    ) -> str:
        return f"""
You are evaluating the recall of a RAG retriever.

Definitions:
- The user query is "input".
- The "expected_output" is the target/ideal answer.
- 'retrieval_context' is the set of retrieved chunks.
- Contextual recall measures whether the context contains enough relevant information
  so that an LLM could produce an answer similar to expected_output.

Input (user query):
{input_text}

Expected output (ideal answer):
{expected_output}

Retrieved context:
{retrieval_context}

Steps:
1. Determine which information in expected_output is essential.
2. Check whether those essential pieces appear or are strongly implied in retrieval_context.
3. Compute recall score in [0, 1] as:
   (# of essential pieces present in context) / (total essential pieces in expected_output).
4. Provide a short explanation.

Return ONLY valid JSON:
{{
  "score": <float between 0 and 1>,
  "reason": "<short explanation>"
}}
"""
