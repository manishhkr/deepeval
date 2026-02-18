class ContextualRelevancyTemplate:
    @staticmethod
    def build_prompt(
        input_text: str,
        retrieval_context: str
    ) -> str:
        return f"""
You are evaluating how relevant retrieved context is to a query.

Definitions:
- Contextual relevancy measures how well the content of retrieval_context
  matches the information needs of the input question.
- High score if most chunks are useful and on-topic.
- Low score if many chunks are off-topic or noisy.

Input (user query):
{input_text}

Retrieved context:
{retrieval_context}

Steps:
1. Identify the key information needs expressed in the input.
2. Judge how well the context as a whole satisfies those needs.
3. Produce a score in [0, 1], where:
   - 1.0 means the context is highly relevant and focused.
   - 0.0 means the context is mostly irrelevant.
4. Provide a short explanation.

Return ONLY valid JSON:
{{
  "score": <float between 0 and 1>,
  "reason": "<short explanation>"
}}
"""
