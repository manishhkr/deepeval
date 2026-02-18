from . import __init__


class AnswerRelevancyTemplate:
    @staticmethod
    def build_prompt(input_text: str, actual_output: str) -> str:
        return f"""
You are an evaluator for a RAG system.

Task:
- Break the answer into statements.
- Determine which statements are relevant to the given input question.
- Compute the answer relevancy score = (number of relevant statements) / (total number of statements),
  with a value between 0 and 1.

Input (user question):
{input_text}

Answer (LLM output):
{actual_output}

Now respond ONLY in valid JSON with this shape:
{{
  "score": <float between 0 and 1>,
  "reason": "<short explanation of why you chose that score>"
}}
"""

