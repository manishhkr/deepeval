from openai import OpenAI
import json
from typing import Optional


class MCPUseMetric:
    """  """

    @staticmethod
    def get_client(api_key: Optional[str] = None):
        """Return an OpenAI client with or without API key."""
        if api_key:
            return OpenAI(api_key=api_key)
        return OpenAI()

    def __init__(self, name="MCP-Use", judge_model="gpt-4o-mini",
                 threshold=0.5, api_key=None):
        self.name = name
        self.judge_model = judge_model
        self.threshold = threshold
        self.client = MCPUseMetric.get_client(api_key)

    def build_prompt(self, case):
        return f"""
You are an evaluator of MCP tool usage.

### User Input
{case["input"]}

### Agent Final Output
{case["actual_output"]}

### MCP Servers Available
{json.dumps(case["mcp_servers"], indent=2)}

### Tool Calls Made
{json.dumps(case["mcp_tools_called"], indent=2)}

### MCP Resources Used
{json.dumps(case["mcp_resources_called"], indent=2)}

### MCP Prompts Used
{json.dumps(case["mcp_prompts_called"], indent=2)}

Evaluate the agent on:
1. Correct tool selection
2. Correct arguments
3. Correct use of the tool result
4. Avoiding unnecessary tool calls

### Output JSON ONLY:
{{
  "score": <float 0-1>,
  "reason": "<short explanation>"
}}
"""

    def evaluate(self, **case):
        prompt = self.build_prompt(case)

        res = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        text = res.choices[0].message.content.strip()

        # Extract JSON only
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {"score": 0, "reason": "Invalid JSON", "pass": False}

        try:
            data = json.loads(text[start:end + 1])
        except:
            return {"score": 0, "reason": "JSON parse error", "pass": False}

        score = data.get("score", 0)

        return {
            "score": score,
            "reason": data.get("reason", ""),
            "pass": score >= self.threshold
        }
