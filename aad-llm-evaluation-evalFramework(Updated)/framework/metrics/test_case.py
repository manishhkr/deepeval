from dataclasses import dataclass
from typing import List, Optional

# Try to import DeepEval's RAGTestCase for compatibility
try:
    from deepeval.test_case import LLMTestCase as DeepEvalRAGTestCase  # type: ignore
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    DeepEvalRAGTestCase = None  # type: ignore


class LLMTestCase:
    def __init__(self, input, actual_output, expected_output=None,
                 mcp_servers=None, mcp_tools_called=None,
                 mcp_resources_called=None, mcp_prompts_called=None,
                 history=None):
        self.input = input
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.mcp_servers = mcp_servers or []
        self.mcp_tools_called = mcp_tools_called or []
        self.mcp_resources_called = mcp_resources_called or []
        self.mcp_prompts_called = mcp_prompts_called or []
        self.history = history or []

    def to_dict(self):
        return {
            "input": self.input,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "mcp_servers": [s.to_dict() for s in self.mcp_servers],
            "mcp_tools_called": [t.to_dict() for t in self.mcp_tools_called],
            "mcp_resources_called": self.mcp_resources_called,
            "mcp_prompts_called": self.mcp_prompts_called,
            "history": self.history,
        }


@dataclass
class RAGTestCase:
    """
    Legacy RAGTestCase for backward compatibility.
    For new code, use DeepEval's RAGTestCase from deepeval.test_case.
    """
    input: str
    actual_output: str
    retrieval_context: List[str]
    expected_output: Optional[str] = None
    
    def to_deepeval(self):
        """Convert to DeepEval's RAGTestCase if available."""
        if DEEPEVAL_AVAILABLE and DeepEvalRAGTestCase:
            return DeepEvalRAGTestCase(
                input=self.input,
                actual_output=self.actual_output,
                retrieval_context=self.retrieval_context,
                expected_output=self.expected_output,
            )
        else:
            raise ImportError("DeepEval is not available. Please install it first.")