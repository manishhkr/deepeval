from .geval import Geval, Runner
from .mcp_server import MCPServer
from .mcp_tool_call import MCPToolCall
from .test_case import LLMTestCase
from .evaluation_runner import EvaluationRunner
from .mcp_use_metric import MCPUseMetric
from .rag_metrics import (
    evaluate_rag_output,
    evaluate_rag_output_custom_metrics,
)
from .test_case import RAGTestCase

# Import DeepEval metrics for direct use (users can import and customize)
try:
    # Try importing from main metrics module
    try:
        from deepeval.metrics import (  # type: ignore
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
        )
    except ImportError:
        # Fallback: try importing from ragas submodule (some DeepEval versions)
        from deepeval.metrics.ragas import (  # type: ignore
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
        )
    
    from deepeval.test_case import LLMTestCase as DeepEvalRAGTestCase  # type: ignore
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    # Create placeholder classes for backward compatibility
    AnswerRelevancyMetric = None  # type: ignore
    FaithfulnessMetric = None  # type: ignore
    ContextualPrecisionMetric = None  # type: ignore
    ContextualRecallMetric = None  # type: ignore
    ContextualRelevancyMetric = None  # type: ignore
    DeepEvalRAGTestCase = None  # type: ignore


__all__ = [
    # Main evaluation functions (general and customizable)
    "evaluate_rag_output",
    "evaluate_rag_output_custom_metrics",
    # Test case classes
    "RAGTestCase",
    "DeepEvalRAGTestCase",
    # DeepEval metrics (users can import and customize model, threshold, etc.)
    "AnswerRelevancyMetric",
    "FaithfulnessMetric",
    "ContextualPrecisionMetric",
    "ContextualRecallMetric",
    "ContextualRelevancyMetric",
    # Legacy/other components
    "Geval",
    "Runner",
    "MCPServer",
    "MCPToolCall",
    "LLMTestCase",
    "EvaluationRunner",
    "MCPUseMetric",
]
