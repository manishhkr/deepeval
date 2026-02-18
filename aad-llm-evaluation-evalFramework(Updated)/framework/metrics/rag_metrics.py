"""
General RAG evaluation functions using DeepEval metrics.
Users can import these functions and customize model, threshold, and other parameters.
"""
from typing import Dict, Any, Optional, List

try:
    # Try importing from main metrics module
    try:
        from deepeval.metrics import (  # type: ignore
            AnswerRelevancyMetric as DeepEvalAnswerRelevancyMetric,
            FaithfulnessMetric as DeepEvalFaithfulnessMetric,
            ContextualPrecisionMetric as DeepEvalContextualPrecisionMetric,
            ContextualRecallMetric as DeepEvalContextualRecallMetric,
            ContextualRelevancyMetric as DeepEvalContextualRelevancyMetric,
        )
    except ImportError:
        # Fallback: try importing from ragas submodule (some DeepEval versions)
        from deepeval.metrics.ragas import (  # type: ignore
            AnswerRelevancyMetric as DeepEvalAnswerRelevancyMetric,
            FaithfulnessMetric as DeepEvalFaithfulnessMetric,
            ContextualPrecisionMetric as DeepEvalContextualPrecisionMetric,
            ContextualRecallMetric as DeepEvalContextualRecallMetric,
            ContextualRelevancyMetric as DeepEvalContextualRelevancyMetric,
        )
    
    from deepeval.test_case import LLMTestCase as DeepEvalRAGTestCase  # type: ignore
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    raise ImportError(
        "DeepEval is not installed. Please install it using: pip install deepeval"
    )


def evaluate_rag_output(
    input_query: str,
    actual_output: str,
    retrieval_context: List[str],
    expected_output: Optional[str] = None,
    model: str = "gpt-4o-mini",
    threshold: float = 0.5,
    verbose: bool = False,
    include_reason: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate RAG output using DeepEval metrics.
    
    This function is general and configurable - users can customize:
    - model: The LLM model to use for evaluation (default: "gpt-4o-mini")
    - threshold: The threshold for pass/fail (default: 0.5)
    - verbose: Whether to print detailed information (default: False)
    - include_reason: Whether to include reasoning in results (default: True)
    
    Args:
        input_query: The input query/question
        actual_output: The actual output from the RAG system
        retrieval_context: List of retrieved context strings
        expected_output: Optional expected output for comparison
        model: LLM model to use for evaluation (customizable)
        threshold: Threshold for pass/fail determination (customizable)
        verbose: Whether to print verbose output (customizable)
        include_reason: Whether to include reasoning (customizable)
    
    Returns:
        Dictionary containing scores and reasons for each metric
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError("DeepEval is not available. Please install it first.")
    
    tc = DeepEvalRAGTestCase(
        input=input_query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    # Use synchronous mode for clearer, less noisy output (no long spinner)
    answer_relevancy = DeepEvalAnswerRelevancyMetric(
        model=model,
        threshold=threshold,
        include_reason=include_reason,
        async_mode=False,
    )
    
    faithfulness = DeepEvalFaithfulnessMetric(
        model=model,
        threshold=threshold,
        include_reason=include_reason,
        async_mode=False,
    )
    
    contextual_precision = DeepEvalContextualPrecisionMetric(
        model=model,
        threshold=threshold,
        include_reason=include_reason,
        async_mode=False,
    )
    
    contextual_recall = DeepEvalContextualRecallMetric(
        model=model,
        threshold=threshold,
        include_reason=include_reason,
        async_mode=False,
    )
    
    contextual_relevancy = DeepEvalContextualRelevancyMetric(
        model=model,
        threshold=threshold,
        include_reason=include_reason,
        async_mode=False,
    )

    metrics = {
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
        "contextual_precision": contextual_precision,
        "contextual_recall": contextual_recall,
        "contextual_relevancy": contextual_relevancy,
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, metric in metrics.items():
        try:
            # Let DeepEval show its own spinner / progress output
            metric.measure(tc)

            score = getattr(metric, "score", 0.0)
            reason = getattr(metric, "reason", "") if include_reason else None
            
            results[name] = {
                "score": float(score) if score is not None else 0.0,
                "reason": reason,
                "pass": float(score) >= threshold if score is not None else False,
            }
            
            if verbose:
                print(f"[{name}] score: {score}, pass: {results[name]['pass']}")
                if reason:
                    print(f"[{name}] reason: {reason}")
        except Exception as e:
            if verbose:
                print(f"[{name}] Error: {str(e)}")
            results[name] = {
                "score": 0.0,
                "reason": f"Error during evaluation: {str(e)}",
                "pass": False,
            }

    return results


def evaluate_rag_output_custom_metrics(
    input_query: str,
    actual_output: str,
    retrieval_context: List[str],
    expected_output: Optional[str] = None,
    metrics_to_use: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
    threshold: float = 0.5,
    verbose: bool = False,
    include_reason: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate RAG output using only specified DeepEval metrics.
    
    This function allows users to select which metrics to use, making it even more general.
    
    Args:
        input_query: The input query/question
        actual_output: The actual output from the RAG system
        retrieval_context: List of retrieved context strings
        expected_output: Optional expected output for comparison
        metrics_to_use: List of metric names to use. Options:
            - "answer_relevancy"
            - "faithfulness"
            - "contextual_precision"
            - "contextual_recall"
            - "contextual_relevancy"
            If None, all metrics are used.
        model: LLM model to use for evaluation (customizable)
        threshold: Threshold for pass/fail determination (customizable)
        verbose: Whether to print verbose output (customizable)
        include_reason: Whether to include reasoning (customizable)
    
    Returns:
        Dictionary containing scores and reasons for selected metrics
    """
    if not DEEPEVAL_AVAILABLE:
        raise ImportError("DeepEval is not available. Please install it first.")
    
    # Default to all metrics if none specified
    if metrics_to_use is None:
        metrics_to_use = [
            "answer_relevancy",
            "faithfulness",
            "contextual_precision",
            "contextual_recall",
            "contextual_relevancy",
        ]
    
    # Create DeepEval RAGTestCase
    tc = DeepEvalRAGTestCase(
        input=input_query,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
    )

    # Map metric names to DeepEval metric classes
    metric_classes = {
        "answer_relevancy": DeepEvalAnswerRelevancyMetric,
        "faithfulness": DeepEvalFaithfulnessMetric,
        "contextual_precision": DeepEvalContextualPrecisionMetric,
        "contextual_recall": DeepEvalContextualRecallMetric,
        "contextual_relevancy": DeepEvalContextualRelevancyMetric,
    }

    results: Dict[str, Dict[str, Any]] = {}

    for metric_name in metrics_to_use:
        if metric_name not in metric_classes:
            if verbose:
                print(f"Warning: Unknown metric '{metric_name}', skipping.")
            continue
        
        try:
            # Initialize metric with configurable parameters
            metric = metric_classes[metric_name](
                model=model,
                threshold=threshold,
                include_reason=include_reason,
                async_mode=False,
            )

            # Let DeepEval show its own spinner / progress output
            metric.measure(tc)

            score = getattr(metric, "score", 0.0)
            reason = getattr(metric, "reason", "") if include_reason else None
            
            results[metric_name] = {
                "score": float(score) if score is not None else 0.0,
                "reason": reason,
                "pass": float(score) >= threshold if score is not None else False,
            }
            
            if verbose:
                print(f"[{metric_name}] score: {score}, pass: {results[metric_name]['pass']}")
                if reason:
                    print(f"[{metric_name}] reason: {reason}")
        except Exception as e:
            if verbose:
                print(f"[{metric_name}] Error: {str(e)}")
            results[metric_name] = {
                "score": 0.0,
                "reason": f"Error during evaluation: {str(e)}",
                "pass": False,
            }

    return results
