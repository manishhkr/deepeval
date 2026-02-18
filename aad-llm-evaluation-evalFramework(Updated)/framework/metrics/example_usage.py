"""
Example usage of the general RAG evaluation functions.

This file demonstrates how users can import and customize the evaluation metrics
by changing parameters like model, threshold, etc.
"""

from framework.metrics import (
    evaluate_rag_output,
    evaluate_rag_output_custom_metrics,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase as RAGTestCase  # type: ignore


def example_basic_usage():
    """Example 1: Basic usage with default parameters."""
    print("=" * 70)
    print("Example 1: Basic Usage (Default Parameters)")
    print("=" * 70)
    
    results = evaluate_rag_output(
        input_query="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        retrieval_context=[
            "France is a country in Europe.",
            "Paris is the largest city in France.",
        ],
        expected_output="Paris is the capital of France.",
    )
    
    print("\nResults:")
    for metric_name, metric_data in results.items():
        print(f"  {metric_name}: {metric_data['score']:.2f} (Pass: {metric_data['pass']})")
    print()


def example_custom_model_and_threshold():
    """Example 2: Customizing model and threshold."""
    print("=" * 70)
    print("Example 2: Custom Model and Threshold")
    print("=" * 70)
    
    # User can change the model and threshold
    results = evaluate_rag_output(
        input_query="What is machine learning?",
        actual_output="Machine learning is a subset of AI.",
        retrieval_context=["AI includes machine learning and deep learning."],
        model="gpt-4o",  # User changed the model
        threshold=0.7,    # User changed the threshold
        verbose=True,     # User enabled verbose output
    )
    
    print("\nResults:")
    for metric_name, metric_data in results.items():
        print(f"  {metric_name}: {metric_data['score']:.2f} (Pass: {metric_data['pass']})")
    print()


def example_selective_metrics():
    """Example 3: Using only specific metrics."""
    print("=" * 70)
    print("Example 3: Selective Metrics")
    print("=" * 70)
    
    # User can choose which metrics to use
    results = evaluate_rag_output_custom_metrics(
        input_query="Explain quantum computing",
        actual_output="Quantum computing uses quantum mechanics.",
        retrieval_context=["Quantum computing is a new computing paradigm."],
        metrics_to_use=["answer_relevancy", "faithfulness"],  # User selected only these
        model="gpt-4o-mini",
        threshold=0.6,
    )
    
    print("\nResults (only selected metrics):")
    for metric_name, metric_data in results.items():
        print(f"  {metric_name}: {metric_data['score']:.2f} (Pass: {metric_data['pass']})")
    print()


def example_direct_metric_usage():
    """Example 4: Using DeepEval metrics directly for maximum customization."""
    print("=" * 70)
    print("Example 4: Direct Metric Usage (Maximum Customization)")
    print("=" * 70)
    
    # Check if DeepEval metrics are available
    if AnswerRelevancyMetric is None or FaithfulnessMetric is None:
        print("\nError: DeepEval metrics are not available.")
        print("Please install DeepEval: pip install deepeval")
        return
    
    # Users can import and use DeepEval metrics directly
    # This gives them full control over all parameters
    
    # Create test case
    test_case = RAGTestCase(
        input="What is Python?",
        actual_output="Python is a programming language.",
        retrieval_context=["Python is a high-level programming language."],
        expected_output="Python is a programming language used for various applications.",
    )
    
    # User can customize each metric individually
    answer_relevancy = AnswerRelevancyMetric(
        model="gpt-4o",      # User's choice
        threshold=0.8,       # User's threshold
        include_reason=True, # User's preference
    )
    
    faithfulness = FaithfulnessMetric(
        model="gpt-4o-mini", # Different model for this metric
        threshold=0.6,        # Different threshold
        include_reason=True,
    )
    
    # Measure metrics
    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)
    
    print(f"\nAnswer Relevancy: {answer_relevancy.score:.2f} (Pass: {answer_relevancy.score >= answer_relevancy.threshold})")
    print(f"Faithfulness: {faithfulness.score:.2f} (Pass: {faithfulness.score >= faithfulness.threshold})")
    print()


def example_batch_evaluation():
    """Example 5: Batch evaluation with custom parameters."""
    print("=" * 70)
    print("Example 5: Batch Evaluation")
    print("=" * 70)
    
    test_cases = [
        {
            "input": "What is AI?",
            "actual_output": "AI is artificial intelligence.",
            "retrieval_context": ["AI stands for artificial intelligence."],
        },
        {
            "input": "What is ML?",
            "actual_output": "ML is machine learning.",
            "retrieval_context": ["ML stands for machine learning."],
        },
    ]
    
    # User can set parameters once and use for all evaluations
    model = "gpt-4o-mini"
    threshold = 0.5
    
    all_results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nEvaluating test case {i}...")
        results = evaluate_rag_output(
            input_query=test_case["input"],
            actual_output=test_case["actual_output"],
            retrieval_context=test_case["retrieval_context"],
            model=model,        # User's custom model
            threshold=threshold, # User's custom threshold
        )
        all_results.append(results)
    
    print("\nBatch Results Summary:")
    for i, results in enumerate(all_results, 1):
        print(f"\nTest Case {i}:")
        for metric_name, metric_data in results.items():
            print(f"  {metric_name}: {metric_data['score']:.2f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RAG Evaluation Examples - General and Customizable Functions")
    print("=" * 70 + "\n")
    
    try:
        example_basic_usage()
        example_custom_model_and_threshold()
        example_selective_metrics()
        example_direct_metric_usage()
        example_batch_evaluation()
        
        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install DeepEval: pip install deepeval")
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
