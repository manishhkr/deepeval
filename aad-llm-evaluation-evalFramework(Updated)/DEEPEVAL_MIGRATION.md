# DeepEval Migration Summary

## What Was Done

This document explains the changes made to replace existing evaluation metrics with DeepEval metrics and make them general/customizable.

## Priority 1: Converted to DeepEval 

### Changes Made:

1. **Replaced Custom Metrics with DeepEval Metrics**
   - Replaced custom `AnswerRelevancyMetric`, `FaithfulnessMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, and `ContextualRelevancyMetric` with DeepEval's built-in metrics
   - All metrics now use DeepEval's framework under the hood

2. **Updated `framework/metrics/rag_metrics.py`**
   - Now imports and uses DeepEval's metrics directly
   - Functions are backward compatible but use DeepEval internally

3. **Updated Dependencies**
   - Added `deepeval>=2.0.0` to `pyproject.toml`

## Priority 2: Made It General and Customizable 

### New General Functions:

1. **`evaluate_rag_output()`** - Main evaluation function
   - **Customizable parameters:**
     - `model`: LLM model to use (default: "gpt-4o-mini")
     - `threshold`: Pass/fail threshold (default: 0.5)
     - `verbose`: Print detailed output (default: False)
     - `include_reason`: Include reasoning in results (default: True)

2. **`evaluate_rag_output_custom_metrics()`** - Selective metrics
   - Allows users to choose which metrics to evaluate
   - Same customizable parameters as above
   - `metrics_to_use`: List of metric names to use

3. **Direct Metric Access**
   - Users can import DeepEval metrics directly from `framework.metrics`
   - Full control over all parameters for each metric individually

### Updated Files:

- `framework/metrics/rag_metrics.py` - Now uses DeepEval with configurable parameters
- `framework/metrics/__init__.py` - Exports DeepEval metrics and new functions
- `framework/metrics/test_case.py` - Added compatibility with DeepEval's RAGTestCase
- `framework/rag_eval.py` - Updated to support threshold and other customizable parameters
- `pyproject.toml` - Added deepeval dependency

## How Users Can Use It

### Example 1: Basic Usage (Default Parameters)
```python
from framework.metrics import evaluate_rag_output

results = evaluate_rag_output(
    input_query="What is AI?",
    actual_output="AI is artificial intelligence.",
    retrieval_context=["AI stands for artificial intelligence."],
)
```

### Example 2: Custom Model and Threshold
```python
from framework.metrics import evaluate_rag_output

results = evaluate_rag_output(
    input_query="What is AI?",
    actual_output="AI is artificial intelligence.",
    retrieval_context=["AI stands for artificial intelligence."],
    model="gpt-4o",      # User changed the model
    threshold=0.7,       # User changed the threshold
    verbose=True,        # User enabled verbose output
)
```

### Example 3: Selective Metrics
```python
from framework.metrics import evaluate_rag_output_custom_metrics

results = evaluate_rag_output_custom_metrics(
    input_query="What is AI?",
    actual_output="AI is artificial intelligence.",
    retrieval_context=["AI stands for artificial intelligence."],
    metrics_to_use=["answer_relevancy", "faithfulness"],  # Only these metrics
    model="gpt-4o-mini",
    threshold=0.6,
)
```

### Example 4: Direct Metric Usage (Maximum Customization)
```python
from framework.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import RAGTestCase

# Create test case
test_case = RAGTestCase(
    input="What is Python?",
    actual_output="Python is a programming language.",
    retrieval_context=["Python is a high-level programming language."],
)

# Customize each metric individually
answer_relevancy = AnswerRelevancyMetric(
    model="gpt-4o",      # User's choice
    threshold=0.8,       # User's threshold
    include_reason=True,
)

faithfulness = FaithfulnessMetric(
    model="gpt-4o-mini", # Different model for this metric
    threshold=0.6,       # Different threshold
)

# Measure
answer_relevancy.measure(test_case)
faithfulness.measure(test_case)
```

## Key Benefits

1. **Uses DeepEval Framework**: All metrics now use DeepEval's research-backed evaluation framework
2. **Fully Customizable**: Users can change model, threshold, and other parameters without modifying core code
3. **Backward Compatible**: Existing code should continue to work
4. **General and Reusable**: Functions can be imported and used with any configuration
5. **Selective Metrics**: Users can choose which metrics to evaluate

## Installation

Make sure DeepEval is installed:
```bash
pip install deepeval
```

Or if using the project's dependency management:
```bash
# The dependency is already added to pyproject.toml
# Just install project dependencies
```

## Files Changed

1. `pyproject.toml` - Added deepeval dependency
2. `framework/metrics/rag_metrics.py` - Complete rewrite using DeepEval
3. `framework/metrics/__init__.py` - Updated exports
4. `framework/metrics/test_case.py` - Added DeepEval compatibility
5. `framework/rag_eval.py` - Added threshold and other parameters
6. `framework/metrics/example_usage.py` - Created example file (NEW)

## Notes

- The old custom metric classes (`AnswerRelevancyMetric`, etc.) are no longer used
- All metrics now use DeepEval's implementation
- Users can import DeepEval metrics directly from `framework.metrics`
- All functions are general and accept customizable parameters
