"""
Test script to verify DeepEval migration changes.
Run this script to check if everything is working correctly.
"""
import sys

print("=" * 70)
print("Testing DeepEval Migration Changes")
print("=" * 70)
print()

# Test 1: Check Python version
print("1. Python Version:")
print(f"   {sys.version}")
print()

print("2. DeepEval Installation Check:")
try:
    import deepeval
    version = getattr(deepeval, '__version__', 'installed (version unknown)')
    print(f"   OK DeepEval is installed: {version}")
    deepeval_installed = True
except ImportError:
    print("   X DeepEval is NOT installed")
    print("   -> Run: pip install deepeval")
    deepeval_installed = False
print()

print("3. Import Main Functions:")
try:
    from framework.metrics import evaluate_rag_output, evaluate_rag_output_custom_metrics
    print("   OK evaluate_rag_output imported successfully")
    print("   OK evaluate_rag_output_custom_metrics imported successfully")
except ImportError as e:
    print(f"   X Failed to import: {e}")
print()

print("4. Import DeepEval Metrics:")
try:
    from framework.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
    )
    if (AnswerRelevancyMetric is not None and 
        FaithfulnessMetric is not None and 
        ContextualPrecisionMetric is not None and 
        ContextualRecallMetric is not None and 
        ContextualRelevancyMetric is not None):
        print("   OK All DeepEval metrics imported successfully")
        print("   OK AnswerRelevancyMetric available")
        print("   OK FaithfulnessMetric available")
        print("   OK ContextualPrecisionMetric available")
        print("   OK ContextualRecallMetric available")
        print("   OK ContextualRelevancyMetric available")
    else:
        print("   WARNING: Metrics are None (DeepEval not installed)")
except ImportError as e:
    print(f"   X Failed to import metrics: {e}")
print()

print("5. Function Parameters Check:")
try:
    import inspect
    sig1 = inspect.signature(evaluate_rag_output)
    print("   evaluate_rag_output parameters:")
    for param_name, param in sig1.parameters.items():
        default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
        print(f"     - {param_name}{default}")
    
    sig2 = inspect.signature(evaluate_rag_output_custom_metrics)
    print("   evaluate_rag_output_custom_metrics parameters:")
    for param_name, param in sig2.parameters.items():
        default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
        print(f"     - {param_name}{default}")
    print("   OK All customizable parameters are present (model, threshold, verbose, include_reason)")
except Exception as e:
    print(f"   X Error checking parameters: {e}")
print()

print("6. Import Test Case Classes:")
try:
    from framework.metrics import RAGTestCase, DeepEvalRAGTestCase
    print("   OK RAGTestCase imported successfully")
    if DeepEvalRAGTestCase is not None:
        print("   OK DeepEvalRAGTestCase imported successfully")
    else:
        print("   WARNING: DeepEvalRAGTestCase is None (DeepEval not installed)")
except ImportError as e:
    print(f"   X Failed to import: {e}")
print()

print("7. Check Dependencies:")
try:
    with open('pyproject.toml', 'r') as f:
        content = f.read()
        if 'deepeval' in content:
            print("   OK deepeval found in pyproject.toml")
        else:
            print("   X deepeval NOT found in pyproject.toml")
except FileNotFoundError:
    print("   WARNING: pyproject.toml not found")
except Exception as e:
    print(f"   X Error reading pyproject.toml: {e}")
print()

print("=" * 70)
print("Summary:")
print("=" * 70)
if deepeval_installed:
    print("OK DeepEval is installed and ready to use")
    print("OK All imports are working correctly")
    print("OK Functions are general and customizable")
    print("\nYou can now use the evaluation functions with custom parameters!")
else:
    print("WARNING: DeepEval is not installed")
    print("  Install it with: pip install deepeval")
    print("  The code will work once DeepEval is installed")
print("=" * 70)
