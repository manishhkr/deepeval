from framework.rag_eval import evaluate_rag_metrics_with_upload
import json

def test_rag_metrics_only(model="gpt-4o-mini"):
    """Test only RAG metrics evaluation for a single model."""
    print("\n" + "="*70)
    print(f"Testing RAG Metrics") # Only for Model: {model}")
    print("="*70 + "\n")
    
    try:
        rag_eval_result = evaluate_rag_metrics_with_upload(
            jsonl_file_path="test/rag_testcases.jsonl",
            model=model,
            verbose=False
        )
        
        # print("\n" + "="*70)
        # print(f"RAG Metrics Evaluation Results") # ({model})")
        # print("="*70 + "\n")
        # print(json.dumps(rag_eval_result["results"], indent=2))
        # print("\n" + "="*70 + "\n")
        
        for i, result in enumerate(rag_eval_result["results"], 1):
            metrics = result.get("metrics", {})
            print(f"Test Case {i}:")
            print(f"  Input: {result.get('input', 'N/A')}")
            print(f"  Answer Relevancy:      {metrics.get('answer_relevancy', {}).get('score', 0.0):.2f}")
            print(f"  Faithfulness:          {metrics.get('faithfulness', {}).get('score', 0.0):.2f}")
            print(f"  Contextual Precision:  {metrics.get('contextual_precision', {}).get('score', 0.0):.2f}")
            print(f"  Contextual Recall:     {metrics.get('contextual_recall', {}).get('score', 0.0):.2f}")
            print(f"  Contextual Relevancy:  {metrics.get('contextual_relevancy', {}).get('score', 0.0):.2f}")
            print()
            
            # Basic validity assertions
            assert metrics.get("answer_relevancy", {}).get("score", 0.0) >= 0.0
            assert metrics.get("faithfulness", {}).get("score", 0.0) >= 0.0
            assert metrics.get("contextual_precision", {}).get("score", 0.0) >= 0.0
            assert metrics.get("contextual_recall", {}).get("score", 0.0) >= 0.0
            assert metrics.get("contextual_relevancy", {}).get("score", 0.0) >= 0.0
        
        print(f"All RAG metrics tests passed") # for model") #: {model}")
        return rag_eval_result["results"]
        
    except Exception as e:
        print(f"\n Error ({model}): {e}")
        import traceback
        traceback.print_exc()
        return None


def test_rag_metrics_multi(models):
    """
    Evaluate RAG metrics for multiple models.
    Returns: { model: [results per test case], ... }
    """
    print("\n" + "="*70)
    print("Running MULTI-MODEL RAG Metrics Evaluation")
    print("="*70 + "\n")

    all_results = {}

    for model in models:
        print("\n--- Evaluating Model:", model)
        results = test_rag_metrics_only(model=model)
        all_results[model] = results

    return all_results


def summarize_rag_results(all_results):
    print("\n" + "="*70)
    print("RAG METRICS SUMMARY TABLE")
    print("="*70)
    print("Model\tRel\tFaith\tCPrec\tCRec\tCRel")

    for model, cases in all_results.items():
        if not cases:
            continue

        def avg(metric):
            return sum(tc["metrics"][metric]["score"] for tc in cases) / len(cases)

        print(
            f"{model}\t"
            f"{avg('answer_relevancy'):.2f}\t"
            f"{avg('faithfulness'):.2f}\t"
            f"{avg('contextual_precision'):.2f}\t"
            f"{avg('contextual_recall'):.2f}\t"
            f"{avg('contextual_relevancy'):.2f}"
        )

    print("="*70 + "\n")




if __name__ == "__main__":
    test_rag_metrics_only("gpt-5.1")

    # models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-5.1"]
    # all_results = test_rag_metrics_multi(models)

    # summarize_rag_results(all_results)
