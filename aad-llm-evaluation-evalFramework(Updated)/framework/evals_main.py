from openai import OpenAI
from framework.framework_utils import get_or_create_eval_id
from framework.eval_ops import upload_dataset, run_evaluation, retrieve_results
from framework.model_test import test_model
from framework.cleanup import cleanup_all 


def evaluate_single(model, api_key=None, dataset="./test/ticket.jsonl"):
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    # Step 1: get instructions
    instructions = test_model(client)

    # Step 2: create or reuse eval object
    eval_id = get_or_create_eval_id(client, instructions, model_name=model)

    # Step 3: upload dataset
    file_id = upload_dataset(client, dataset)

    # Step 4: run evaluation
    run_id = run_evaluation(client, eval_id, file_id, instructions)

    # Step 5: get results
    results = retrieve_results(client, eval_id, run_id)

    return {
        "eval_id": eval_id,
        "run_id": run_id,
        "results": results
    }


def evaluate_models(models, dataset="./test/ticket.jsonl"):
    client = OpenAI()
    all_results = {}

    for model in models:
        print(f"\n--- Running evaluation for {model} ---")

        instructions = test_model(client)
        eval_id = get_or_create_eval_id(client, instructions, model_name=model)
        file_id = upload_dataset(client, dataset)
        run_id = run_evaluation(client, eval_id, file_id, instructions, model=model)

        all_results[model] = {
            "eval_id": eval_id,
            "run_id": run_id,
            "results": retrieve_results(client, eval_id, run_id)
        }

    return all_results


if __name__ == "__main__":
    print("\n=== Running single model evaluation ===")
    single_result = evaluate_single()

    # -------- Clean Single Evaluation Output --------
    print("\n=== Single Evaluation Result ===")
    print(f"Eval ID : {single_result['eval_id']}")
    print(f"Run ID  : {single_result['run_id']}")
    print(f"Status  : {single_result['results']['status']}")

    c = single_result["results"]["counts"]
    print(f"Passed  : {c.passed}")
    print(f"Failed  : {c.failed}")
    print(f"Total   : {c.total}")

    # -------- Multi Model Evaluation --------
    print("\n=== Running multiple model evaluations ===")
    models = ["gpt-4o-mini", "gpt-4.1", "gpt-4o"]
    multi_results = evaluate_models(models)

    # -------- Summary Table --------
    print("\n=== Summary Table ===")
    print("Model\tPassed\tFailed\tTotal\tAccuracy")

    for model, result in multi_results.items():
        rc = result["results"]["counts"]
        total = rc.total
        passed = rc.passed
        failed = rc.failed
        acc = round((passed / total) * 100, 1) if total > 0 else 0

        print(f"{model}\t{passed}\t{failed}\t{total}\t{acc}%")

    # -------- Cleanup --------
    print("\nStarting cleanup...\n")
    cleanup_all()
