import time


def create_eval_object(client, instructions):
    eval_obj = client.evals.create(
        name="Hr leave",
        data_source_config={
            "type": "custom",
            "include_sample_schema": True,
            "item_schema": {
                "type": "object",
                "properties": {
                    "ticket_text": {"type": "string"},
                    "correct_label": {"type": "string"},
                },
                "required": ["ticket_text", "correct_label"]
            }
        },
        testing_criteria=[{
            "type": "string_check",
            "name": "Match output to human label",
            "input": "{{sample.output_text}}",
            "operation": "eq",
            "reference": "{{item.correct_label}}",
        }],
    )

    # print("\nEval object created successfully:", eval_obj.id)
    return eval_obj.id



def upload_dataset(client, file_path):
    file_obj = client.files.create(
        file=open(file_path, "rb"),
        purpose="evals"
    )
    # print("Dataset uploaded:", file_obj.id)
    return file_obj.id


def run_evaluation(client, eval_id, file_id, instructions, model="gpt-4.1"):
    run = client.evals.runs.create(
        eval_id=eval_id,
        name="Hr Assistant",
        data_source={
            "type": "responses",
            "model": model,
            "input_messages": {
                "type": "template",
                "template": [
                    {"role": "developer", "content": instructions},
                    {"role": "user", "content": "{{item.ticket_text}}"},
                ],
            },
            "source": {"type": "file_id", "id": file_id},
        },
    )
    # print("Eval run started:", run.id)
    return run.id


def retrieve_results(client, eval_id, run_id):
    # print("Waiting for eval to complete...")

    while True:
        run = client.evals.runs.retrieve(eval_id=eval_id, run_id=run_id)
        # print(f"Status: {run.status} | Passed={run.result_counts.passed}")

        if run.status in ["completed", "failed", "cancelled"]:
            break

        time.sleep(2)

    return {
        "status": run.status,
        "results": run.per_testing_criteria_results,
        "counts": run.result_counts
    }
