import os
from typing import Optional
from openai import OpenAI,OpenAIError
import time

class ModelError(Exception):
    pass

#Checks for Open API key availability and gives access to OpenAI
def get_client(api_key: Optional[str] = None):
    try:
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            client = OpenAI()
        return client
    except OpenAIError as e:
        raise ModelError(str(e))

#Defining system prompts
def test_model(client):
    instructions = """You are an HR assistant. Categorize the following leave request into one of:
    "Sick Leave", "Casual Leave", "Other".
    Respond with only one of those words."""

    ticket = "I have a family function tomorrow."

    response = client.responses.create(
        model="gpt-4o-2024-08-06",
        #model="gpt-5.1",
        input=[
            {"role": "developer", "content": instructions},
            {"role": "user", "content": ticket}
        ]
    )

    # print("\nStep 1: Model Output")
    # print(response.output_text)

    return instructions

#Creating evaluation function
def create_eval_object(client, instructions):
    eval_obj = client.evals.create(
        name="Hr leave",
        data_source_config={
            "type": "custom",
            "item_schema": {
                "type": "object",
                "properties": {
                    "ticket_text": {"type": "string"},
                    "correct_label": {"type": "string"},
                },
                "required": ["ticket_text", "correct_label"]
            },
            "include_sample_schema": True,
        },
        testing_criteria=[
            {
                "type": "string_check",
                "name": "Match output to human label",
                "input": "{{sample.output_text}}",
                "operation": "eq",
                "reference": "{{item.correct_label}}",
            }
        ],
    )

    # print("\nStep 2: Eval Created")
    # print(eval_obj)
    return eval_obj.id

#Uploading sample data set
def upload_dataset(client, file_path: str):
    file_obj = client.files.create(
        file=open(file_path, "rb"),
        purpose="evals"
    )

    # print("\nStep 3: File Uploaded")
    # print(file_obj)
    return file_obj.id

#Executing the evaluation
def run_evaluation(client, eval_id, file_id, instructions):
    run = client.evals.runs.create(
        eval_id=eval_id,
        name="Hr Assistant",
        data_source={
            "type": "responses",
            "model": "gpt-4.1",
            #"model": "gpt-5.1",
            "input_messages": {
                "type": "template",
                "template": [
                    {"role": "developer", "content": instructions},
                    {"role": "user", "content": "{{item.ticket_text}}"},
                ],
            },
            "source": {"type": "file_id", "id": file_id}
        },
    )

    # print("\nStep 4: Eval Run Started")
    # print(run)
    # print("Run id:", run.id)

    return run.id


import time
#Returing results from Open AI server
def retrieve_results(client, eval_id: str, run_id: str):
    # print("\nStep 5: Waiting for Eval to Complete...")

    while True:
        run = client.evals.runs.retrieve(
            eval_id=eval_id,
            run_id=run_id
        )

        # print(f"Status: {run.status} | Passed={run.result_counts.passed} | Failed={run.result_counts.failed}")

        # Stop when done
        if run.status in ["completed", "failed", "cancelled"]:
            break

        time.sleep(2)

    # print("\nStep 6: Final Results")
    results = {
        "testing": run.per_testing_criteria_results,
        "results_count": run.result_counts,
        "data_source": run.data_source,
        "status": run.status
    }

    # print(results)
    return results


#Primary function for triggering runs
def evaluate_1(api_key: Optional[str] = None, dataset_path: str = "./test/ticket.jsonl"):
    try:

        # print(os.getcwd())
        client = get_client(api_key)
        instructions = test_model(client)
        eval_id = create_eval_object(client, instructions)
        file_id = upload_dataset(client, dataset_path)
        run_id = run_evaluation(client, eval_id, file_id, instructions)
        return eval_id, run_id

    except ModelError as e:
        print("Model Error:", str(e))
        return None, None

    except Exception as e:
        print("Unexpected Error:", str(e))
        return None, None
