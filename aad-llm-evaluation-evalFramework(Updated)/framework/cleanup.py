from openai import OpenAI
from framework.framework_utils import load_eval_ids


def clear_local_eval_json():
    """
    Remove local eval_ids.json content.
    """
    path = "./eval_ids.json"
    try:
        with open(path, "w") as f:
            f.write("{}")
        print("Cleared local eval_ids.json")
    except Exception as e:
        print("Could not clear eval_ids.json:", e)


def delete_eval_from_openai(client, eval_id: str):
    """
    Delete an eval object from OpenAI's server.
    """
    try:
        resp = client.evals.delete(eval_id)
        return resp
    except Exception as e:
        print(f"Could not delete eval {eval_id} from OpenAI: {e}")
        return None


def cleanup_all():
    """
    Deletes ALL evals stored in eval_ids.json from:
      1. OpenAI server
      2. Local eval_ids.json file

    Prints the full cleanup details.
    """
    print("\n=== Cleanup Started ===")

    # Load local eval IDs
    data = load_eval_ids()

    # 1. Show local JSON eval IDs
    if not data:
        print("No eval IDs stored locally.\n")
    else:
        print("\nEval IDs stored in JSON:")
        for key, eval_id in data.items():
            print(f" - {eval_id}")

    # 2. Show OpenAI eval IDs (same list)
    if data:
        print("\nEval IDs on OpenAI server to delete:")
        for key, eval_id in data.items():
            print(f" - {eval_id}")

        print("\nDeleting eval IDs from OpenAI server...")
        client = OpenAI()

        for key, eval_id in data.items():
            print(f" - Deleting {eval_id}...")
            resp = delete_eval_from_openai(client, eval_id)
            if resp:
                print(f"   Deleted: {eval_id}")
            else:
                print(f"   Failed/Not found: {eval_id}")

    # 3. Clear local JSON file
    print("\nClearing local eval_ids.json...")
    clear_local_eval_json()

    print("\nCleanup completed.\n")


if __name__ == "__main__":
    cleanup_all()
