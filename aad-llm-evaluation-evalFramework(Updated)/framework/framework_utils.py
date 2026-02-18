import os, json, hashlib


def generate_task_key(instructions: str, model: str):
    return hashlib.sha256((instructions + model).encode()).hexdigest()


def load_eval_ids():
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(root_path, "eval_ids.json")
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return {}


def save_eval_ids(data):
    # Absolute path of the *project root*, based on this file's directory
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    file_path = os.path.join(root_path, "eval_ids.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def get_or_create_eval_id(client, instructions, model_name):
    from framework.eval_ops import create_eval_object  # lazy import to avoid circular dependencies

    key = generate_task_key(instructions, model_name)
    data = load_eval_ids()

    if key in data:
        # print(f"Reusing eval ID: {data[key]}")
        return data[key]

    eval_id = create_eval_object(client, instructions)
    data[key] = eval_id
    save_eval_ids(data)
    # print(f"New eval ID saved: {eval_id}")
    
    return eval_id
