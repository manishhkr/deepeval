# Framework for LLM Evaluation

[Backbone](https://platform.openai.com/docs/guides/evaluation-getting-started?api-mode=responses)

### Instructions for devs


- Any logic related to framework must be kept in `framework` directory
- And the code to test your logic must be put in `test/evals.py` import the function or class and create a function to test it
- Creating a pull request for a branch to be merged in main will automatically run workflow file
- Always create a new branch and pull request so that the main branch is always working

### Steps to run it locally (If api key is available)

- Install `uv` if not available (or pip for which I will not be providing information)
- git clone repo
- `uv sync`
- Install framework as a package
- from the project root folder run

```bash
uv pip install -e .
```

- run program using uv run
