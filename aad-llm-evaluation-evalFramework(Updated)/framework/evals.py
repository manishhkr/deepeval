from typing import Optional
from openai import OpenAI, OpenAIError


class ModelError(Exception):
    pass


def evaluate(api_key: Optional[str] = None):

    try:

        if api_key:

            client = OpenAI(api_key)

        else:

            client = OpenAI()

    except OpenAIError as e:

        text = str(e)
        raise ModelError(text)

    instructions = """
    You are an expert in categorizing IT support tickets. Given the support
    ticket below, categorize the request into one of "Hardware", "Software",
    or "Other". Respond with only one of those words.
    """

    ticket = "My monitor won't turn on - help!"

    response = client.responses.create(
        model="gpt-4.1-nano-2025-04-14",
        input=[
            {"role": "developer", "content": instructions},
            {"role": "user", "content": ticket},
        ],
    )

    # print(response.output_text)
