def test_model(client):
    instructions = (
        "You are an HR assistant. Categorize the leave request into one of:\n"
        "\"Sick Leave\", \"Casual Leave\", \"Other\".\n"
        "Respond with only one word."
    )

    ticket = "I have a family function tomorrow."

    response = client.responses.create(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "developer", "content": instructions},
            {"role": "user", "content": ticket}
        ]
    )

    # print("Model Output:", response.output_text)
    return instructions
