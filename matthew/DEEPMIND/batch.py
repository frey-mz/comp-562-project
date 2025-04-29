import os, json

from openai import OpenAI


def problem_gen(directory):
    # Open all files in the directory
    files = []
    for filename in sorted(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        print(filepath)
        if os.path.isfile(filepath):
            files.append(open(filepath, 'r'))
    
    try:
        while files:
            next_files = []
            for f in files:
                lines = []
                try:
                    for _ in range(2):
                        line = next(f)
                        lines.append(line.rstrip('\n'))
                except StopIteration:
                    f.close()
                    continue

                if lines:
                    yield lines
                    next_files.append(f)

            files = next_files
    finally:
        for f in files:
            f.close()

system_prompt = (
             "You are a math problem solver."
             "Please first solve any problem given to you step by step"
             "then put your final answer in one \"\\boxed{}\""
             "if the answer asks for a list of numbers, put a comma between each number."
             "if the answer asks for a True/False response, write either 'True' or 'False'"
        )

with open("batch.jsonl", 'w', encoding='utf-8') as outfile:
    for idx, two_lines in enumerate(problem_gen("train-hard")):
        if idx <= 5000:
            continue
        [problem, _] = two_lines
        res = {
            "custom_id": "request-" + str(idx),
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": 'gpt-4.1-mini',
                "input": problem,
                "instructions": system_prompt
            }
        }

        outfile.write(json.dumps(res) + '\n')

        if idx == 15000:
            break

    outfile.flush()
    
    outfile.close()

    client = OpenAI()

    batch_input_file = client.files.create(
        file=open("batch.jsonl", "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id
    res = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "description": "DEEPMIND TEST"
        }
    )
    print(res)

