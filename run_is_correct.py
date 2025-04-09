import json
from is_correct import is_equiv


input_file_path = 'output_cleaned.jsonl'   # Input file path (JSON Lines format)
output_file_path = 'validated.jsonl' # Output file path

with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        obj = json.loads(line)
        answer = obj.get("answer")
        solution = obj.get("solution")
        obj["correct"] = is_equiv(answer, solution)
        outfile.write(json.dumps(obj) + '\n')