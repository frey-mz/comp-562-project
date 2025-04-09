import json
from transformation import get_basic_answer

offset = float(input("Please enter the start line: "))
end = float(input("Please enter problems to solve: "))
linecount = 0

input_file_path = 'math_dataset_clean.jsonl'   # Each line should be a JSON object
output_file_path = 'output.jsonl'  # Output as JSONL (one JSON per line)

with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:

    for line in infile:
        if linecount == end:
            print("!ending!")
            break

        if offset > 0:
            offset = offset - 1
            continue

        if line.strip():  # Skip empty lines
            data = json.loads(line)
            problem = data.get('problem')
            solution = data.get('solution')
            print("new output for id " + str(data.get('id')))
            answer = ""
            try:
                answer = get_basic_answer(problem)
            except:
                print("error for line " + str(data.get('id')))
                linecount = linecount + 1
                continue

            output_data = {
                'id': data.get('id'),
                'problem': problem,
                'solution': solution,
                'answer': answer
            }

            print("solved output for id " + str(data.get('id')))

            outfile.write(json.dumps(output_data) + '\n')

        linecount = linecount + 1