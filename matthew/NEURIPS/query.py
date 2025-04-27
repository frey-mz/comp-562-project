import json
import sys
import time
sys.path.insert(0, '..')
import os
import models
import grader

def _extract_boxed_text(s):
    results = []
    i = 0
    while i < len(s):
        if s.startswith(r'\boxed{', i):
            i += len(r'\boxed{')
            brace_level = 1
            start = i
            while i < len(s) and brace_level > 0:
                if s[i] == '{':
                    brace_level += 1
                elif s[i] == '}':
                    brace_level -= 1
                i += 1
            if brace_level == 0:
                results.append(s[start:i-1])  # Exclude the final closing brace
        else:
            i += 1
    return results

offset = float(input("Please enter the start line: "))
end = float(input("Please enter problems to solve: "))
linecount = 0

input_file_path = 'math_dataset_clean.jsonl'   # Each line should be a JSON object
output_file_path = 'output.jsonl'  # Output as JSONL (one JSON per line)


TEST_MODELS = ["gemini", "gemma"]


def isEquiv(x, y):
    ans_box = _extract_boxed_text(x)
    sol_box = _extract_boxed_text(y)

    ans = ans_box[-1]
    sol = sol_box[-1]

    res = grader.grade_answer(ans, sol)

    if not res:
        print("FAIL, SOL, MODEL ANS")
        print(ans)
        print(sol)

    return res

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
            system_prompt = "You are a math problem solver. Please first solve any problem given to you step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\". \n."
            res = models.getMultiModelResponse(problem, system_prompt, lambda x: isEquiv(solution, x))
            if res == None:
                continue
            correct, costs = res

            output_data = {
                'id': data.get('id'),
                'problem': problem,
                #'solution': solution,
                #'answer': answer,
                'correct': correct,
                'costs': costs
            }

            print("solved output for id " + str(data.get('id')))

            outfile.write(json.dumps(output_data) + '\n')
            outfile.flush()

        linecount = linecount + 1
        time.sleep(4)

