import json
import sys
import time
sys.path.insert(0, '..')
import os
import models
import grader


def load_model(t): #helper models
    # Load the API key from environment or use the provided key
    system_prompt = "You are a math problem solver. Please first solve any problem given to you step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\". \n"
    key = os.getenv("GEMINI_API_KEY")
    if t == "gemini":
        model = models.Gemini_Model(api_key = key, system_prompt = system_prompt)
    elif t=="gemma":
        model = models.Gemma_Model(system_prompt = system_prompt)
    else:
        model = models.DeepSeek_Model(system_prompt = system_prompt)

    return model


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
            model_succ = {}
            for TEST_MODEL in TEST_MODELS:
                answer = ""
                model = load_model(TEST_MODEL)
                for attempt_count in range(5):
                    try:
                        answer = model.get_response(problem)
                        break
                    except Exception as e:
                        print(e)
                        print("error for line " + str(data.get('id')))
                    time.sleep(30)
                if answer == "":
                    print("WTF")
                print(solution)
                ans_box = _extract_boxed_text(answer)
                sol_box = _extract_boxed_text(solution)

                ans = ans_box[-1]
                sol = sol_box[-1]

                model_succ[TEST_MODEL] = grader.grade_answer(ans, sol)

                print("SOLVED " + TEST_MODEL)

            

            output_data = {
                'id': data.get('id'),
                'problem': problem,
                #'solution': solution,
                #'answer': answer,
                'correct': model_succ
            }

            print("solved output for id " + str(data.get('id')))

            outfile.write(json.dumps(output_data) + '\n')

        linecount = linecount + 1
        time.sleep(4)

