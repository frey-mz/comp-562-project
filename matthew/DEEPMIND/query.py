import os
import sys
sys.path.insert(0, '..')
import models
import grader
import json

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

def problem_gen(directory):
    # Open all files in the directory
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
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


def isEquiv(x, sol):
    ans_box = _extract_boxed_text(x)
    ans = ans_box[-1]
    res = grader.grade_answer(ans, sol)

    if not res:
        print("FAIL, SOL, MODEL ANS")
        print(ans)
        print(sol)

    return res


offset = int(input("Please enter the start line: "))
end = int(input("Please enter problems to solve: "))

with open("output.jsonl", 'w', encoding='utf-8') as outfile:
    for idx, two_lines in enumerate(problem_gen("train-hard")):
        if idx < offset:
            continue
        
        [problem, solution] = two_lines

        print("new output for id " + str(idx))
        system_prompt = "You are a math problem solver. Please first solve any problem given to you step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\". \n."
        res = models.getMultiModelResponse(problem, system_prompt, lambda x: isEquiv(x, solution))
        if res == None:
            continue
        
        correct, costs = res

        output_data = {
            'id': str(idx),
            'problem': problem,
            'correct': correct,
            'costs': costs
        }

        print("solved output for id " + str(idx))

        outfile.write(json.dumps(output_data) + '\n')
        outfile.flush()

        if idx == end:
            break