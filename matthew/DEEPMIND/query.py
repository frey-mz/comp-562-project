import os
import sys
sys.path.insert(0, '..')
import models
import grader
import json
import string

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


def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def isEquiv(x, sol):
    if string.ascii_lowercase.index(x) + 1 == safe_cast(sol, int):
        print(x)
        print(sol)

        return True

    ans_box = _extract_boxed_text(x)
    ans = ans_box[-1]
    if s == "\\text\{Yes\}":
        ans = "True"
    elif ans == "\\text\{No\}":
        ans = "False"

    ans = ans.split('=', 1)[1] if '=' in ans else ans

    res = False
    if ',' in sol:
        res = arrays_are_equal(ans.split(','), sol.split(','), lambda x, y: grader.grade_answer(x,y))
    else:
        res = grader.grade_answer(ans, sol)

    return res

    if not res:
        print("FAIL, SOL, MODEL ANS")
        print(ans)
        print(sol)

    return res


def arrays_are_equal(arr1, arr2, is_equal) -> bool:
    if len(arr1) != len(arr2):
        return False

    used_indices = set()

    for item1 in arr1:
        found_match = False
        for idx, item2 in enumerate(arr2):
            if idx not in used_indices and is_equal(item1, item2):
                used_indices.add(idx)
                found_match = True
                break
        if not found_match:
            return False

    return True

offset = int(input("Please enter the start line: "))
end = int(input("Please enter problems to solve: "))


with open("output.jsonl", 'w', encoding='utf-8') as outfile:
    for idx, two_lines in enumerate(problem_gen("train-hard")):
        if idx < offset:
            continue
        
        [problem, solution] = two_lines

        print("new output for id " + str(idx))
        system_prompt = (
             "You are a math problem solver."
             "Please first solve any problem given to you step by step"
             "then put your final answer in one \"\\boxed{}\""
             "if the answer asks for a list of numbers, put a comma between each number."
             "if the answer asks for a True/False response, write either 'True' or 'False'"
        )
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