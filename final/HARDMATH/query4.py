import os
import re
import argparse 
import utils
import create_prompt
import sys
sys.path.insert(0, '..')
import models
import grader
from tqdm import tqdm
import answer_extraction
from sympy import simplify, sympify

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

def is_equiv(x, y):
    ans_box = _extract_boxed_text(x)
    sol_box = _extract_boxed_text(y)

    ans = ans_box[-1]
    sol = sol_box[-1]

    return grader.grade_answer(ans, sol)

def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    return True

import os

if __name__ == '__main__':
    TEST_MODELS = ["gemini", "gemma", "deepseek"]
    GRADING_MODEL = "gemini"
    data = utils.read_json("hardmath.json")
    data = {key: value for key, value in data.items() if value.get('question_type')  in ["integral", "ODE","polynomial_roots", "nondimensionalization_symbolic", 'nondimensionalization_numeric']}
    # load examples
    # output file
    output_file = "hardmath_output.json"
    # filter problems for testing
    test_pids = list(data.keys())
    print("Number of test problems in total:", len(test_pids))
    # strings
    skip_pids = list(map(str, range(0,750)))
    print("Removing problems with existing valid response...")
    with open("hardmath_output4.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            obj = json.loads(line)
            if "id" in obj:
                skip_pids.append(obj["id"])
    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))
    with open("hardmath_output4.jsonl", 'a+', encoding='utf-8') as outfile:
        for pid in test_pids:
            problem_dict = data[pid]
            examples = {}
            for i in test_pids:
                if i == pid:
                    continue
                if problem_dict["question_type"] == data[i]["question_type"]:
                    examples[i] = data[i]
                    if len(examples) >= 10:
                        break
            user_prompt = create_prompt.create_query_prompt(problem_dict, examples)
            system_prompt = (
            "You are a helpful assistant designed to help with advanced applied mathematics problems, "
            "specifically focusing on tasks like nondimensionalizing polynomials, using approximation methods to solve for polynomial "
            "roots, PDEs, integrals, etc. When given a physical math question, you should answer the question according to the user's prompt."
            )
            print(f"Generating response for {pid}...")
            try:
                def isCorrect(response):
                    print("trying new resp")
                    latex_response = utils.display_content(response,False)
                    system_prompt = (
                    "You are a helpful grading assistant designed to help with advanced applied mathematics problems, "
                    "specifically focusing on tasks like nondimensionalizing polynomials, using approximation methods to solve for polynomial roots, PDEs, integrals, etc. "
                    "When given a response and a ground truth solution, you should score the response according to the user's grading criteria."
                    )
                    integral_subtype = None
                    if problem_dict["question_type"] == 'integral':
                        if problem_dict.get('answer_type') == 'list':
                            integral_subtype = 'traditional'
                        elif problem_dict.get('answer_type') == 'math_expression':
                            integral_subtype = 'laplace'
                    grading_prompt = create_prompt.create_grading_prompt(latex_response, problem_dict['solution'],\
                                    question_type=problem_dict["question_type"],integral_subtype=integral_subtype)
                    grade_response = models.getOpenAIResponseTexts("gpt-4.1", grading_prompt, system_prompt)
                    if len(grade_response) == 0:
                        print("no grade response?? thatsn ot good ")
                        return False
                    latex_grade_response = utils.display_content(grade_response[-1],False)
                    res = answer_extraction.extract_final_answer_allform(latex_response = latex_grade_response, answer_type = 'float')
                    res = res != None and float(res) == 1
                    if not res:
                        print(grade_response)
                    return res
                res = models.getMultiModelResponse(user_prompt, system_prompt, lambda x: isCorrect(x))
                if res == None:
                    continue
                correct, costs = res

                output_data = {
                    'id': pid,
                    'problem': problem_dict['question'],
                    #'solution': problem_dict['answer_val'],
                    #'answer': answer,
                    'correct': correct,
                    'costs': costs
                }

                print("solved output for id " + pid)
                outfile.write(json.dumps(output_data) + '\n')
                outfile.flush()


            
            except Exception as e:
                print(e)
                print(f"Error in processing for {pid}")
                #results[pid]['error'] = e