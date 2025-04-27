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

def remove_dollar_signs(text):
    return text.replace('$', '')

def verify_response(response):
    if isinstance(response, str):
        response = response.strip() 
    if response == "" or response == None:
        return False
    return True

def compare_answers(extracted_answer, model_answer):
    if model_answer is None:
        return 0
    try:
        # Convert the string answers to sympy expressions
        extracted_answer_expr = utils.safe_parse_latex(remove_dollar_signs(extracted_answer))
        model_answer_expr = utils.safe_parse_latex(remove_dollar_signs(model_answer))
        # Compare the simplified difference
        if simplify(extracted_answer_expr - model_answer_expr) == 0:
            return 1
    except Exception as e:
        print(f"Error in comparing answers: {e}")
        return 0
    
    return 0

import os

def load_model(role, t): #helper models
    # Load the API key from environment or use the provided key
    key = os.getenv("GEMINI_API_KEY")
    system_prompt = ""

    if role == 'grader':
        system_prompt = (
                "You are a helpful grading assistant designed to help with advanced applied mathematics problems, "
                "specifically focusing on tasks like nondimensionalizing polynomials, using approximation methods to solve for polynomial roots, PDEs, integrals, etc. "
                "When given a response and a ground truth solution, you should score the response according to the user's grading criteria."
            )
    else:
        system_prompt = (
                    "You are a helpful assistant designed to help with advanced applied mathematics problems, "
                    "specifically focusing on tasks like nondimensionalizing polynomials, using approximation methods to solve for polynomial "
                    "roots, PDEs, integrals, etc. When given a physical math question, you should answer the question according to the user's prompt."
                )

    if t == "gemini":
        model = models.Gemini_Model(api_key = key, system_prompt = system_prompt)
    elif t=="gemma":
        model = models.Gemma_Model(system_prompt = system_prompt)
    else:
        model = models.DeepSeek_Model(system_prompt = system_prompt)


    return model


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


    skip_pids = []
    print("Removing problems with existing valid response...")

    with open("hardmath_output.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            obj = json.loads(line)
            if "id" in obj:
                skip_pids.append(obj["id"])
    test_pids = [pid for pid in test_pids if pid not in skip_pids]

    print("Number of test problems to run:", len(test_pids))


    with open("hardmath_output.jsonl", 'w', encoding='utf-8') as outfile:
        for i in range(100):
            test_pids.pop(0)
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
            #print(results)
            print(f"Generating response for {pid}...")
            try:
                model_succ = {}
                for TEST_MODEL in TEST_MODELS:
                    model = load_model('math_assistant', TEST_MODEL)
                    #results[pid] = {}
                    #results[pid]['prompt'] = user_prompt
                    response = model.get_response(user_prompt)
                    latex_response = utils.display_content(response,False)
                    #results[pid]['response'] = response
                    correct = False

                    if 'nondimensionalization' in problem_dict["question_type"]:
                        extracted_answer = problem_dict['answer_val']
                        #results[pid]['extracted_answer'] = extracted_answer
                        model_answer = answer_extraction.extract_final_answer_allform(latex_response = latex_response, answer_type=problem_dict['answer_type'])
                        #matches = re.findall(r'\$(.*?)\$', latex_response, re.DOTALL)
                        #boxed_list = [match for match in matches if "boxed" in match]
                        #results[pid]['model_answer'] = model_answer
                        #results[pid]['score'] = compare_answers(extracted_answer, model_answer)
                        if grader.grade_answer(extracted_answer, model_answer):
                            print("WOOO")
                            correct = True
                    else:
                        grading_model = load_model('grader', GRADING_MODEL)

                        integral_subtype = None

                        if problem_dict["question_type"] == 'integral':
                            if problem_dict.get('answer_type') == 'list':
                                integral_subtype = 'traditional'
                            elif problem_dict.get('answer_type') == 'math_expression':
                                integral_subtype = 'laplace'


                        grading_prompt = create_prompt.create_grading_prompt(latex_response, problem_dict['solution'],\
                                        question_type=problem_dict["question_type"],integral_subtype=integral_subtype)
                        #results[pid]['grade_prompt'] = grading_prompt
                        grade_response = grading_model.get_response(grading_prompt)
                        #results[pid]['grade_response'] = grade_response
                        latex_grade_response = utils.display_content(grade_response,False)
                        #results[pid]['score'] = answer_extraction.extract_final_answer_allform(latex_response = latex_grade_response,answer_type = 'float')
                        res = answer_extraction.extract_final_answer_allform(latex_response = latex_grade_response,answer_type = 'float')
                        if res != None and int() == 1:
                            correct = True
                    model_succ[TEST_MODEL] = correct
                    print(TEST_MODEL + " processed")

                results = {
                    'id': pid,
                    'problem': problem_dict["question"],
                    #'solution': problem_dict["answer_val"],
                    #'answer': response,
                    'correct': model_succ
                }
                print("writing response")
                outfile.write(json.dumps(results) + '\n')
            
            except Exception as e:
                results = {
                    'id': pid,
                    'error': e,
                }
                outfile.write(json.dumps(results) + '\n')
                print(e)
                print(f"Error in processing for {pid}")
                #results[pid]['error'] = e