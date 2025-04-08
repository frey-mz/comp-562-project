import json
import pandas as pd
from google import genai
from google.genai import types
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_basic_answer(problem):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
        system_instruction="You are a math problem solver. Please first solve any problem given to you step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\". \n"),
        contents=problem,
    )

    return json.loads(response.text)
    

def get_solvable(df):
    for index, row in df.iterrows():
        problem, solution = row["problem"], row["solution"]
        basic_output = get_basic_answer(problem)["answer"]
        solvable = get_validation(basic_output, solution)
        df.at[index, "solvable"] = solvable
    return df

def add_validation():
    rows = []
    with open("math_dataset_clean.jsonl", "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    problem_df = pd.DataFrame(rows)

    get_solvable(problem_df)

    output_df = problem_df[["id", "solvable"]]
    return output_df

def get_dataset():
    return add_validation()

