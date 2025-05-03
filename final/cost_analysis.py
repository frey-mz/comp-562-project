import pandas as pd
from pathlib import Path
import numpy as np

# $/1m tokens as of 27-apr-2025
_MODEL_RATES = {
    0: {"in": 0.40, "out": 1.60},   # gpt-4.1-mini
    1: {"in": 2.00, "out": 8.00},   # gpt-4.1
    2: {"in": 1.10, "out": 4.40},   # o4-mini
}

_per_token = lambda ppm: ppm / 1_000_000

def _estimate_cost(row):
    idx = row["correct"]
    if idx == 3:
        return -1.0
    rate   = _MODEL_RATES[idx]
    token_counts = row["cost"][idx]
    return token_counts["input_tokens"]  * _per_token(rate["in"]) + \
           token_counts["output_tokens"] * _per_token(rate["out"])


def _create_cost_adjusted():
    # load
    df = pd.read_json("problems.jsonl", lines=True)

    # compute + overwrite `cost`
    df["cost"] = df.apply(_estimate_cost, axis=1)

    # dump
    output_path = Path("cost_adjusted.jsonl")
    df.to_json(output_path, orient="records", lines=True)
    print("wrote", output_path.resolve())


def cost_analysis():
    problems = pd.read_json("problems.jsonl", lines=True)
    cost_adjusted = pd.read_json("cost_adjusted.jsonl", lines=True)

    reasoning_solvable = cost_adjusted[problems["correct"] == 2]
    reasoning_cost_sum = reasoning_solvable["cost"].sum()
    unsolvable = cost_adjusted[problems["correct"] == 3]
    unsolvable_cost_sum = unsolvable["cost"].sum()

    avg_reasoning_problem_cost = (reasoning_cost_sum + unsolvable_cost_sum) / (len(reasoning_solvable) + len(unsolvable))

    avg_gpt41_cost = cost_adjusted[problems["correct"] == 1]["cost"].mean()

    # create new jsonl with savings computed based on the price of which model could solve minus the price of the higher model (so if a problem could be solved with 4.1 instead of o4, 4.1 - o4 = savings)
    cost_adjusted["savings"] = np.where(
        problems["correct"] == 1,
        cost_adjusted["cost"] - avg_reasoning_problem_cost,
        np.where(
            problems["correct"] == 0,
            cost_adjusted["cost"] - avg_gpt41_cost,
            0
        )
    )

    # dump
    output_path = Path("cost_adjusted_with_savings.jsonl")
    cost_adjusted.to_json(output_path, orient="records", lines=True)
    print("wrote", output_path.resolve())

def main():
    _create_cost_adjusted()
    cost_analysis()