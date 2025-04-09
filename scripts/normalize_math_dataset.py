import json
import random
from pathlib import Path

INPUT_ROOT = "MATH"
OUTPUT_FILE = "jsonls/input_jsonls/math_dataset_clean.jsonl"

def normalize_math_dataset_to_jsonl(root_dir, output_file):
    total_rows = 0

    # Split the two folders into train and test
    all_rows = {"train": [], "test": []}

    # Split into each segment
    for split in ["train", "test"]:
        split_path = Path(root_dir) / split
        for topic_path in split_path.iterdir():
            if topic_path.is_dir():
                topic = topic_path.name
                for json_file in topic_path.glob("*.json"):
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            problem = data.get("problem", "").strip()
                            solution = data.get("solution", "").strip()
                            problem_id = json_file.stem

                            if problem and solution:
                                row = {
                                    "id": f"{topic}_{split}_{problem_id}",
                                    "split": split,
                                    "topic": topic,
                                    "problem": problem,
                                    "solution": solution
                                }
                                all_rows[split].append(row)
                                total_rows += 1
                    
                    # Minor error handling
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

    # Shuffle train and test examples independently
    random.seed(10)
    random.shuffle(all_rows["train"])
    random.shuffle(all_rows["test"])

    # Combine the two segments as first train then test
    with open(output_file, "w", encoding="utf-8") as out_f:
        for row in all_rows["train"] + all_rows["test"]:
            out_f.write(json.dumps(row) + "\n")

    print(f"Wrote {total_rows} examples to {output_file}")

if __name__ == "__main__":
    normalize_math_dataset_to_jsonl(INPUT_ROOT, OUTPUT_FILE)
