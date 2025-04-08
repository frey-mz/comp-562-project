import os
import json
from pathlib import Path

INPUT_ROOT = "MATH"
OUTPUT_FILE = "math_dataset_clean.jsonl"

def normalize_math_dataset_to_jsonl(root_dir, output_file):
    rows_written = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
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
                                    out_f.write(json.dumps(row) + "\n")
                                    rows_written += 1
                        except Exception as e:
                            print(f"Error reading {json_file}: {e}")

    print(f"Wrote {rows_written} examples to {output_file}")

if __name__ == "__main__":
    normalize_math_dataset_to_jsonl(INPUT_ROOT, OUTPUT_FILE)
