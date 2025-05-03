import json
import time
from transformation import get_basic_answer
from itertools import islice

available_inputs = {
    "math": "jsonls/input_jsonls/math_dataset_clean.jsonl",
    "omni": "jsonls/input_jsonls/omni_math_dataset_clean.jsonl",
}

# Prompt user for dataset name
print("Available datasets:")
for key in available_inputs:
    print(f"{key}")

dataset_choice = input("Which dataset would you like to use? ").strip().lower()

if dataset_choice not in available_inputs:
    print(f"Invalid dataset selection: {dataset_choice}")
    exit(1)

input_file_path = available_inputs[dataset_choice]
output_file_path = f"jsonls/output_jsonls/output_{dataset_choice}.jsonl"
linecount = 0

# Prompt user for line range
try:
    start_line = int(input("Enter the start line (0-based index): "))
    num_problems = int(input("Enter the number of problems to solve: "))
    if start_line < 0 or num_problems <= 0:
        raise ValueError
except ValueError:
    print("Start line must be non-negative and number of problems must be positive.")
    exit(1)

print(f"Processing {num_problems} problems starting from line {start_line}...")

with open(input_file_path, 'r', encoding='utf-8') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:

    lines_to_process = islice(infile, start_line, start_line + num_problems)

    for line in lines_to_process:
        if line.strip():  
            data = json.loads(line)

            problem = data.get('problem')
            solution = data.get('solution')
            current_id = data.get('id', f'unknown_line_{start_line + linecount}')

            print(f"Processing problem ID: {current_id} (Original line: {start_line + linecount})")
            answer = ""
            success = False

            while not success:
                try:
                    answer = get_basic_answer(problem)
                    success = True
                    break
                except Exception as e:
                    if "429" in str(e):
                        error_string = str(e)
                        delay_seconds = int(error_string.split("retryDelay")[1].split("s")[0].split(": '")[1])
                        print(f"Rate limit encountered for ID {current_id}. Retrying in {delay_seconds} seconds...")
                        time.sleep(delay_seconds)
                    elif "503" in str(e):
                        time.sleep(60)
                        print(f"Service unavailable for ID {current_id}. Retrying in 60 seconds...")
                    else:
                        print(f"Non-rate-limit API Error for ID {current_id}: {e}")
                        break

            if not success:
                print(f"Failed to process ID {current_id} after rate limit retries.")
                break

            output_data = {
                'original_line': start_line + linecount,
                'id': current_id,
                'problem': problem,
                'solution': solution,
                'answer': answer
            }

            json.dump(output_data, outfile)
            outfile.write('\n')
            linecount += 1
            print(f"Processed line {start_line + linecount} (ID: {current_id})")
        else:
            print(f"Skipping empty line at original line number {start_line + linecount}")
            linecount += 1

print(f"Finished processing. {linecount} problems attempted.")
print(f"Output written to {output_file_path}")
