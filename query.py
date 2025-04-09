import json
import time
from transformation import get_basic_answer
from itertools import islice

try:
    start_line = int(input("Please enter the start line (0-based index): "))
    num_problems = int(input("Please enter the number of problems to solve: "))
    if start_line < 0 or num_problems <= 0:
        raise ValueError("Start line must be non-negative and number of problems must be positive.")
except ValueError as e:
    print(f"Invalid input: {e}")
    exit(1)

linecount = 0

input_file_path = 'math_dataset_clean.jsonl'
output_file_path = 'output.jsonl'

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
