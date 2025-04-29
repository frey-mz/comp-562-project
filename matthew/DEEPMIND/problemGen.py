import os
def problem_gen(directory):
    # Open all files in the directory
    files = []
    for filename in sorted(os.listdir(directory)):
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

system_prompt = (
             "You are a math problem solver."
             "Please first solve any problem given to you step by step"
             "then put your final answer in one \"\\boxed{}\""
             "if the answer asks for a list of numbers, put a comma between each number."
             "if the answer asks for a True/False response, write either 'True' or 'False'"
        )
