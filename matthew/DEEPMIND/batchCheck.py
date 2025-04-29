import string
from problemGen import problem_gen
import json
import sys
sys.path.insert(0, '..')

import grader


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



def isEquiv(x, sol):
    x = x.strip()
    sol = sol.strip()

    ans_box = _extract_boxed_text(x)
    ans = ans_box[-1]

    if len(ans) == 1 and len(sol) == 1:
        try:
            index = string.ascii_lowercase.index(sol) + 1
            if index == int(ans):
                return True
        except:
            pass


    ans = ans.replace(r'\text{Yes}', "True")
    ans = ans.replace(r'\text{No}', "False")

    ans = ans.split('=', 1)[1] if '=' in ans else ans

    res = False
    if ',' in sol:
        res = arrays_are_equal(ans.split(','), sol.split(','), lambda x, y: grader.grade_answer(x,y))
    else:
        res = grader.grade_answer(ans, sol)

    #return res

    if not res:
        print("FAIL, SOL, MODEL ANS")
        print(sol)
        print(ans)

    return res


wins = 0
fails = 0

with open("batchRes.jsonl", 'r', encoding='utf-8') as infile, \
open("outputBatch.jsonl", 'w', encoding='utf-8') as outfile:
  for idx, i in enumerate(problem_gen("train-hard")):
    try:
        res = json.loads(next(infile))
    except StopIteration:
        break

    answer = res["response"]["body"]["output"][0]["content"][0]["text"]

    correct = isEquiv(answer, i[1])

    output_data = {
    'id': str(idx),
    'problem': i[0],
    'correct': correct,
    'costs': res["response"]["body"]["usage"]
    }

    if not correct:
        fails+=1
    else: wins +=1


    outfile.write(json.dumps(output_data) + '\n')
    outfile.flush()

print(wins, fails)