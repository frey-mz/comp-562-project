import re
def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def is_equiv(db_solution, ai_solution, verbose=False):
    if db_solution is None and ai_solution is None:
        print("WARNING: Both None")
        return True
    if db_solution is None or ai_solution is None:
        return False

    try:
        b1 = _extract_boxed_text(db_solution)
        if len(b1) == 0:
            print("db solution without boxed text found, not good error")
            return False
        b2 = _extract_boxed_text(ai_solution)
        if len(b2) == 0:
            print("db solution without boxed text found, kinda not good error")
            return False
        ss1 = _strip_string(b1[0])
        ss2 = _strip_string(b2[0])
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return db_solution == ai_solution


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

def main():
    e1 = R"""Let's break this down step by step.

**1. Find the expression for h(x):**
We are given $f(x) = 7x + 5$ and $g(x) = x - 1$.
We need to find $h(x) = f(g(x))$. This means we substitute $g(x)$ into $f(x)$ wherever we see $x$.
$h(x) = f(x-1) = 7(x-1) + 5$
$h(x) = 7x - 7 + 5$
$h(x) = 7x - 2$

**2. Find the inverse of h(x):**
To find the inverse of a function, we can follow these steps:
   a. Replace $h(x)$ with $y$:
      $y = 7x - 2$
   b. Swap $x$ and $y$:
      $x = 7y - 2$
   c. Solve for $y$:
      $x + 2 = 7y$
      $y = \frac{x + 2}{7}$
   d. Replace $y$ with $h^{-1}(x)$:
      $h^{-1}(x) = \frac{x + 2}{7}$

So, the inverse of $h(x)$ is $\frac{x+2}{7}$.

Final Answer: The final answer is $\boxed{\frac{x+2}{7}}$')"""

    e2 = R"""\[h(x)=f(g(x))=7(x-1)+5=7x-2.\]Let's replace $h(x)$ with $y$ for simplicity, so \[y=7x-2.\]In order to invert $h(x)$ we may solve this equation for $x$. That gives \[y+2=7x\]or \[x=\frac{y+2}{7}.\]Writing this in terms of $x$ gives the inverse function of $h$ as \[h^{-1}(x)=\boxed{\frac{x+2}{7}}.\]"""

    x = is_equiv(e1, e2)
    print(x)

if __name__ == '__main__':
    main()