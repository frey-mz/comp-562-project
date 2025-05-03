import openai
import time

from openai import OpenAI

client = OpenAI(max_retries=10)



def getMultiModelResponse(problem, system, isCorrect):
    models = ["gpt-4.1-mini", "gpt-4.1", "o4-mini"]
    curr = 0
    costs = []
    for model in models:
        resp = getOpenAIResponse(model, problem, system)
        if resp.error != None:
            print(resp.error)
            return None
        costs.append({
            'input_tokens': resp.usage.input_tokens,
            'output_tokens': resp.usage.output_tokens,
        })
        correct = False
        for output in resp.output:
            if output.type == "reasoning":
                for content in output.summary:
                    if isCorrect(content.text):
                        correct = True
                        break
            else:
                for content in output.content:
                    if content.type == "output_text" and isCorrect(content.text):
                        correct = True
                        break
            if correct:
                break
        if correct: 
            print(model + " WON")
            break
        print(model + " FAILED")
        curr+=1
    return (curr, costs)



def getOpenAIResponse(model, problem, system):
    response = client.responses.create(
        model= model,
        input= problem,
        instructions = system
    )
    return response

def getOpenAIResponseTexts(model, problem, system):
    resp = getOpenAIResponse(model, problem, system)
    if resp.error != None:
        print(resp.error)
        return None
    texts = []
    for output in resp.output:
        if output.type == "reasoning":
            for content in output.summary:
                texts.append(content.text)
            break
        else:
            for content in output.content:
                texts.append(content.text)
    return texts

"""
class DeepSeek_Model():
    def __init__(self, system_prompt=""):
        self.system_prompt = system_prompt
    def get_response(self, problem):
        return ollama.generate(model='deepscaler', system=self.system_prompt, prompt=problem)['response']

class Gemma_Model():
    def __init__(self, system_prompt=""):
        self.system_prompt = system_prompt
    def get_response(self, problem):
        return ollama.generate(model='gemma3', system=self.system_prompt, prompt=problem)['response']

class Gemini_Model():
    def __init__(self, api_key, model="gemini-2.0-flash", system_prompt=""):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt

    def get_response(self, user_prompt):
        patience = 10
        while patience > 0:
            patience = patience - 1
            try:
                response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(
                system_instruction=self.system_prompt),
                contents=user_prompt,
                )
                return response.text
            except Exception as e:
                print("error", e)
            time.sleep(10)
        return ""
"""