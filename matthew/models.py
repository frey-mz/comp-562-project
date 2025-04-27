import openai
import time
import requests
import json
from google import genai
from google.genai import types

import ollama

#sync cpu option

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