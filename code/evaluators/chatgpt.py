import pandas as pd
from tqdm import tqdm
import hashlib
import json
import glob
import os
import requests
import openai
from openai.error import RateLimitError
import backoff
import time


openai.api_key = '[your api key]'

class ChatGPT_Evaluator():
    def __init__(self):
        super(ChatGPT_Evaluator, self).__init__()
        self.model = 'ChatGPT'
        self.metadata = {'model': 'gpt-3.5-turbo',
                         'temperature': 0.8,
                         'max_tokens': 4096
                         }
        self.template = '''
        '''
        
    def format_prompt(self, query, history):
        messages = []
        for _ in history:
            messages.append({'role': 'user', 'content': _[0]})
            messages.append({'role': 'assistant', 'content': _[1]})
        messages.append({'role': 'user', 'content': self.template + query})
        return messages
    
    @backoff.on_exception(backoff.expo, RateLimitError)
    def generate(self, q, history):
        prompt = self.format_prompt(q, history)
        print(prompt)
        
        inference_not_done = True
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                temperature=0.7
            )
            res_msg = completion.choices[0].message
            time.sleep(20)
            return res_msg["content"].strip()
        except Exception as e:
            print(f"Waiting 3 minutes")
            print(f"Error was: {e}")
            time.sleep(180)