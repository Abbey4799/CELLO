"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.3
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastchat.model import load_model, get_conversation_template, add_model_args
from evaluators.evaluator import Evaluator


class LongChat_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(LongChat_Evaluator, self).__init__(model_path)

        self.model, self.tokenizer = load_model(
        model_path = model_path,
        device = "cuda",
        num_gpus = 1,
        )


        self.overall_instruction = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        
    def format_prompt(self, query, history, memory_limit=10, input=None):
        roles=("USER", "ASSISTANT")
        seps = [" ", "</s>"]
        messages = []
        for (old_query, response) in history[-memory_limit:]:
            messages.append([roles[0],old_query])
            messages.append([roles[1],response])

        messages.append([roles[0], query])
        messages.append([roles[1], ''])

        prompt = self.overall_instruction + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                prompt += role + ": " + message + seps[i % 2]
            else:
                prompt += role + ":"

        return prompt
    
    @torch.inference_mode()
    def generate(self, q, history):
        prompt = self.format_prompt(q, history)
        print(prompt)


        input = self.tokenizer(prompt, return_tensors="pt")
        prompt_length = input.input_ids.size()[-1]
        output = self.model.generate(input.input_ids.to(self.model.device), max_new_tokens=32000, use_cache=False)[0]
        output = output[prompt_length:]
        output = self.tokenizer.batch_decode([output], skip_special_tokens=True)

        response = output[0]

        return response