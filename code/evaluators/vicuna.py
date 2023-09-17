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
import pdb


class Vicuna_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(Vicuna_Evaluator, self).__init__(model_path)

        self.model, self.tokenizer = load_model(
        model_path = model_path,
        device = "cuda",
        num_gpus = 3 # 用几张改几张
        )
        self.overall_instruction = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        
    def format_prompt(self, query, history, memory_limit = 10, input=None):
        roles=("USER", "ASSISTANT")
        seps = [" ", "</s>"]
        messages = []
        if len(history) != 0:
            for (old_query, response) in enumerate(history[-memory_limit:]):
                messages.append([roles[0],old_query])
                messages.append([roles[1],response])

        messages.append([roles[0], query])
        messages.append([roles[1], ''])

        prompt = self.overall_instruction + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                prompt += str(role) + ": " + str(message) + str(seps[i % 2])
            else:
                prompt += str(role) + ":"

        return prompt
    
    @torch.inference_mode()
    def generate(self, q, history):
        prompt = self.format_prompt(q, history)
        input_ids = self.tokenizer([prompt]).input_ids
        output_ids = self.model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.0,
            max_new_tokens=16000,
        )
        if self.model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]

        response = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
        print(f"self.conv.roles[0]: {q}")
        print(f"self.conv.roles[1]: {response}")
        return response
