from transformers import AutoTokenizer, AutoModelForCausalLM
# from auto_gptq import AutoGPTQForCausalLM
from evaluators.evaluator import Evaluator
import torch
import pdb

class LLama2_FlagAlpha_Evaluator(Evaluator):
    def __init__(self, model_path):
        super().__init__(model_path)
        # self.model = AutoGPTQForCausalLM.from_quantized(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
        self.device = 'cuda'

    def format_prompt(self, query, history, memory_limit=10, input=None):
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += f'<s>Human: {old_query}\n</s><s>Assistant: {response}'

        prompt += f'<s>Human: {query}\n</s><s>Assistant: '
        return prompt
    
    def generate(self, q, history):
        # pdb.set_trace()
        prompt = self.format_prompt(q,history)
        # input_ids = self.tokenizer(prompt, return_tensors="pt",add_special_tokens=False).to(self.device)    
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)    
        generate_input = {
            "input_ids":inputs.input_ids,
            "max_new_tokens":2048,
            "do_sample":True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":self.tokenizer.eos_token_id,
            "bos_token_id":self.tokenizer.bos_token_id,
            "pad_token_id":self.tokenizer.pad_token_id
        }
        generate_ids  = self.model.generate(**generate_input)
        response = self.tokenizer.decode(generate_ids[0])
        if '<s>Assistant:' in response:
            response = response.split("<s>Assistant:")[-1]
        elif '<s> Assistant:' in response:
            response = response.split("<s> Assistant:")[-1]
        else:
            response = response.replace(prompt,"")
        print(f"response:{response}")
        return response
