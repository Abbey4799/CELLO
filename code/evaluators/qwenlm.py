import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from evaluators.evaluator import Evaluator


class QwenLM_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(QwenLM_Evaluator, self).__init__(model_path)
        # try adding 'mirror="tuna"' and 'resume_download=True' if facing the 'read timed out' problem
        # or directly clone the model
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

        self.overall_instruction = ""
        
    def format_prompt(self, query, history, input=None):
        prompt = query
        return prompt
    
    def generate(self, q, history):
        prompt = self.format_prompt(q, history)
        print(prompt)

        response, history = self.model.chat(self.tokenizer, prompt, history = history)

        return response