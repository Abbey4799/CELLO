import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator


class InterLM7b_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(InterLM7b_Evaluator, self).__init__(model_path)
        # try adding 'mirror="tuna"' and 'resume_download=True' if facing the 'read timed out' problem
        # or directly clone the model
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()
        self.model = self.model.eval()

        self.overall_instruction = ""
        
    def format_prompt(self, query, history, input=None):
        prompt = query
        return prompt
    
    def generate(self, q, history):
        prompt = self.format_prompt(q, history)
        print(prompt)

        response, history = self.model.chat(self.tokenizer, prompt, history = history)

        return response