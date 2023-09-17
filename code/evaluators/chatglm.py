import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class ChatGLM_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(ChatGLM_Evaluator, self).__init__(model_path)
        # try adding 'mirror="tuna"' and 'resume_download=True' if facing the 'read timed out' problem
        # or directly clone the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

        self.device = torch.device("cuda")
        
    def format_prompt(self, query, history, input=None):
        prompt = query
        return prompt
    
    def generate(self, q, history):
        prompt = self.format_prompt(q, history)
        print(prompt)

        response, history = self.model.chat(self.tokenizer, prompt, history = history)

        return response
    