from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class BELLE_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(BELLE_Evaluator, self).__init__(model_path)
        # try adding 'mirror="tuna"' and 'resume_download=True' if facing the 'read timed out' problem
        # or directly clone the model
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelWithLMHead.from_pretrained(model_path, torch_dtype=torch.float16).half().cuda()

        self.device = torch.device("cuda")

        self.overall_instruction = ""
        
    def format_prompt(self, query, history, input=None):
        prompt = self.overall_instruction
        for i, (old_query, response) in enumerate(history):
            prompt += "Human:{}\n\nAssistant:{}".format(old_query, response)
        prompt += "Human:{}\n\nAssistant:".format(query)
        return prompt
    
    def generate(self, q, history):
        prompt = self.format_prompt(q, history)
        print(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=4096, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.35, repetition_penalty=1.2)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return response