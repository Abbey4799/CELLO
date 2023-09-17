import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import  PeftModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class Alpaca_Evaluator(Evaluator):
    def __init__(self, model_path, LORA_WEIGHTS = ""):
        super(Alpaca_Evaluator, self).__init__(model_path)
        # try adding 'mirror="tuna"' and 'resume_download=True' if facing the 'read timed out' problem
        # or directly clone the model
        self.tokenizer = LlamaTokenizer.from_pretrained(LORA_WEIGHTS)
        self.base_model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
        )
        model_vocab_size = self.base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(self.tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size!=tokenzier_vocab_size:
            assert tokenzier_vocab_size > model_vocab_size
            print("Resize model embeddings to fit tokenizer")
            self.base_model.resize_token_embeddings(tokenzier_vocab_size)
        if LORA_WEIGHTS is not None:
            print("loading peft model")
            self.model = PeftModel.from_pretrained(self.base_model, LORA_WEIGHTS,torch_dtype=torch.float16,device_map='auto',)
        else:
            self.model = self.base_model
        self.model = self.model.half()
        self.model = self.model.eval()
        
        # 33b
        if '33b' in model_path:
            self.generation_config = dict(
                temperature=0.2,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.1,
                max_new_tokens=4096
                )
        # 13b 7b
        else:
            self.generation_config = dict(
                temperature=0.2,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.3,
                max_new_tokens=4096
                )   
        


        self.overall_instruction = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        
    def format_prompt(self, query, history, memory_limit = 10, input=None):
        prompt = self.overall_instruction

        for i, (old_query, response) in enumerate(history[-memory_limit:]):
            prompt += "### Instruction:\n\n{}\n\n### Response:\n\n{}".format(old_query, response)

        prompt += "### Instruction:\n\n{}\n\n### Response:\n\n".format(query)
        return prompt
    
    def generate(self, q, history):
        prompt = self.format_prompt(q, history)
        print(prompt)

        inputs = self.tokenizer(prompt,return_tensors="pt") 
        generation_output = self.model.generate(
                    input_ids = inputs["input_ids"].to("cuda"),
                    attention_mask = inputs['attention_mask'].to("cuda"),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self.generation_config
                    )
        s = generation_output[0]
        output = self.tokenizer.decode(s,skip_special_tokens=True)
        response = output.split("### Response:")[-1].strip()

        return response