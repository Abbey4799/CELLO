from evaluators.evaluator import Evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch
import pdb
import tqdm
import transformers
from typing import Optional, Dict, Sequence


class WizardLM13b_Evaluator(Evaluator):
    def __init__(self, model_path, LORA_WEIGHTS=''):
        super(WizardLM13b_Evaluator, self)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path
        )

        self.model.eval()

        self.temperature=1
        self.top_p=0.9
        self.top_k=40
        self.num_beams=1
        self.max_new_tokens=4096
        self.max_length_tokens=4096
        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

    def format_prompt(self, query, history, input=None):
        prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
        for i, (old_query, response) in enumerate(history):
            prompt += "USER: {} ASSISTANT:{}\n".format(old_query, response)
        prompt += "USER: {} ASSISTANT:".format(query)
        return prompt

    def generate(self, query, history):
        prompt = self.format_prompt(query, history)
        print(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = torch.device("cuda")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=16000,
                temperature=0.1,
                top_p=0.5,
                repetition_penalty=1.1
            )

        s = outputs[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(s)
        response = response.strip()
        response = response.replace("<end>", "").replace("<s>", "").replace("</s>", "")
        print(response)
        return response
