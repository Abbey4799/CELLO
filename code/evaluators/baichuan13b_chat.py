from evaluators.evaluator import Evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.utils import GenerationConfig
# from peft import PeftModel
import torch
import pdb


class Baichuan13bChat_Evaluator(Evaluator):
    def __init__(self, model_path, LORA_WEIGHTS=''):
        super(Baichuan13bChat_Evaluator, self)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=False,
            add_eos_token=False,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.eval()

    def format_prompt(self, query, history, input=None):
        messages = []
        if len(history) == 0:
            messages.append({"role": "user", "content": query})
        else:
            for i, (old_query, response) in enumerate(history):
                messages.append({"role": "user", "content": old_query})
                messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": query})

        for i in messages:
            print(f"messages:{i}\n")
        return messages

    def generate(self, query, history):
        prompt = self.format_prompt(query, history)
        print(f"prompt:{prompt}")

        device = torch.device("cuda")

        with torch.no_grad():
            response = self.model.chat(
                tokenizer = self.tokenizer,
                messages = prompt
            )
        print(f"response:{response}\n")
        return response
