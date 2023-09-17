from typing import Optional
from evaluators.evaluator import Evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import transformers
import torch
import pdb


class Openchat_Evaluator(Evaluator):
    def __init__(self, model_path):
        super().__init__(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = transformers.pipeline(
                    "text-generation",
                    model = model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        self.device = torch.device("cuda")


    def format_prompt(self, query, history, memory_limit=3, input=None):
        prompt = ""
        if len(history) != 0:
            for i, (old_query, response) in enumerate(history):
                prompt += f"GPT4 User:{old_query.strip()}<|end_of_turn|> GPT4 Assistant:{response.strip()}<|end_of_turn|>"
        prompt += f"GPT4 User:{query.strip()}<|end_of_turn|>GPT4 Assistant:"
        # print(prompt)
        return prompt

    def generate(self, q, history):
        prompt = self.format_prompt(q, history)

        sequences = self.pipeline(
            prompt,
            do_sample=False, # 选择概率最大的
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=10000,
        )

        output = ""
        for seq in sequences:
            print(f"seq:{seq}")
            print(f"Result: {seq['generated_text']}")
            try:
                output += seq["generated_text"]
            except:
                output += seq[0]['generated_text']
        print(f"output:{output}")
        try:
            response = output.replace(prompt,"")
        except:
            for i in prompt:
                response = output.replace(i,"")
        print(f"response:{response}")
        return response