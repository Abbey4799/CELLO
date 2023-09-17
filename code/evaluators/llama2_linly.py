from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluators.evaluator import Evaluator
import torch
import pdb

class LLama2_Linly_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(LLama2_Linly_Evaluator,self).__init__(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.device = torch.device('cuda')


    def format_prompt(self, query, history, memory_limit=3, input=None):
        prompt = ""
        # 单轮对话
        if len(history) == 0:
            prompt = f"### Instruction:{query.strip()}  ### Response:"
        # 多轮对话
        else:
            for i, (old_query, response) in enumerate(history):
                prompt += f"### Instruction:{old_query.strip()}  ### Response:{response.strip()}"
            prompt += f"### Instruction:{query.strip()}  ### Response:"
        print(prompt)
        return prompt

    def generate(self, q, history):
        # pdb.set_trace()
        prompt = self.format_prompt(q,history)
        # print(f"prompt:{prompt}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids, do_sample=True, max_new_tokens=4096, top_k=10, top_p=0.85, temperature=1, repetition_penalty=1.15, eos_token_id=2, bos_token_id=1, pad_token_id=0).to(self.device)
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # response = response.lstrip(prompt)
        response = response.split("### Response:")[-1]
        # print(f"response:{response}")
        return response

