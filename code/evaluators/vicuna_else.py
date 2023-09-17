from evaluators.evaluator import Evaluator
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel
import torch

class Vicuna_else_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(Vicuna_else_Evaluator, self).__init__(model_path)

        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        
        self.device = torch.device("cuda")
        self.overall_instruction = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        self.generation_config = GenerationConfig(
             do_sample=True,
            temperature=0.7,
            repetition_penalty=1.0,
            max_new_tokens=16000,
        )
    
    def format_prompt(self, query, history, memory_limit = 10, input=None):
        prompt = self.overall_instruction
        for i, (old_query, response) in enumerate(history[-memory_limit:]):
            prompt += "USER: {} ASSISTANT: {}</s>".format(old_query, response)
        prompt += "USER: {} ASSISTANT: ".format(query)
        return prompt

    def generate(self, q, history):
        prompt = self.format_prompt(q, history)
        print(prompt)
        
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False)
        device = torch.device("cuda")
        input_ids = input_ids["input_ids"].to(device)

        with torch.no_grad():
            outputs= self.model.generate(input_ids=input_ids,
                    generation_config= self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
            )

        s = outputs.sequences[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(s, skip_special_tokens=True, spaces_between_special_tokens=False)
        print(response)
        return response