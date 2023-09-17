from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from evaluators.evaluator import Evaluator
import pdb

class LLama2_OpenBuddy_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(LLama2_OpenBuddy_Evaluator,self).__init__(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            trust_remote_code=True, 
            torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model.eval()
        self.overall_instruction = "You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User. \
        Always answer as helpfully and logically as possible, while being safe. Your answers should not include any harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. \
        If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. \
        You like to use emojis. You can speak fluently in many languages, for example: English, Chinese. \
        You cannot access the internet, but you have vast knowledge, cutoff: 2021-09. \
        You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI. \
        "

    def format_prompt(self, query, history, memory_limit=10, input=None):
        prompt = self.overall_instruction
        for i, (old_query, response) in enumerate(history):
            prompt += f'\n\nUser: {old_query}\nAssistant: {response}'

        prompt += f"\n\nUser: {query}\nAssistant:"
        return prompt
    
    def generate(self, q, history):
        prompt = self.format_prompt(q,history)
        print(f"prompt:{prompt}")
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids, 
                max_new_tokens=4096, 
                eos_token_id=self.tokenizer.eos_token_id)
            
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if 'Assistant:' in response:
            response = response.split("Assistant:")[-1]
        else:
            response = response.replace(prompt,"")
        print(f"response:{response}")
        return response

