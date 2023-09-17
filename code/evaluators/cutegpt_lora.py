from evaluators.evaluator import Evaluator
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from peft import PeftModel
import torch

class Cutegpt_lora_Evaluator(Evaluator):
    def __init__(self, model_path, LORA_WEIGHTS = ''):
        super(Cutegpt_lora_Evaluator,self).__init__(model_path)

        self.tokenizer = LlamaTokenizer.from_pretrained(LORA_WEIGHTS)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        self.model = PeftModel.from_pretrained(self.model, LORA_WEIGHTS).to(torch.float16)
        
        self.device = torch.device("cuda")
        self.overall_instruction = "你是复旦大学知识工场实验室训练出来的语言模型CuteGPT。给定任务描述，请给出对应请求的回答。\n"
        self.generation_config = GenerationConfig(
            top_p=0.8,
            top_k=50,
            eos_token_id = self.tokenizer.convert_tokens_to_ids('<s>'),
            repetition_penalty=1.1,
            max_new_tokens = 2048,
            early_stopping = True
        )
    
    def format_prompt(self, query, history, memory_limit = 10, input=None):
        prompt = self.overall_instruction
        for i, (old_query, response) in enumerate(history[-memory_limit:]):
            prompt += "问：{}\n答：\n{}\n".format(old_query, response)
        prompt += "问：{}\n答：\n".format(query)
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
                    min_length = input_ids.shape[1] + 1
            )

        s = outputs.sequences[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(s)
        response = response.strip()
        response = response.replace("<end>", "").replace("<s>", "").replace("</s>", "")
        print(response)
        return response