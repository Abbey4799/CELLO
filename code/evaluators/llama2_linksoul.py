from transformers import AutoTokenizer
import transformers
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, TextStreamer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator
import pdb


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class LLama2_LinkSoul(Evaluator):
    def __init__(self, model_path):
        super(LLama2_LinkSoul, self).__init__(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.overall_instruction = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
        
    def format_prompt(self, query, history, input=None):
        prompt = ""
        if len(history) == 0:
            prompt += f"<s>{B_INST} {B_SYS} {self.overall_instruction} {E_SYS} {query.strip()} {E_INST} "
        else:
            prompt += f"<s>{B_INST} {B_SYS} {self.overall_instruction} {E_SYS}"
            for i, (old_query, response) in enumerate(history):
                if i>0:
                    prompt += f"<s>{B_INST}"
                prompt += f"{old_query.strip()} {E_INST} {response.strip()}</s>"
            prompt += f"<s>{B_INST} {query.strip()} {E_INST}"
        print(prompt)
        return prompt
    
    def generate(self, q, history):
        prompt = self.format_prompt(q, history)

        generate_ids = self.model.generate(self.tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=self.streamer)
        generate_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        response = generate_text.split("[/INST]")[-1]
        print(f"response:{response}")
        return response
    