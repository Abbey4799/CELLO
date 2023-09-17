from transformers import AutoTokenizer
import transformers
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator
import pdb


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class LLama2_chat_Evaluator(Evaluator):
    def __init__(self, model_path):
        super(LLama2_chat_Evaluator, self).__init__(model_path)
        self.model = "meta-llama/Llama-2-13b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.device = torch.device("cuda")
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
        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=10000,
        )

        output = ""
        for seq in sequences:
            try:
                output += seq["generated_text"]
            except:
                output += seq[0]['generated_text']

        try:
            response = output.replace(prompt,"")
        except:
            for i in prompt:
                response = output.replace(i,"")
        return response
    