from evaluators.evaluator import Evaluator
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import pdb


class Baize_Evaluator(Evaluator):
    def __init__(self, model_path, LORA_WEIGHTS=''):
        super(Baize_Evaluator, self)
        torch.cuda.empty_cache()

        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            add_eos_token=True
        )
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self.tokenizer.pad_token_id = 0
        self.max_length = 10000
        self.max_context_length_tokens = 4096
        self.max_length_tokens = 10000
        self.temperature = 1.0
        self.top_p = 1.0
        self.top_k = 25

    def format_prompt(self, query, history, input=None):
        prompt = "The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n[|Human|]Hello!\n[|AI|]Hi!"
        for i, (old_query, response) in enumerate(history):
            prompt += "\n[|Human|]{}\n[|AI|]{}".format(old_query, response)
        prompt += "\n[|Human|]{}\n[|AI|]".format(query)
        return prompt

    def is_stop_word_or_prefix(self, s: str, stop_words: list) -> bool:
        for stop_word in stop_words:
            if s.endswith(stop_word):
                return True
            for i in range(1, len(stop_word)):
                if s.endswith(stop_word[:i]):
                    return True
        return False

    def sample_decode(
        self, 
        input_ids: torch.Tensor,
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        stop_words: list,
        max_length: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 25,
    ):
        generated_tokens = []
        past_key_values = None
        current_length = 1
        for i in range(max_length):
            with torch.no_grad():
                if past_key_values is None:
                    outputs = model(input_ids)
                else:
                    outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
            # apply temperature
            logits /= temperature

            probs = torch.softmax(logits, dim=-1)
            # apply top_p
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0

            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)

            input_ids = torch.cat((input_ids, next_token), dim=-1)

            generated_tokens.append(next_token[0].item())
            text = tokenizer.decode(generated_tokens)

            yield text
            if any([x in text for x in stop_words]):
                return

    def generate(self, query, history):
        prompt = self.format_prompt(query, history)
        print(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False, add_special_tokens=False, max_length=self.max_length)
        device = torch.device("cuda")
        input_ids = inputs["input_ids"].to(device)

        for x in self.sample_decode(
            input_ids,
            self.model,
            self.tokenizer,
            stop_words=["[|Human|]", "[|AI|]"],
            max_length=self.max_length_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        ):
            if self.is_stop_word_or_prefix(x, ["[|Human|]", "[|AI|]"]) is False:
                if "[|Human|]" in x:
                    x = x[: x.index("[|Human|]")].strip()
                if "[|AI|]" in x:
                    x = x[: x.index("[|AI|]")].strip()
                x = x.strip(" ")
            else:
                print(x)
                return x
