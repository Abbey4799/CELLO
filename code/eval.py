from evaluators.baichuan13b_chat import Baichuan13bChat_Evaluator
from evaluators.baize import Baize_Evaluator
from evaluators.cutegpt_lora import Cutegpt_lora_Evaluator
from evaluators.chatglm import ChatGLM_Evaluator
from evaluators.belle import BELLE_Evaluator
from evaluators.moss import MOSS_Evaluator
from evaluators.alpaca import Alpaca_Evaluator
from evaluators.vicuna import Vicuna_Evaluator
from evaluators.vicuna_else import Vicuna_else_Evaluator
from evaluators.longChat import LongChat_Evaluator
from evaluators.llama2_chat import LLama2_chat_Evaluator
from evaluators.llama2_linksoul import LLama2_LinkSoul
from evaluators.llama2_linly import LLama2_Linly_Evaluator
from evaluators.llama2_flagalpha import LLama2_FlagAlpha_Evaluator
from evaluators.llama2_openbuddy import LLama2_OpenBuddy_Evaluator
from evaluators.wizardlm13b import WizardLM13b_Evaluator
from evaluators.openchat import Openchat_Evaluator
from evaluators.internlm7b import InterLM7b_Evaluator
from evaluators.batgpt15b import BatGPT15b_Evaluator
from evaluators.qwenlm import QwenLM_Evaluator
from evaluators.chatgpt import ChatGPT_Evaluator
from evaluators.gpt4 import GPT4_Evaluator
import argparse
import utils
import pdb
import numpy as np
import os
import glob

def main(args):
    if "cutegpt_lora" == args.model_name:
        model_path = "XuYipei/kw-cutegpt-13b-base"
        LORA_WEIGHTS = "Abbey4799/kw-cutegpt-13b-ift-lora"
        evaluator = Cutegpt_lora_Evaluator(model_path, LORA_WEIGHTS = LORA_WEIGHTS)
    elif "baichuan13b_chat" == args.model_name:
        model_path = "/data/lina/Baichuan-13B-Chat"
        evaluator = Baichuan13bChat_Evaluator(model_path)
    elif "baize7b" == args.model_name:
        model_path = "project-baize/baize-v2-7b"
        evaluator = Baize_Evaluator(model_path)
    elif "baize13b" == args.model_name:
        model_path = "project-baize/baize-v2-13b"
        evaluator = Baize_Evaluator(model_path)
    elif "wizardlm13b" == args.model_name:
        model_path = "[Your path to the merged-WizardLM-13B....]"
        evaluator = WizardLM13b_Evaluator(model_path)
    elif "ChineseAlpaca7b" == args.model_name:
        model_path = "minlik/chinese-alpaca-7b-merged"
        evaluator = WizardLM13b_Evaluator(model_path)
    elif "openchat13b" == args.model_name:
        model_path = "openchat/openchat_v3.2"
        evaluator = Openchat_Evaluator(model_path)
    elif "chatglm" == args.model_name:
        model_path = "THUDM/chatglm-6b"
        evaluator = ChatGLM_Evaluator(model_path)
    elif "chatglm2" == args.model_name:
        model_path = "THUDM/chatglm2-6b"
        evaluator = ChatGLM_Evaluator(model_path)
    elif "chatglm2-32k" == args.model_name:
        model_path = "THUDM/chatglm2-6b-32k" 
        evaluator = ChatGLM_Evaluator(model_path)
    elif "belle" == args.model_name:
        model_path = "BelleGroup/BELLE-7B-2M"
        evaluator = BELLE_Evaluator(model_path)
    elif "moss" == args.model_name:
        model_path = "fnlp/moss-moon-003-sft"
        evaluator = MOSS_Evaluator(model_path)
    elif "alpaca7b" == args.model_name:
        model_path = "decapoda-research/llama-7b-hf"
        LORA_WEIGHTS = "ziqingyang/chinese-alpaca-lora-7b"
        evaluator = Alpaca_Evaluator(model_path, LORA_WEIGHTS = LORA_WEIGHTS)
    elif "alpaca13b" == args.model_name:
        model_path = "decapoda-research/llama-13b-hf"
        LORA_WEIGHTS = "ziqingyang/chinese-alpaca-lora-13b"
        evaluator = Alpaca_Evaluator(model_path, LORA_WEIGHTS = LORA_WEIGHTS)
    elif "alpaca33b" == args.model_name:
        model_path = "decapoda-research/llama-30b-hf"
        LORA_WEIGHTS = "ziqingyang/chinese-alpaca-lora-33b"
        evaluator = Alpaca_Evaluator(model_path, LORA_WEIGHTS = LORA_WEIGHTS)
    elif "vicuna7b" == args.model_name:
        model_path = "lmsys/vicuna-7b-v1.3"
        evaluator = Vicuna_Evaluator(model_path)
    elif "vicuna13b" == args.model_name:
        model_path = "lmsys/vicuna-13b-v1.3"
        evaluator = Vicuna_Evaluator(model_path)
    elif "vicuna33b" == args.model_name:
        model_path = "lmsys/vicuna-33b-v1.3"
        evaluator = Vicuna_else_Evaluator(model_path)
    elif "vicuna7b_16k" == args.model_name:
        model_path = "lmsys/vicuna-7b-v1.5-16k"
        evaluator = Vicuna_Evaluator(model_path)
    elif "vicuna13b_16k" == args.model_name:
        model_path = "lmsys/vicuna-13b-v1.5-16k"
        evaluator = Vicuna_else_Evaluator(model_path)
    elif "longchat7b" == args.model_name:
        model_path = "lmsys/longchat-7b-16k"
        evaluator = LongChat_Evaluator(model_path)
    elif "longchat13b" == args.model_name:
        model_path = "lmsys/longchat-13b-16k"
        evaluator = LongChat_Evaluator(model_path)
    elif "longchat7b_32k" == args.model_name:
        model_path = "lmsys/longchat-7b-v1.5-32k"
        evaluator = LongChat_Evaluator(model_path)
    elif "llama2-7b-chat" == args.model_name:
        model_path = "meta-llama/Llama-2-7b-chat-hf"
        evaluator = LLama2_chat_Evaluator(model_path)
    elif "llama2-13b-chat" == args.model_name:
        model_path = "/data/lina/Llama-2-13b-chat-hf"
        evaluator = LLama2_chat_Evaluator(model_path)
    elif "llama2-70b-chat" == args.model_name:
        model_path = "meta-llama/Llama-2-70b-chat-hf"
        evaluator = LLama2_chat_Evaluator(model_path)
    elif "llama2_linksoul" == args.model_name:
        model_path = "LinkSoul/Chinese-Llama-2-7b"
        evaluator = LLama2_LinkSoul(model_path)
    elif "llama2_linly" == args.model_name:
        model_path = "Linly-AI/Chinese-LLaMA-2-7B-hf"
        evaluator = LLama2_Linly_Evaluator(model_path)
    elif "llama2_flagalpha" == args.model_name:
        model_path = "FlagAlpha/Llama2-Chinese-7b-Chat"
        evaluator = LLama2_FlagAlpha_Evaluator(model_path)
    elif "llama2_openbuddy" == args.model_name:
        model_path = 'OpenBuddy/openbuddy-llama2-13b-v8.1-fp16'
        evaluator = LLama2_OpenBuddy_Evaluator(model_path)
    elif "internlm7b" == args.model_name:
        model_path = "internlm/internlm-chat-7b"
        evaluator = InterLM7b_Evaluator(model_path)
    elif "batgpt15b" == args.model_name:
        model_path = "MLP-lab/BatGPT-15B-sirius"
        evaluator = BatGPT15b_Evaluator(model_path)
    elif "qwenlm" == args.model_name:
        model_path = "Qwen/Qwen-7B-Chat"
        evaluator = QwenLM_Evaluator(model_path)
    elif "chatgpt" == args.model_name:
        evaluator = ChatGPT_Evaluator()
    elif "gpt4" == args.model_name:
        evaluator = GPT4_Evaluator()
    else:
        print("Unknown model name")
        return -1
    
    utils.check_folder("results/")
    utils.check_folder("results/{}/".format(args.save_name))

    import glob
    datas = glob.glob("data/*.json")
    if args.category == "all":
        categories = [i.split("/")[-1].split(".")[0] for i in datas]
    else:
        categories = [args.category]

    for category in categories:
        file_path = "data/{}.json".format(category)
        try:
            samples = utils.readjson(file_path)
            print(f"{category} Reading...")
        except:
            print(f"\n{category} Reading Error\n")

        # Check whether the sample already has results
        finished_instruction_id = []
        results = []
        try:
            if os.path.exists("results/{}/res_{}.json".format(args.save_name, category)):
                if not args.overwrite:
                    results = utils.readjson("results/{}/res_{}.json".format(args.save_name, category))
                    # If the answer is NaN, rerun
                    results = list(filter(lambda x: not isinstance(x['model_answer'], float) and not isinstance(x['model_answer'], type(None)), results))
                    finished_instruction_id = [i["Instruction_id"] for i in results]

                    print("results/{}/res_{}.json already exists, only pick the unseen samples".format(args.save_name, category))
                else:
                    print('overwrite results/{}/res_{}.json...'.format(args.save_name, category))
        except:
            pass

        for sample in samples:
            if not args.overwrite and sample['Instruction_id'] in finished_instruction_id:
                print('{} exits, skip!'.format(sample['Instruction_id']))
                continue
        
            print('new_sample!')
            histories = sample['histories']
            round_instruction = sample['Instruction']
            try:
                ans = evaluator.generate(round_instruction, histories)
                histories.append((round_instruction, ans))
            except:
                ans = np.nan
                histories.append((round_instruction, ''))
            sample["model_answer"] = ans

            results.append(sample)
            utils.writejson(results, "results/{}/res_{}.json".format(args.save_name, category))
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    tasks = [i.split('/')[-1].replace('.json','') for i in glob.glob('data/*.json')]

    parser.add_argument("--model_name",type=str)
    parser.add_argument("--save_name",type=str,default="")
    parser.add_argument("--category",type=str, choices=tasks + ['all'], default="all",help="The task name you want to evaluate. ")
    parser.add_argument("--overwrite",action="store_true",help="whether to overwrite the existing results")
    args = parser.parse_args()
    if args.save_name == "":
        args.save_name = args.model_name
    main(args)
