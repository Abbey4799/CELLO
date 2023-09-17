from scorers.answer_format import Answer_format_Scorer
from scorers.Input_dependent_query import Input_dependent_Scorer
from scorers.Task_prescribed_phrases import Task_phrases_Scorer
from scorers.count_limit import Count_Limit_Scorer
import argparse
import utils
import glob
import numpy as np
import pandas as pd
import os
import re
import copy
import pdb
import warnings


Format_evaluator = Answer_format_Scorer()
Input_evaluator = Input_dependent_Scorer()
Task_evaluator = Task_phrases_Scorer()
Count_evaluator = Count_Limit_Scorer()



MODEL_mapping = {
    # Chinese-oriented Models (Continue Pretraining)
    'Ch_cc': ['belle', 'baize7b', 'baize13b', 'alpaca7b', 'alpaca13b', 'alpaca33b','cutegpt_lora', 'llama2_linksoul', 'llama2_linly', 'llama2_openbuddy','llama2_flagalpha'],
    # Chinese-oriented Models (From Scratch)
    'Ch_fs': ['moss', 'batgpt15b','chatglm', 'chatglm2','chatglm2-32k', 'baichuan13b_chat','qwenlm', 'internlm7b'],
    # English-oriented Models
    'Eg': ['llama2-7b-chat','llama2-13b-chat', 'llama2-70b-chat', 'vicuna7b', 'vicuna13b','vicuna7b_16k', 'vicuna13b_16k', 'vicuna33b', 'wizardlm13b','longchat7b', 'longchat7b_32k','longchat13b','openchat13b','chatgpt','gpt4']
}


warnings.filterwarnings("ignore")

def get_score_each_task(res_list, labeled_data, interset_sample_ids):
    score_list = []
    for sample in res_list:
        if sample['Instruction_id'] not in interset_sample_ids:
            continue
        
        try:
            matched_sample = list(filter(lambda x: x['Instruction_id'] == sample['Instruction_id'], labeled_data))[0]
        except:
            continue

        format_score = Format_evaluator.get_final_score(sample['model_answer'], sample['histories'], matched_sample['Answer_format'])
        input_score = Input_evaluator.get_final_score(sample['model_answer'], sample['histories'], sample['input_text'], matched_sample['Input_depdent_query'])
        task_score = Task_evaluator.get_final_score(sample['model_answer'], sample['histories'], matched_sample['Task_prescribed_phrases'])
        count_limit_score = Count_evaluator.get_final_score(sample['model_answer'], sample['histories'], matched_sample['Count_limit'], matched_sample["lang"])

        score_list.append([format_score, input_score, task_score, count_limit_score])

    if len(score_list) == 0:
        return np.nan
    return np.nanmean(np.array(score_list), axis = 0)


def get_interset(res_path, model_names, data_ins_ids, min_num = 522):
    '''
        Valid Samples:
        1. All models have answers
        2. In the data/ folder
    '''
    set_list = []
    avail_model_names = []
    fail_model_names = []
    for model_name in model_names:
        if model_name == 'test':
            continue

        # 1. All models have answers
        paths = glob.glob(res_path + model_name + '/*.json')
        _ = [list(filter(lambda x:  not isinstance(x['model_answer'], float) and not isinstance(x['model_answer'], type(None)), utils.readjson(path))) for path in paths]
        ids = [[sample['Instruction_id'] for sample in samples] for samples in _ ]
        avail_ins_ids = list(set([element for lis in ids for element in lis]))

        # 2. In the data/ folder
        avail_ins_ids = set(filter(lambda x: x in data_ins_ids, avail_ins_ids))

        # print(model_name)
        # print(len(avail_ins_ids))
        # print(len(data_ins_ids))

        # Only consider models whose sample size is greater than *min_num* in their results 
        # to prevent the final valid sample size from being too small.
        if len(avail_ins_ids) > min_num:
            avail_model_names.append(model_name)
            set_list.append(avail_ins_ids)
        else:
            fail_model_names.append(model_name)

    return set.intersection(*set_list), avail_model_names, fail_model_names

def merge_tasks(xml_df_tasks, original_cols, new_col_name):
    '''
    Collating the columns of the final csv file
    '''
    xml_df_tasks[new_col_name] = xml_df_tasks[original_cols].mean(axis=1)
    xml_df_tasks.drop(original_cols, axis=1, inplace=True)
    last_column_index = len(xml_df_tasks.columns) - 2
    xml_df_tasks.insert(last_column_index, new_col_name, xml_df_tasks.pop(new_col_name))
    return xml_df_tasks


def model_formal_name(df):
    name_mapping = {
        'llama2-7b-chat': 'Llama2-chat-7B',
        'llama2-70b-chat': 'Llama2-chat-70B',
        'llama2-13b-chat': 'Llama2-chat-13B',
        'alpaca': 'Chinese-Alpaca-V1-',
        'baize': 'Baize-V2-',
        'baichuan13b_chat': 'Baichuan-chat',
        'batgpt15b': 'BatGPT-sirius',
        'belle': 'BELLE',
        'chatglm': 'ChatGLM',
        'chatgpt': 'GPT-3.5-turbo',
        'cutegpt_lora': 'CuteGPT',
        'gpt4': 'GPT-4',
        'internlm7b': 'InternLM',
        'llama2_': 'Llama2-',
        'flagalpha':'FlagAlpha',
        'linksoul': 'LinkSoul',
        'linly': 'Linly',
        'openbuddy': 'OpenBuddy',
        'longchat7b_32k': 'LongChat-V1.5-7B',
        'longchat7b': 'LongChat-V1-7B',
        'longchat13b': 'LongChat-V1-13B',
        'moss':'MOSS',
        'openchat13b': 'OpenChat-V3.2',
        'qwenlm': 'Qwen',
        'vicuna7b_16k': 'Vicuna-V1.5-7B',
        'vicuna13b_16k': 'Vicuna-V1.5-13B',
        'vicuna7b': 'Vicuna-V1.3-7B',
        'vicuna13b': 'Vicuna-V1.3-13B',
        'vicuna33b': 'Vicuna-V1.3-33B',
        'wizardlm13b': 'WizardLM'
    }
    for origin_name in name_mapping:
        df['model_name'] = df['model_name'].str.replace(origin_name, name_mapping[origin_name])
    return df

def main(args):
    paths = glob.glob(args.result_path + '*/')
    model_names = [utils.get_name(path, re.compile(args.result_path + "(.*?)/"), mode=1) for path in paths]
    for forbid_name in ['final', 'all', 'test']:
        if forbid_name in model_names:
            model_names.remove(forbid_name)
    print(model_names)

    # samples in the data/ folder
    paths = glob.glob(args.labeled_data_path + '*.json')
    ids = [[sample['Instruction_id'] for sample in samples] for samples in [utils.readjson(path) for path in paths]]
    data_ins_ids = list(set([element for lis in ids for element in lis]))

    # samples in the results/ folder
    new_model_names = copy.copy(model_names)
    for model_name in model_names:
        _ = glob.glob(os.path.join(args.result_path + model_name + '/', "*.json"))
        if len(_) < len(paths):
            new_model_names.remove(model_name)
    model_names = new_model_names
    print(model_names)
    
    # get valid samples id
    id_list, model_names, fail_model_names = get_interset(args.result_path, model_names, data_ins_ids)
    print(len(id_list))
    
    models_by_constraints = []
    models_by_tasks = []
    constraints_names =  ['model_name', 'format_score', 'input_score', 'task_score', 'count_limit_score', 'Average']
    tasks_names = ['model_name']
    
    for category in MODEL_mapping:
        models_by_constraints = []
        models_by_tasks = []
        tasks_names = ['model_name']
        for model_name in model_names:
            # get scores for each model
            if model_name not in MODEL_mapping[category]:
                continue
            print(f'{model_name} processing...')

            list_res = []
            all_scores = []
            paths = glob.glob(os.path.join(args.result_path + model_name + '/', "*.json"))
            
            for path in paths:
                res_list = utils.readjson(path)
                task_name = utils.get_name(path, r"(?<=\/res_).*?(?=\.)")
                if 'test' == task_name or len(res_list) == 0:
                    continue
                labeled_data = utils.readjson(args.labeled_data_path + task_name + '.json')
                scores = get_score_each_task(res_list, labeled_data, id_list)

                if isinstance(scores,float):
                    continue
                list_res.append([task_name] + list(scores) +  [np.nanmean(scores)])
                all_scores.append(list(scores) + [np.nanmean(scores)])

            list_res.append(['Average'] + list(np.nanmean(np.array(all_scores), axis = 0)))
            column_name = ['task_name', 'format_score', 'input_score', 'task_score', 'count_limit_score', 'Average']
            xml_df = pd.DataFrame(list_res, columns=column_name)
            models_by_constraints.append([model_name] + list_res[-1][1:])
            models_by_tasks.append([model_name] + list(xml_df['Average']))
        tasks_names.extend(list(xml_df['task_name']))

        xml_df_constraints = pd.DataFrame(models_by_constraints, columns=constraints_names)
        xml_df_tasks = pd.DataFrame(models_by_tasks, columns=tasks_names)


        # categorize model scores (csv files) by tasks or constraints
        xml_df_tasks = merge_tasks(xml_df_tasks, ['Structure_hard',  'Structure_simple', 'Structure_None'], 'Structure')
        xml_df_tasks = xml_df_tasks[['model_name','extraction','Planning', 'meta_prompt', 'brainstorming_single_round','writing_single_round','keywords_extraction','closed_qa','Summarization', 'Structure','brainstorming_multi_rounds', 'writing_multi_rounds','Average']]
        average_values = xml_df_tasks[['extraction','Planning', 'meta_prompt', 'brainstorming_single_round','writing_single_round']].mean(axis=1)
        xml_df_tasks.insert(loc=xml_df_tasks.columns.get_loc('keywords_extraction'), column='Ave_complex_ins', value=average_values)
        average_values = xml_df_tasks[['keywords_extraction','closed_qa','Summarization', 'Structure','brainstorming_multi_rounds', 'writing_multi_rounds']].mean(axis=1)
        xml_df_tasks.insert(loc=xml_df_tasks.columns.get_loc('Average'), column='Ave_complex_input', value=average_values)

        xml_df_tasks['Average'] = (xml_df_tasks['Ave_complex_ins'] + xml_df_tasks['Ave_complex_input']) / 2
        xml_df_constraints['Average'] = xml_df_tasks['Average']

        xml_df_constraints = xml_df_constraints.round(3)
        xml_df_tasks = xml_df_tasks.round(3)
    
        xml_df_constraints = xml_df_constraints.sort_values(by=xml_df.columns[-1])
        xml_df_tasks = xml_df_tasks.sort_values(by=xml_df.columns[-1])

        xml_df_tasks = model_formal_name(xml_df_tasks)
        xml_df_constraints = model_formal_name(xml_df_constraints)

        print(f'===={category}====')
        print(xml_df_constraints)
        print(xml_df_tasks)

        xml_df_constraints.to_csv(args.saved_score_path + category + '_scores_by_constraints.csv', index=None, encoding='utf_8_sig')
        xml_df_tasks.to_csv(args.saved_score_path + category + '_scores_by_tasks.csv', index=None, encoding='utf_8_sig')

    # Models with sample sizes smaller than *min_num* are therefore discarded.
    print('you should check...')
    print(list(filter(lambda x: x in [element for lis in list(MODEL_mapping.values()) for element in lis],fail_model_names)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", type=str, default='results/')
    parser.add_argument("--labeled_data_path", type=str, default='data/')
    parser.add_argument("--saved_score_path", type=str, default='scores/')
    args = parser.parse_args()

    main(args)
