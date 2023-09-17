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


warnings.filterwarnings("ignore")

def get_score_each_task(res_list, labeled_data, interset_sample_ids):
    score_list = []
    matched_ins = []
    for sample in res_list:
        # interset_sample_ids = ['5df63325dafc3b787adb65c7108ac41ed71244c847019d35f15021620a83ffe8']
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
        # print(sample['model_answer'])

        # score_list.append(input_score)
        score_list.append(format_score)
        matched_ins.append(sample['Instruction_id'])

    return score_list, matched_ins

# task_name = ''
# task_name = 'closed_qa'
task_name = 'keywords_extraction'
model1 = 'llama2_linksoul'
model2 = 'cutegpt_lora'
# model1 = 'chatgpt'
# model2 = 'gpt4'

res_list_1 = utils.readjson('../results/' + model1 + '/res_' + task_name + '.json')
res_list_2 = utils.readjson('../results/' + model2 + '/res_' + task_name + '.json')
labeled_data = utils.readjson('../data/' + task_name + '.json')

set_list = [set([i['Instruction_id'] for i in res_list_1]),set([i['Instruction_id'] for i in res_list_2])]
interset_sample_ids = set.intersection(*set_list)

# score_list_1 = get_score_each_task(res_list_1[9:12], labeled_data[9:12])
# score_list_2 = get_score_each_task(res_list_2[9:12], labeled_data[9:12])
score_list_1, matched_ins_1 = get_score_each_task(res_list_1, labeled_data, interset_sample_ids)
score_list_2, matched_ins_2 = get_score_each_task(res_list_2, labeled_data, interset_sample_ids)


for i in range(len(matched_ins_1)):
    score1 = score_list_1[i]
    score2 = score_list_2[matched_ins_2.index(matched_ins_1[i])]
    if score1 > score2 and score2 < 1:
    # if score2 > 0.8:
    # if np.nanmean(score_list_2[i]) < 0.8:
        matched_sample = list(filter(lambda x: x['Instruction_id'] == matched_ins_1[i], labeled_data))[0]
        res1 = list(filter(lambda x: x['Instruction_id'] == matched_ins_1[i], res_list_1))[0]
        res2 = list(filter(lambda x: x['Instruction_id'] == matched_ins_1[i], res_list_2))[0]
        print('====')
        print(i)
        print(matched_sample['Instruction'])
        print(matched_sample['Instruction_id'])
        print('#############')
        print(res1['model_answer'])
        print('***')
        for ll in matched_sample['Input_depdent_query'][0]['limit']:
            if ll not in res1['model_answer']:
                print(ll)
        print('#############')
        print(res2['model_answer'])
        print('***')
        for ll in matched_sample['Input_depdent_query'][0]['limit']:
            if ll not in res2['model_answer']:
                print(ll)
        print('#############')
        print(matched_sample['Answer_format'])
        print(matched_sample['Input_depdent_query'][0]['limit'])
        print(matched_sample['Task_prescribed_phrases'])
        print(matched_sample['Count_limit'])

        print()
        print(score_list_1[i])
        print(score_list_2[matched_ins_2.index(matched_ins_1[i])])
        # print(score_list_2[i])
        print()

print(np.mean(score_list_1))
print(np.mean(score_list_2))