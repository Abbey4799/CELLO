from scorers.scorer import Scorer
import numpy as np
import xml.etree.ElementTree as ET
from nltk.translate.bleu_score import corpus_bleu
import yaml #pip install PyYAML
import re
import pdb

class Input_dependent_Scorer(Scorer):
    def __init__(self):
        super(Scorer,self).__init__()

    def get_final_score(self, ans, histories, input_text, inputs):
        '''
        Input:
        - ans (str): The answer to be scored.
        - formats (list of dictionaries): Criteria for scoring the answer.

        Output:
        - final_score (float): The final score for the answer regarding *Input-dependent Query*

        Function:
        Calculate the final score for the answer based on specified criteria.
        '''
        input_score = np.nan
        
        if not isinstance(ans, str):
            input_score = 0
        else:
            input_score = self.keywords_score(ans, inputs, input_text)

        return input_score
    
    def get_BLEUScore(self, cand, gt):
        bleu_score = 0
        bleu_score = corpus_bleu([[gt]], [cand],weights=(0.25, 0.25, 0.25, 0.25))
        return bleu_score
    
    def keywords_score(self, ans, criteria, input_text):
        keywords = ''
        for criterion in criteria:
            if criterion['criterion'] in ["keywords"]:
                keywords = criterion['limit']
                break
            elif criterion['criterion'] == 'NULL':
                return np.nan
        
        assert (keywords != '' and not isinstance(keywords, float) and  len(keywords) != 0)
        
        cnt = 0 
        for keyword in keywords:
            if keyword in ans:
                cnt += 1
        score = cnt / len(keywords)

        if not isinstance(input_text, str) or input_text == "NULL":
            pass
        else:
            bleu_score = self.get_BLEUScore(ans, input_text)
            score = (1 - bleu_score) * score
        return score