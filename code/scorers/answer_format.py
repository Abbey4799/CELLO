from scorers.scorer import Scorer
import xml.etree.ElementTree as ET
import yaml #pip install PyYAML
import numpy as np
import re
import pdb

class Answer_format_Scorer(Scorer):
    def __init__(self):
        super(Scorer,self).__init__()

    def get_final_score(self, ans,  histories, formats):
        '''
        Input:
        - ans (str): The answer to be scored.
        - formats (list of dictionaries): Criteria for scoring the answer.

        Output:
        - final_score (float): The final score for the answer regarding *Answer Format*

        Function:
        Calculate the final score for the answer based on specified criteria.
        '''
        resolve_score = np.nan
        keywords_score = np.nan
    
        if not isinstance(ans, str):
            resolve_score  = 0
            keywords_score = 0
        else:
            resolve_score = self.resolve_score(ans, formats)
            keywords_score = self.keywords_score(ans, formats)

        return np.nanmean([resolve_score, keywords_score])

    def resolve_score(self, ans, criteria):
        '''
        Input:
        - ans (str): The answer to be scored.
        - criteria (list of dictionaries): Criteria for scoring the answer.

        Output:
        - score (float): The score for the resolve criterion.

        Function:
        One point is awarded if the model answer can be directly parsed, 0 points otherwise
        '''
        resolve = ''
        for criterion in criteria:
            if criterion['criterion'] in ["resolve"]:
                resolve = criterion['limit']
                break
            elif criterion['criterion'] == 'NULL':
                return np.nan
        
        if resolve == '':
            return np.nan

        score = 0
        try:
            if resolve == 'json' or resolve == 'dict':
                temp = eval(ans)
                assert(isinstance(temp,dict))
                score = 1
            elif resolve == 'list':
                temp = eval(ans)
                assert(isinstance(temp,list))
                score = 1
            elif resolve == 'tuple':
                temp = eval(ans)
                assert(isinstance(temp,tuple))
                score = 1
            elif resolve == 'xml':
                ET.fromstring(ans)
                score = 1
            elif resolve == 'yaml':
                yaml.safe_load(ans)
                score = 1
        except:
            score = 0
        return score

    
    def keywords_score(self, ans, criteria):
        '''
        Input:
        - ans (str): The answer to be scored.
        - criteria (list of dictionaries): Criteria for scoring the answer.

        Output:
        - score (float): The score for the keywords criterion.

        Function:
        The higher the keyword coverage, the higher the final score
        '''
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

        return cnt / len(keywords)