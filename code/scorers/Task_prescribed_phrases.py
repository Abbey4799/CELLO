from scorers.scorer import Scorer
import numpy as np

class Task_phrases_Scorer(Scorer):
    def __init__(self):
        super(Scorer,self).__init__()


    def get_final_score(self, ans, histories, tasks):
        '''
        Input:
        - ans (str): The answer to be scored.
        - formats (list of dictionaries): Criteria for scoring the answer.

        Output:
        - final_score (float): The final score for the answer regarding *Task-prescribed Query*

        Function:
        Calculate the final score for the answer based on specified criteria.
        '''
        task_score = np.nan
        if not isinstance(ans, str):
            task_score = 0
        else:
            task_score = self.keywords_score(ans, tasks)
        
        return task_score
    
    def keywords_score(self, ans, criteria):
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