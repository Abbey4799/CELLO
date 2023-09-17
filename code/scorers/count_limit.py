from scorers.scorer import Scorer
import numpy as np
import re
import nltk
import string


def count_chinese_sentences(text):
    pattern = r'[^。！？；\n]+[。！？；\n]'
    sentences = re.findall(pattern, text)
    return len(sentences)


def count_valid_words(text):
    valid_characters = string.ascii_letters + string.digits 
    words = text.split()  

    valid_word_count = 0
    for word in words:
        cleaned_word = ''.join(char for char in word if char in valid_characters)
        if cleaned_word: 
            valid_word_count += 1

    return valid_word_count


class Count_Limit_Scorer(Scorer):
    def __init__(self):
        super(Scorer, self).__init__()

    def get_final_score(self, ans, histories, count_limits, lang):
        '''
        Input:
        - ans (str): The answer to be scored.
        - histories (list): A list of historical answers.
        - count_limits (list of dictionaries): Criteria for scoring the answer regarding *Count Limit*
        - lang (str): The language of the answer (e.g., 'ch' for Chinese, 'eg' for English).

        Output:
        - final_score (float): The final score for the answer.

        Function:
        Calculate the final score for the answer based on specified criteria.
        '''
        word_count_score = np.nan
        sample_count_score = np.nan
        sentence_count_score = np.nan
        revise_score = np.nan

        if not isinstance(ans, str):
            word_count_score = 0
            sample_count_score = 0
            sentence_count_score = 0
            revise_score = 0
        else:
            word_count_score = self.word_count_score(ans, count_limits, lang)
            sample_count_score = self.sample_count_score(ans, count_limits, lang)
            sentence_count_score = self.sentence_count_score(ans, count_limits, lang)
            if len(histories) > 0:
                if not isinstance(histories[-1][1], str):
                    revise_score = 0
                else:
                    revise_score = self.revise_score((histories[-1][1], ans), count_limits, lang)

        return np.nanmean([word_count_score, sample_count_score, sentence_count_score, revise_score])

    def word_count_score(self, ans, criteria, language):
        mode = ''
        limit = ''
        for criterion in criteria:
            if criterion['criterion'] in ["word-max", "word-min", "word-exact", "word-min-max"]:
                mode = criterion['criterion']
                limit = criterion['limit']
                break
            elif criterion['criterion'] == 'NULL':
                return np.nan

        if mode == '':
            return np.nan

        le = 0
        score = 1
        if language == 'ch':
            le = len(re.findall('[\u4e00-\u9fa5]', ans))
        elif language == 'eg':
            le = count_valid_words(ans)

        if mode == "word-max":
            if le > limit:
                score = max(1 - abs(limit - le) / le, 0)
            else:
                score = 1
        if mode == "word-min":
            if le < limit:
                score = max( le / limit, 0)
            else:
                score = 1
        return score

    def sample_count_score(self, ans, criteria, language):
        mode = ''
        limit = ''
        
        score_list = []
        for criterion in criteria:
            # markdown table can be categoried into sample-number
            if criterion['criterion'] in ["sample-table", "sample-min", "sample-number"]:
                mode = criterion['criterion']
                limit = criterion['limit']
            elif criterion['criterion'] == 'NULL':
                return np.nan

            if mode == '':
                return np.nan

            if mode == 'sample-min':
                score = 0
                for ll in limit:
                    if ans.count(ll[0]) >= ll[1]:
                        score += 1
                score = score / len(limit)
            elif mode == 'sample-table':
                score = 0
                try:
                    be = ans.index('|')
                    en = ans.rfind('|')
                    ans = ans[be:en+1]
                    if ans.count('\n') == limit + 1:
                        score = 1
                except ValueError:
                    score = 0
            elif mode == "sample-number":                
                score = 0
                result = re.findall(r'\d+', ans)
                result = list(set(result))
                tmp_score = 0
                for r in result:
                    if int(r) == limit:
                        tmp_score = 1
                    if int(r) > limit:
                        tmp_score = 0
                        break
                score = tmp_score
            
            score_list.append(score)

        return np.max(score_list)

    def sentence_count_score(self, ans, criteria, language):
        mode = ''
        limit = ''
        for criterion in criteria:
            if criterion['criterion'] in ["sentence-max", "sentence-min", "sentence-min-max", "sentence-exact"]:
                mode = criterion['criterion']
                limit = criterion['limit']
                break
            elif criterion['criterion'] == 'NULL':
                return np.nan

        if mode == '':
            return np.nan

        mode = mode
        score = 0
        le = 0
        if language == 'ch':
            le = count_chinese_sentences(ans)
        elif language == 'eg':
            le = len(nltk.sent_tokenize(ans))

        if mode == "sentence-max":
            if le > limit:
                score = max(1 - abs(limit - le) / le, 0)
            else:
                score = 1
            score = np.nan
        if mode == "sentence-min":
            if le < limit:
                score = max(le / limit, 0)
            else:
                score = 1
        if mode == "sentence-min-max" and limit[0] <= le <= limit[1]:
            score = 1
        if mode == "sentence-exact" and le == limit:
            score = 1

        return score

    def revise_score(self, ans_pairs, criteria, language):
        mode = ''
        limit = ''
        for criterion in criteria:
            if criterion['criterion'] in ["revise"]:
                mode = criterion['criterion']
                limit = criterion['limit']
                break
            elif criterion['criterion'] == 'NULL':
                return np.nan

        if mode == '':
            return np.nan

        assert (limit in ["longer", "shorter"])

        prev_ans, now_ans = ans_pairs
        score = 0
        if limit == "longer" and len(now_ans) >= len(prev_ans):
            score = 1
        if limit == "shorter" and len(now_ans) <= len(prev_ans):
            score = 1

        return score