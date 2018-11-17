import numpy as np


class RougeL(object):
    def __init__(self, gamma=1.2):
        self.gamma = gamma  
        self.inst_scores = []

    def lcs(self, string: str, sub: str) -> int:
       

        str_length = len(string)
        sub_length = len(sub)

        lengths = np.zeros(((str_length + 1), (sub_length + 1)), dtype=np.int)
        for i in range(1, str_length + 1):
            for j in range(1, sub_length + 1):
                if string[i - 1] == sub[j - 1]:
                    lengths[i][j] = lengths[i - 1][j - 1] + 1
                else:
                    lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
        return lengths[str_length, sub_length]

    def add_inst(self, cand: str, ref: str):
        

        basic_lcs = self.lcs(cand, ref)
        p_denom = len(cand)
        r_denom = len(ref)
        prec = basic_lcs / p_denom if p_denom > 0. else 0.
        rec = basic_lcs / r_denom if r_denom > 0. else 0.
        if prec != 0 and rec != 0:
            score = ((1 + self.gamma ** 2) * prec * rec) / \
                float(rec + self.gamma**2 * prec)
        else:
            score = 0
        self.inst_scores.append(score)

    def get_score(self) -> float:
        
        return 1. * sum(self.inst_scores) / len(self.inst_scores)


if __name__ == '__main__':
    print('Hello World')
