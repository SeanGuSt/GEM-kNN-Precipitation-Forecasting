import numpy as np
from ResultsHandler import getDataFrameHeader
def scoreFunction(self, bigDicti):
    exp = bigDicti["exp"]
    obs = np.zeros(self.num_tarDays)
    con = np.zeros(self.num_tarDays)
    CO_error = np.zeros(self.num_tarDays)
    GEMO_error = np.zeros(exp.shape)
    GEMC_compare = np.zeros(exp.shape)
    returnEarly = True
    conLen = 30
    for day in range(self.num_tarDays):
        dateIndex = self.getIndices(self.y[day], self.m[day], self.d[day], 0, 0, returnEarly)#Moddable Function
        obs[day] = self.result(dateIndex[-1])
        if len(dateIndex) > conLen:
            for i in range(1, 1 + conLen):
                con[day] += self.result(dateIndex[-1-i])
            con[day] /= conLen
        else:
            con[day] = -1
        CO_error[day] = np.abs(con[day] - obs[day])
        GEMO_error[day, :] = np.abs(exp[day, :] - obs[day])
        GEMC_compare[day, :] = GEMO_error[day, :] <= CO_error[day]
    bigDicti["GEMC_compare"] = GEMC_compare    
    return bigDicti
def reassignABPairs(self, bigDicti):
    NUM_MONTHS = 12
    header = getDataFrameHeader(self)
    GEMC_compare = bigDicti["GEMC_compare"]
    num_ab = GEMC_compare.shape[1]
    chosen_pairs_real = ["" for _ in range(NUM_MONTHS)]
    chosen_pairs = np.zeros(NUM_MONTHS, dtype = int)
    final_scores = np.zeros((NUM_MONTHS, num_ab))
    for i in range(NUM_MONTHS):
        if i+1 not in self.target_months:
            chosen_pairs[i] = -1
            chosen_pairs_real[i] = "-1"
        else:
            final_scores[i, :] = [np.sum(GEMC_compare[self.m == i+1, j]) for j in range(num_ab)]
            chosen_pairs[i] = np.argmax(final_scores[i,:])
            chosen_pairs_real[i] = header[chosen_pairs[i]]
    bigDicti["chosen_pairs"] = chosen_pairs
    bigDicti["chosen_pairs_real"] = chosen_pairs_real
    return bigDicti