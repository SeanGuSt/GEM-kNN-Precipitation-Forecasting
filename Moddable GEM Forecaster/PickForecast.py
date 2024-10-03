import numpy as np
def neighborWeight(self):
    suffix = "neighbor"
    K = 1/np.arange(1,self.k+1)
    K = K/K.sum()
    return K, suffix
def forecast(self, dateIndex_analog, amin_b, amax_b, dist_ind, castWeight):
    suffix = "kNN"
    kexp = np.zeros([amax_b, self.k])
    exp = np.zeros(amax_b)
    ymd_mat = []
    for a in range(amax_b):
        ymd_row = []
        for i in range(self.k):
            kexp[a, i] = self.result(dateIndex_analog[dist_ind[a, i]])
            ymd_row.append(f'{self.dates[dateIndex_analog[dist_ind[a, i]], 2]}/{self.dates[dateIndex_analog[dist_ind[a, i]], 1]}/{self.dates[dateIndex_analog[dist_ind[a, i]], 0]}')
            exp[a] += kexp[a, i]*castWeight[i]
        ymd_mat.append(ymd_row)
    return kexp, exp, ymd_mat, suffix
def result(self, index):
    return self.data[index + self.fvec, self.precipCol].mean()