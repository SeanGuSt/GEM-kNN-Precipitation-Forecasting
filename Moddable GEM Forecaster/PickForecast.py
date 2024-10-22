import numpy as np
from GEMObj import GEMObj
def neighborWeight(self: GEMObj, dist_ind: np.ndarray):
    suffix = "neighbor"
    k = self.getk(dist_ind)
    K = 1/np.arange(1, k+1)
    K = K/K.sum()
    return K, suffix
def forecast(self: GEMObj, dateIndex_analog: int, amin_b: int, amax_b: int, dist_ind: np.ndarray, castWeight: np.ndarray):
    suffix = "kNN"
    k = castWeight.shape[0]#This second k is created just in case some of the target dates have fewer analogs than self.k
    kexp = -np.ones([amax_b, self.k])
    exp = np.zeros(amax_b)
    ymd_mat = [["" for _ in range(self.k)] for _ in range(amax_b)]
    for a in range(amin_b-1, amax_b):
        for i in range(k):
            kexp[a, i] = self.result(dateIndex_analog[dist_ind[a, i]])
            ymd_mat[a][i] = (f'{self.dates[dateIndex_analog[dist_ind[a, i]], 2]}/{self.dates[dateIndex_analog[dist_ind[a, i]], 1]}/{self.dates[dateIndex_analog[dist_ind[a, i]], 0]}')
            exp[a] += kexp[a, i]*castWeight[i]
    return kexp, exp, ymd_mat, suffix
def result(self: GEMObj, index: int):
    return self.data[index + self.fvec, self.precipCol].mean()
def getk(self: GEMObj, dist_ind: np.ndarray):
    if self.k == -1:
        k = np.floor(np.sqrt(dist_ind.shape[1])).astype(int)
    else:
        k = np.minimum(self.k, dist_ind.shape[-1])
    return k