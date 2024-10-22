import numpy as np
import copy
from GEMObj import GEMObj
def uniformWeight(self: GEMObj, a: int):
    suffix = "Uniform"
    return np.ones([a, self.num_var]), suffix
def randomWeight(self: GEMObj, a: int):
    suffix = "DoNotUse"
    return np.random([a, self.num_var]), suffix
def eucliDist(self: GEMObj, data_normalized: np.ndarray, dateIndex_target: np.ndarray, dateIndex_analog: np.ndarray, amin_b: int, amax_b: int, b: int, distWeight: np.ndarray):
    #self: the object holding all universal data
    #data_normalized: The "feature vectors" are never actually built, per se. 
    #Instead, features are pulled as needed from data_normalized when finding distance between two feature vectors.
    suffix = "Euclidean"
    dist = np.empty([amax_b, dateIndex_analog.shape[0]])
    ed = np.zeros([dateIndex_analog.shape[0]])
    for a in range(amax_b):
        #Move back (a, b) days to get that particular feature.
        dit = dateIndex_target - a*b
        dia = dateIndex_analog - a*b
        ed += np.dot(np.power(data_normalized[dit,:] - data_normalized[dia, :], 2), distWeight[a, :])
        dist[a, :] = copy.deepcopy(np.sqrt(ed/((a+1)*self.num_var)))
    return dist, suffix