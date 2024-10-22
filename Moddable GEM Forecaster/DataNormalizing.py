import numpy as np
from GEMObj import GEMObj
def standardNormalize(self: GEMObj):
    suffix = "Gaussian"
    data_normalized_original = np.zeros(self.data.shape)
    for col in range(self.num_var):
        data_normalized_original[:, col] = self.data[:,col] - np.average(self.data[:,col])
        data_normalized_original[:, col] /= np.std(self.data[:,col])
    return data_normalized_original, suffix