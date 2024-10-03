import numpy as np
def standardNormalize(self):
    suffix = "Gaussian"
    data_normalized_original = np.zeros(self.data.shape)
    for col in range(self.num_var):
        data_normalized_original[:, col] = self.data[:,col] - np.average(self.data[:,col])
        data_normalized_original[:, col] /= np.std(self.data[:,col])
    return data_normalized_original, suffix