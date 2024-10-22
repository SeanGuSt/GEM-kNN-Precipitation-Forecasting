import numpy as np
from GEMObj import GEMObj
def getIndices(self: GEMObj, y: int, m: int, d: int, amax_b: int, b: int, returnEarly = False):
    suffix = "StdInd"
    dateIndex_bools = [self.dates[i,0]<=y and self.dates[i,1]==m and self.dates[i,2]==d for i in range(self.num_samples)]
    dateIndex = np.nonzero(dateIndex_bools)
    dateIndex = dateIndex[0]
    if returnEarly:
        return dateIndex
    dateIndex_target = dateIndex[-1]
    dateIndex = np.delete(dateIndex, dateIndex >= dateIndex_target - self.f - self.lead)
    dateIndex = np.delete(dateIndex, dateIndex < amax_b*b - self.lag)
    num_analogDates = self.windowSize*dateIndex.size
    dateIndex_analog = np.arange(num_analogDates)
    for i in range(dateIndex.size):
        dateIndex_analog[(self.windowSize*i):(self.windowSize*(i+1))] = np.arange(self.lag + dateIndex[i] + 1, self.lead + dateIndex[i] + 1, 1)
    return dateIndex_target, dateIndex_analog, suffix