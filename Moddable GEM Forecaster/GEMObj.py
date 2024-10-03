from InputHandler import input_defaultsFile, input_getFile, input_getResponse, removeImpossibleDates, np
import concurrent.futures as cf
import ModdedFunctions as mf
class GEMObj:
    def __init__(self):
        self.filepath, self.dates, self.data = input_getFile()
        self.num_samples, self.num_var = np.shape(self.data)
        while True:
            myDict = input_defaultsFile()
            try:
                self.precipCol = myDict["precipCol"]
                self.amin = myDict["amin"]
                self.amax = myDict["amax"]
                self.bmin = myDict["bmin"]
                self.bmax = myDict["bmax"]
                self.abmin = myDict["abmin"]
                self.abmax = myDict["abmax"]
                self.f = myDict["f"]
                self.k = myDict["k"]
                self.w = myDict["w"]
                self.lag = -self.w-1
                self.lead = self.w
                self.windowSize = 2*self.w+1
                self.fvec = np.arange(1,self.f+1)
                def dateShortcuts(x):
                    if x[0] < 0:
                        if len(x) == 2:
                            x = range(-x[0], x[1] + 1)
                        else:
                            x = range(1, -x[0] + 1)
                    return x
                self.target_days = dateShortcuts(myDict["target_days"])
                self.target_months = dateShortcuts(myDict["target_months"])
                self.target_years = dateShortcuts(myDict["target_years"])
                y, m, d = np.meshgrid(self.target_years, self.target_months, self.target_days)
                y, m, d = removeImpossibleDates(y.flatten(), m.flatten(), d.flatten())
                self.num_tarDays = len(d)
                if self.num_tarDays==0:
                    print("Oops! The parameters you gave resulted in no possible target dates! Please try again.")
                    continue
                self.precipCol -= 1#Moves the precipCol 1 to the left (since python starts counting with 0)
                self.y = y
                self.m = m
                self.d = d
                self.max_workers = myDict["max_workers"]
                break
            except:
                pass
        testTrain = input_getResponse("Moddable functions initialized! Please select operation type", ["test", "train"])
        doTraining = testTrain == "train"
        if doTraining:
            print("Beginning (a, b) pair training...")
        else:
            print("Beginning forecasting...") 
        self.TestTrain(doTraining)

    def TestTrain(self, doTraining):
        bigDicti = self.makeEmptyResults()
        self.data_normalized_original, bigDicti["FNS_norm"] = self.dataNormalize()#Moddable Function
        with cf.ProcessPoolExecutor(max_workers = self.max_workers) as executor:
            # Submit tasks to the executor
            ans = executor.map(self.parloop_func, [b for b in range(self.bmin, self.bmax+1)], chunksize = 1)
            for dicti in ans:
                bigDicti = self.stackResults(dicti, bigDicti)#Moddable Function
                bigDicti["FNS_distWeight"] = dicti["FNS_distWeight"]
                bigDicti["FNS_dist"] = dicti["FNS_dist"]
                bigDicti["FNS_castWeight"] = dicti["FNS_castWeight"]
                bigDicti["FNS_cast"] = dicti["FNS_cast"]
        print("Data Collection Complete!")
        if doTraining:
            print("Beginning (a, b) pair scoring...")
            bigDicti = self.calcScore(bigDicti)#Moddable Function
            print("Choosing best (a, b) pairs...")
            bigDicti = self.selectGFV(bigDicti)#Moddable Function
        self.saveResults(bigDicti, doTraining)#Moddable Function
        

    def parloop_func(self, b: int):
        amin_b = np.maximum(np.ceil(self.abmin/b), self.amin).astype(int) - 1
        amax_b = np.minimum(np.floor(self.abmax/b), self.amax).astype(int)
        data_normalized = np.zeros(self.data_normalized_original.shape)
        grabber = np.arange(1-b, 1)
        for row in range(b, self.num_samples):
            data_normalized[row, :] = self.data_normalized_original[row + grabber, :].mean(0)
        exp = np.zeros((self.num_tarDays, amax_b))
        kexp = -np.ones((self.num_tarDays, amax_b, self.k))
        dmy = []
        excess = np.zeros(12)
        distWeight, FNS_distWeight = self.calcDistWeight(amax_b)#Moddable Function
        for day in range(self.num_tarDays):
            dateIndex_target, dateIndex_analog, FNS_ind = self.getIndices(self.y[day], self.m[day], self.d[day], amax_b, b)#Moddable Function
            if (dateIndex_target + self.f) > len(self.data):
                excess[self.m[day]-1] += 1
                exp[day, :] = -1
                kexp[day, :, :] = -1
                dmy[day, :, :] = "Not Valid"
                continue
            dist, FNS_dist = self.calcDist(data_normalized, dateIndex_target, dateIndex_analog, amin_b, amax_b, b, distWeight)#Moddable Function
            dist_ind = dist.argsort()
            castWeight, FNS_castWeight = self.calcCastWeight()#Moddable Function
            kexp[day, :, :], exp[day, :], x, FNS_cast = self.calcCast(dateIndex_analog, amin_b, amax_b, dist_ind, castWeight)#Moddable Function   
            dmy.append(x)
        print(f"b = {b} completed")
        return {"exp": trim_(exp, amin_b), "kexp" : trim_(kexp, amin_b), "dmy" : trim_(dmy, amin_b), "FNS_dist" : FNS_dist, "FNS_distWeight" : FNS_distWeight, "FNS_ind" : FNS_ind, "FNS_castWeight" : FNS_castWeight, "FNS_cast" : FNS_cast}
    
    def dataNormalize(self):
        data_normalized_original, suffix = mf.dataNormalize(self)
        return data_normalized_original, suffix
    def getIndices(self, y, m, d, amax_b, b, returnEarly = False):
        if returnEarly:
            dateIndex = mf.getIndices(self, y, m, d, amax_b, b, returnEarly)
            return dateIndex
        dateIndex_target, dateIndex_analog, suffix = mf.getIndices(self, y, m, d, amax_b, b)
        return dateIndex_target, dateIndex_analog, suffix
    def result(self, index):
        res = mf.result(self, index)
        return res
    def calcDistWeight(self, a):
        distWeight, suffix = mf.calcDistWeight(self, a)
        return distWeight, suffix
    def calcDist(self, data_normalized, dateIndex_target, dateIndex_analog, amin_b, amax_b, b, distWeight):
        dist, suffix = mf.calcDist(self, data_normalized, dateIndex_target, dateIndex_analog, amin_b, amax_b, b, distWeight)
        return dist, suffix
    def calcCastWeight(self):
        castWeight, suffix = mf.calcCastWeight(self)
        return castWeight, suffix
    def calcCast(self, dateIndex_analog, amin_b, amax_b, dist_ind, castWeight):
        kexp, exp, ymd_mat, suffix = mf.calcCast(self, dateIndex_analog, amin_b, amax_b, dist_ind, castWeight)
        return kexp, exp, ymd_mat, suffix
    def makeEmptyResults(self):
        bigDicti = mf.makeEmptyResults(self)
        return bigDicti
    def stackResults(self, dicti, bigDicti):
        bigDicti = mf.stackResults(self, dicti, bigDicti)
        return bigDicti
    def saveResults(self, bigDicti, doTraining):
        mf.saveResults(self, bigDicti, doTraining)
    def selectGFV(self, bigDicti):
        bigDicti = mf.selectGFV(self, bigDicti)
        return bigDicti
    def calcScore(self, bigDicti):
        bigDicti = mf.calcScore(self, bigDicti)
        return bigDicti
def trim_(initial, amin_b):#Trim off a values whose product with b is less than abmin
    return np.delete(initial, range(amin_b), 1)

if __name__ == "__main__":
    y = GEMObj()