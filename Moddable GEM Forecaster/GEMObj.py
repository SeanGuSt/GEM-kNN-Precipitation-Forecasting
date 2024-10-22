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
                self.A = myDict["A"]
                self.B = myDict["B"]
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
                self.y, self.m, self.d = self.getAndCheckTargetDates(self.target_years, self.target_months, self.target_days)
                if self.num_tarDays==0:
                    print("Oops! The parameters you gave resulted in no possible target dates!")
                    continue
                self.precipCol -= 1#Moves the precipCol 1 to the left (since python starts counting with 0)
                self.max_workers = myDict["max_workers"]
                break
            except:
                pass
        testTrain = input_getResponse("Moddable functions initialized! Please select operation type", ["test", "train"])
        doTraining = testTrain == "train"
        if doTraining:
            print("Beginning (a, b) pair training...")
            self.Train(doTraining)
        else:
            print("Beginning forecasting...") 
            self.Test(doTraining)
        

    def Train(self, doTraining):
        bigDicti = self.makeEmptyResults()
        self.data_normalized_original, bigDicti["FNS_norm"] = self.dataNormalize()#Moddable Function
        with cf.ProcessPoolExecutor(max_workers = self.max_workers) as executor:
            # Submit tasks to the executor
            ans = executor.map(self.parloop_func, [b for b in range(self.bmin, self.bmax+1)], chunksize = 1)
            for dicti in ans:
                bigDicti = self.stackResults(dicti, bigDicti)#Moddable Function
        print("Data Stacking Complete!")
        if True:
            print("Performing Dr. Zhang's idea...")
            exp = np.arange(self.num_tarDays)
            for day in range(self.num_tarDays):
                superDist = bigDicti["distBest"][day, :, :].flatten()
                superkexp = bigDicti["kexp"][day, :, :].flatten()
                superDist_ind = superDist.argsort()
                castWeight, _ = self.calcCastWeight(superDist_ind)#Moddable Function
                k = self.getk(superDist_ind)
                exp[day] = np.dot(superkexp[superDist_ind[:k]], castWeight)
            bigDicti["exp_Zhang"] = exp
            print("Dr. Zhang's idea has been tested!")
            print(bigDicti["exp_Zhang"])
        print("Beginning (a, b) pair scoring...")
        bigDicti = self.calcScore(bigDicti)#Moddable Function
        print("(a, b) pairs scored! Choosing best (a, b) pairs...")
        bigDicti = self.selectGFV(bigDicti)#Moddable Function
        print("(a, b) pairs chosen! Saving results to data file's directory...")
        self.saveResults(bigDicti, doTraining)#Moddable Function
    
    def Test(self, doTraining):
        bigDicti = self.makeEmptyResults()
        self.data_normalized_original, bigDicti["FNS_norm"] = self.dataNormalize()#Moddable Function
        for M in range(len(self.B)):
            if M+1 in self.target_months:
                self.amin = self.A[M] - 1
                self.amax = self.A[M]
                dicti = self.parloop_func(self.B[M])
                bigDicti = self.stackResults(dicti, bigDicti)#Moddable Function
        print("Data Collection Complete! Saving results to data file's directory...")
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
        dmy = np.array([[["" for _ in range(self.k)] for _ in range(amax_b)] for _ in range(self.num_tarDays)], dtype = str)
        excess = np.zeros(12)
        distWeight, FNS_distWeight = self.calcDistWeight(amax_b)#Moddable Function
        #At most, we can only use the k best neighbors, so only bothering recording the k lowest distances.
        distBest = -np.ones([self.num_tarDays, amax_b, self.k])
        for day in range(self.num_tarDays):
            dateIndex_target, dateIndex_analog, FNS_ind = self.getIndices(self.y[day], self.m[day], self.d[day], amax_b, b)#Moddable Function
            if (dateIndex_target + self.f) > len(self.data):
                excess[self.m[day]-1] += 1
                exp[day, :] = -1
                kexp[day, :, :] = -1
                distBest[day, :, :] = -1
                dmy[day, :, :] = "Not Valid"
                continue
            dist, FNS_dist = self.calcDist(data_normalized, dateIndex_target, dateIndex_analog, amin_b, amax_b, b, distWeight)#Moddable Function
            dist_ind = dist.argsort()
            distBest[day, :, :] = dist_ind[:, :self.k]
            castWeight, FNS_castWeight = self.calcCastWeight(dist_ind)#Moddable Function
            kexp[day, :, :], exp[day, :], dmy[day, :, :], FNS_cast = self.calcCast(dateIndex_analog, amin_b, amax_b, dist_ind, castWeight)#Moddable Function   
            
        print(f"b = {b} completed")
        return {"exp": trim_(exp, amin_b), "kexp" : trim_(kexp, amin_b), "dmy" : trim_(dmy, amin_b), "distBest" : trim_(distBest, amin_b),
                "FNS_dist" : FNS_dist, "FNS_distWeight" : FNS_distWeight, "FNS_ind" : FNS_ind, 
                "FNS_cast" : FNS_cast, "FNS_castWeight" : FNS_castWeight}
    def getAndCheckTargetDates(self, y, m, d):
        y, m, d = np.meshgrid(y, m, d)
        y, m, d = removeImpossibleDates(y.flatten(), m.flatten(), d.flatten())
        self.num_tarDays = len(d)
        return y, m, d
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
    def calcCastWeight(self, dist_ind):
        castWeight, suffix = mf.calcCastWeight(self, dist_ind)
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
    def getk(self, dist_ind):
        k = mf.getk(self, dist_ind)
        return k
def trim_(initial, amin_b):#Trim off a values whose product with b is less than abmin
    return np.delete(initial, range(amin_b), 1)

if __name__ == "__main__":
    y = GEMObj()