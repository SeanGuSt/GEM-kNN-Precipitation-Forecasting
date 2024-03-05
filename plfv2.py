from Messages import *
import numpy as np
import pandas as pd
def trim_(initial, amin_b):#Trim off a values whose product with b is less than abmin
    return np.delete(initial, range(amin_b), 1)
def fixed_(initial, amin_b, excess, DY, do_sqrt = False):#Preform trim_, then find the average while omitting impossible dates, and taking the square root if desired.
    add_on = trim_(initial, amin_b)
    if add_on.ndim == 3:
        for i in range(np.shape(add_on)[2]):
            for M in range(12):
                add_on[M, :, i] /= (DY - excess[M])
    else:
        for M in range(12):
            add_on[M, :] /= (DY - excess[M])
    if do_sqrt:
        add_on = np.sqrt(add_on)
    return add_on
def parloop_func(gd):
    b = gd["b"]
    amin_b = gd["amin_b"]
    amax_b = gd["amax_b"]
    data = gd["data"]
    num_tarDays = gd["num_tarDays"]
    kmax = gd["kmax"]
    num_samples = gd["num_samples"]
    data_normalized_original = gd["data_normalized_original"]
    dates = gd["dates"]
    f = gd["f"]
    d = gd["d"]
    m = gd["m"]
    y = gd["y"]
    windowSize = gd["windowSize"]
    data_len = gd["data_len"]
    df = gd["df"]
    precipCol = gd["precipCol"]
    lag = gd["lag"]
    lead = gd["lead"]
    abmax = gd["abmax"]
    weight_type = gd["weight_type"]
    kTemp = gd["kTemp"]
    weight = gd["weight"]
    grabber = np.arange(1-b, 1)
    DY = gd["DY"]
    data_normalized = np.zeros(data.shape)
    exp = np.zeros((num_tarDays, amax_b))
    kexp = -np.ones((num_tarDays, amax_b, kmax))
    obs = np.zeros(num_tarDays)
    square_errors = np.zeros((12, amax_b))
    ci = np.zeros((12, amax_b, kmax))
    cl = np.zeros((12, amax_b, kmax))
    mink = np.zeros((num_tarDays, amax_b, kmax))
    maxk = np.zeros((num_tarDays, amax_b, kmax))
    excess = np.zeros(12)
    for row in range(b, num_samples):
        data_normalized[row, :] = data_normalized_original[row + grabber, :].mean(0)
    for day in range(num_tarDays):
        dateIndex_bools = [True]*num_samples
        for i in range(num_samples):
            dateIndex_bools[i] = dates[i,0]<=y[day] and dates[i,1]==m[day] and dates[i,2]==d[day]
        dateIndex = np.nonzero(dateIndex_bools)
        dateIndex = dateIndex[0]
        dateIndex_target = dateIndex[-1]
        if (dateIndex_target + f) > data_len:
            excess[m[day]-1] += 1
            obs[day] = -1
            exp[day, :] = -1
            kexp[day, :, :] = -1
            continue
        obs[day] = data[dateIndex_target + df, precipCol].mean()
        dateIndex = np.delete(dateIndex, dateIndex >= dateIndex_target - f)
        dateIndex = np.delete(dateIndex, dateIndex < abmax - lag)
        num_analogDates = windowSize*dateIndex.size
        dateIndex_analog0 = np.arange(num_analogDates)
        if not isinstance(kTemp, int):
            k = int(np.floor(np.sqrt(num_analogDates)))
        else:
           k = kmax
        if weight_type == "uniform":
            K = np.ones(k)/k
        elif weight_type == "neighbor":
            K = 1/np.arange(1,k+1)
            K = K/K.sum()
        for i in range(dateIndex.size):
            dateIndex_analog0[(windowSize*i):windowSize*(i+1)] = np.arange(lag + dateIndex[i] + 1, lead + dateIndex[i] + 1, 1)
        dateIndex_analog = dateIndex_analog0
        num_analogDates = dateIndex_analog.size
        ed = np.zeros(num_analogDates)
        for a in range(amax_b):
            ed += np.dot(np.power(data_normalized[dateIndex_target,:] - data_normalized[dateIndex_analog, :], 2), weight)
            dateIndex_target -= b
            dateIndex_analog -= b
            if a < amin_b:
                continue
            ed_ind = ed.argsort()
            if weight_type == "dist":
                K = 1./np.sqrt(ed[ed_ind[range(k)]])
                K /= sum(K)
            for i in range(k):
                kexp[day, a, i] = data[dateIndex_analog0[ed_ind[i]] + df, precipCol].mean()
                exp[day, a] += kexp[day, a, i]*K[i]
            for i in range(1, k):
                mink[day, a, i], maxk[day, a, i], ci[m[day]-1, a, i], cl[m[day]-1, a, i] = ci_and_cl(kexp[day, a, range(i+1)], obs[day], ci[m[day]-1, a, i], cl[m[day]-1, a, i])
            square_errors[m[day]-1, a] += np.power(obs[day]-exp[day, a], 2)
    print(f"b = {b} completed")
    return {"sq_err": fixed_(square_errors, amin_b, excess, DY, True), "exp": trim_(exp, amin_b), "obs": obs, "ci": fixed_(ci, amin_b, excess, DY), "cl": fixed_(cl, amin_b, excess, DY), "kexp" : np.delete(kexp, range(amin_b), 1), "mink" : np.delete(mink, range(amin_b), 1), "maxk" : np.delete(maxk, range(amin_b), 1)}

def parloop_getk(gd):
    filepath = gd["filepath"]
    i = gd["i"]
    x = pd.read_csv(f"{filepath}{i+1}.csv").to_numpy()
    return x

def ci_and_cl(kvals, o, ci, cl):
    lb = np.min(kvals)
    ub = np.max(kvals)
    ci += ub - lb
    cl += (lb <= o and o <= ub)
    return lb, ub, ci, cl