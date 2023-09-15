from InputHandler import input_defaultsFile, input_getFile, input_getArray, input_getInt, input_getYN, removeImpossibleDates, np, pd, tryAgain_standard
filepath, dates, data = input_getFile()
num_samples, num_var = np.shape(data)
precipCol = 0; amin = 0; amax = 0; bmin = 0; bmax = 0; abmin = 0; abmax = 0; f = 0; kTemp = 0; w = 0; test_target_years = 0; test_target_months = 0; test_target_days = 0; weight_type = 0;
var_names = ["precipCol", "amin", "amax", "bmin", "bmax", "abmin", "abmax", "f", "kTemp", "w", "test_target_days", "test_target_months", "test_target_years", "weight_type"]
myDict = input_defaultsFile()
while True:
    try:
        for name in var_names:
            locals()[name] = myDict[name]
        if test_target_years[0] < 0:
            test_target_years = range(-test_target_years[0], test_target_years[1] + 1)
        y, m, d = np.meshgrid(test_target_years, test_target_months, test_target_days)
        y, m, d = removeImpossibleDates(y.flatten(), m.flatten(), d.flatten())
        num_tarDays = len(d)
        if num_tarDays==0:
            print("Oops! The parameters you gave resulted in no possible training dates! Please try again.")
            continue
        dist_weight_type_list = ["uniform", "dist", "neighbor"]
        if all(weight_type != x for x in dist_weight_type_list):
            print("Oops! This is not a valid weight type! Please try again.")
            continue
        if not (isinstance(kTemp, int) or kTemp == "root_analogues"):
            print("Oops! This is not a valid k value! Please try again.")
            continue
        precipCol -= 1
        break
    except:
        myDict = input_defaultsFile()            
print("Thank you! all data and parameters are input. Beginning training of GEM model parameters.")
print("Initializing and normalizing data...")
data_normalized_original = np.zeros(data.shape)
weight = np.ones(num_var)
for col in range(num_var):
    data_normalized_original[:, col] = data[:,col] - np.average(data[:,col])
    data_normalized_original[:, col] /= np.std(data[:,col])
range_b = np.arange(bmin, bmax+1).astype(int)
num_b = len(range_b)
amin_ = np.maximum(np.ceil(abmin/range_b), amin).astype(int) - 1
amax_ = np.minimum(np.floor(abmax/range_b), amax).astype(int)
num_ab = np.sum(amax_ - amin_)
header = []
for b in range(num_b):
    for a in range(amin_[b]+1,amax_[b]+1):
        header.append(f"({a}, {range_b[b]})")
index_ea = []
for i in range(num_tarDays):
    index_ea.append(f"{y[i]}-{m[i]}-{d[i]}")
index_sea = ["Jan.", "Feb", "Mar.", "Apr.", "May", "June", "July", "Aug.", "Sep.", "Oct.", "Nov", "Dec."]
obs = np.zeros(num_tarDays)
exp_all = np.empty((num_tarDays,0))
square_errors_all = np.empty((12,0))
lag = -w-1
lead = w
windowSize = 2*w+1
if isinstance(kTemp, int):
    kmax = kTemp
    k = kTemp
else:
    kmax = int(np.ceil(np.sqrt((np.amax(y) - np.amin(dates[:,0]) + 1)*windowSize)))
for i in range(kmax):
    locals()[f"k{i}"] = np.empty((num_tarDays, 0))
df = np.arange(1,f+1)
estimated_seconds = (0.000033*num_samples + 0.125*num_tarDays)*num_b
estimated_hours = np.floor(estimated_seconds/3600)
estimated_seconds -= 3600*estimated_hours
estimated_minutes = np.floor(estimated_seconds/60)
estimated_seconds -= 60*estimated_minutes
print(f"Proceeding with training. Please allow {estimated_hours.astype(int)} hr. {estimated_minutes.astype(int)} min. {np.floor(estimated_seconds).astype(int)} sec.")
for ind_b in range(num_b):
    b = range_b[ind_b]
    grabber = np.arange(1-b, 1)
    data_normalized = np.zeros(data.shape)
    amin_b = amin_[ind_b]
    amax_b = amax_[ind_b]
    exp = np.zeros((num_tarDays, amax_b))
    kexp = -np.ones((num_tarDays, amax_b, kmax))
    square_errors = np.zeros((12, amax_b))
    for row in range(b, num_samples):
        data_normalized[row, :] = data_normalized_original[row + grabber, :].mean(0)
    for day in range(num_tarDays):
        dateIndex_bools = [True]*num_samples
        for i in range(num_samples):
            dateIndex_bools[i] = dates[i,0]<=y[day] and dates[i,1]==m[day] and dates[i,2]==d[day]
        dateIndex = np.nonzero(dateIndex_bools)
        dateIndex = dateIndex[0]
        dateIndex_target = dateIndex[-1]
        obs[day] = data[dateIndex_target + df, precipCol].mean()
        dateIndex = np.delete(dateIndex, dateIndex >= dateIndex_target - f)
        dateIndex = np.delete(dateIndex, dateIndex < abmax - lag)
        num_analogDates = windowSize*dateIndex.size
        dateIndex_analog0 = np.arange(num_analogDates)
        if not isinstance(kTemp, int):
            k = int(np.floor(np.sqrt(num_analogDates)))
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
                K = np.sqrt(ed[ed_ind[range(k)]])
                K /= sum(K)
            for i in range(k):
                kexp[day, a, i] = data[dateIndex_analog0[ed_ind[i]] + df, precipCol].mean()
                exp[day, a] += kexp[day, a, i]*K[i]
            square_errors[m[day]-1, a] += np.power(obs[day]-exp[day, a], 2)
    add_on = np.delete(square_errors, range(amin_b), 1)
    square_errors_all = np.hstack((square_errors_all, np.sqrt(add_on/(len(test_target_days)*len(test_target_years)))))
    add_on = np.delete(exp, range(amin_b), 1)
    exp_all = np.hstack((exp_all, add_on))
    for i in range(kmax):
        k_on = np.delete(kexp[:, :, i], range(amin_b), 1)
        locals()[f"k{i}"] = np.hstack((locals()[f"k{i}"], k_on))
    
chosen_pairs = []
for i in range(1, 13):
    if i not in test_target_months:
        chosen_pairs.append(["(0, 0)", -1])
    else:
        ind_min = np.argmin(square_errors_all[i-1,:])
        chosen_pairs.append([header[ind_min], square_errors_all[i-1, ind_min]])
filename = filepath.split("\\")[-1].split(" ")[0]
dataframe_ea = pd.DataFrame(exp_all, columns=header, index=index_ea)
dataframe_sea = pd.DataFrame(square_errors_all, columns=header, index=index_sea)
dataframe_obs = pd.DataFrame(obs, columns=["No (a, b)"], index=index_ea)
dataframe_cab = pd.DataFrame(chosen_pairs, columns = ["Chosen", "RMSE (unit/day)"], index=index_sea)
with pd.ExcelWriter(f'{filename} training k_type{kTemp} weight_type{weight_type} w{w} f{f}.xlsx') as writer:
    dataframe_ea.to_excel(writer, sheet_name = "Forecasts")
    dataframe_obs.to_excel(writer, sheet_name = "Observed")
    dataframe_sea.to_excel(writer, sheet_name = "RMSE")
    dataframe_cab.to_excel(writer, sheet_name = "Chosen Pairs")
    for i in range(kmax):
        dataframe_k = pd.DataFrame(locals()[f"k{i}"], columns=header, index=index_ea)
        dataframe_k.to_excel(writer, sheet_name = f"k = {i+1}")








