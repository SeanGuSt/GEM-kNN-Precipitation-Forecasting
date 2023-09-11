from InputHandler import input_defaultsFile, input_getFile, input_getYN, input_trainingFile, removeImpossibleDates, np, pd, tryAgain_standard
filepath, dates, data = input_getFile()
num_samples, num_var = np.shape(data)
precipCol = 0; f = 0; A=None; B=None; kTemp = None; w = 0; test_target_years = 0; test_target_months = 0; test_target_days = 0; weight_type = 0;
var_names = ["precipCol", "A", "B", "f", "kTemp", "w", "test_target_days", "test_target_months", "test_target_years", "weight_type"]
myDict = input_defaultsFile()
giveTraining = "n"
while True:
    try:
        for name in var_names:
            locals()[name] = myDict[name]
        dist_weight_type_list = ["uniform", "dist", "neighbor"]
        if all(weight_type != x for x in dist_weight_type_list):
            print("Oops! This is not a valid weight type! Please try again.")
            continue
        if not (isinstance(kTemp, int) or kTemp == "root_analogues"):
            print("Oops! This is not a valid k value! Please try again.")
            continue
        if not isinstance(A, type(B)):
            print("Oops! A and B aren't the same length!")
            continue
        elif isinstance(A, list):
            if not A and not B:
                giveTraining = input_getYN("You provided empty lists for A and B. Would you like to provide training data?", tryAgain_standard)
                if giveTraining.__eq__("n"):
                    print("In that case, please go back and edit the default parameters file so A and B aren't empty.")
                    continue
                else:
                    print("Alright! We'll get that from you in a moment.")
            elif len(A) != len(B):
                print("Oops! A and B aren't the same length!")
                continue
            elif not isinstance(test_target_months, list):
                print("Oops! A and B aren't the same length as test_target_months!")
                continue
            elif len(test_target_months) != len(A):
                print("Oops! A and B aren't the same length as test_target_months!")
                continue
        y, m, d = np.meshgrid(test_target_years, test_target_months, test_target_days)
        y, m, d = removeImpossibleDates(y.flatten(), m.flatten(), d.flatten())
        num_tarDays = len(d)
        if num_tarDays==0:
            print("Oops! The parameters you gave resulted in no possible training dates! Please try again.")
            continue
        precipCol -= 1
        break
    except Exception as e: 
        print(e)
        myDict = input_defaultsFile()
if giveTraining.__eq__("y"):
    A, B = input_trainingFile()
print("Initializing and normalizing data...")
data_normalized_original = np.zeros(data.shape)
data_normalized = np.zeros(data.shape)
for col in range(num_var):
    data_normalized_original[:, col] = data[:,col] - np.average(data[:,col])
    data_normalized_original[:, col] /= np.std(data[:,col])
index_ea = []
for i in range(num_tarDays):
    index_ea.append(f"{y[i]}-{m[i]}-{d[i]}")
index_sea = ["Jan.", "Feb", "Mar.", "Apr.", "May", "June", "July", "Aug.", "Sep.", "Oct.", "Nov", "Dec."]
lag = -w-1
lead = w
windowSize = 2*w+1
if isinstance(kTemp, int):
    kmax = kTemp
    k = kTemp
else:
    kmax = int(np.ceil(np.sqrt((np.amax(y) - np.amin(dates[:,0]) + 1)*windowSize)))
df = np.arange(1,f+1)
obs = np.zeros((num_tarDays,1))
exp = np.zeros((num_tarDays,1))
kexp = np.zeros((num_tarDays, kmax))
abs_err = np.zeros((num_tarDays,1))
square_errors = np.zeros(12)
weight = np.ones(num_var)
K = 1/np.arange(1,kmax+1)
K = K/K.sum()
def trueNormalData():
    grabber = np.arange(1-b, 1, dtype = int)
    for row in np.arange(b, num_samples):
        data_normalized[row, :] = data_normalized_original[row + grabber, :].mean(0)
print("Beginning testing...")
divisor = 0
for day in range(num_tarDays):
    dateIndex_bools = [True]*num_samples
    for i in range(num_samples):
        dateIndex_bools[i] = dates[i,0]<=y[day] and dates[i,1]==m[day] and dates[i,2]==d[day]
    if day==0 or m[day] != m[day-1]:
        a = int(A[m[day] - 1])
        b = int(B[m[day] - 1])
        if day == 0 or b != B[m[day-1]-1]:
            trueNormalData()
    dateIndex = np.nonzero(dateIndex_bools)
    dateIndex = dateIndex[0]
    dateIndex_target0 = dateIndex[-1]
    dateIndex_target = dateIndex_target0
    dateIndex = np.delete(dateIndex, dateIndex >= dateIndex_target - f - lead)
    dateIndex = np.delete(dateIndex, dateIndex < a*b - lag)
    num_analogDates = windowSize*dateIndex.size
    dateIndex_analog0 = np.arange(num_analogDates)
    if not isinstance(kTemp, int):
        k = int(np.floor(np.sqrt(num_analogDates)))
    if weight_type == "uniform":
        K = np.ones(k)
    elif weight_type == "neighbor":
        K = 1/np.arange(1,k+1)
        K = K/K.sum()
    for i in range(dateIndex.size):
        dateIndex_analog0[(windowSize*i):windowSize*(i+1)] = np.arange(lag + dateIndex[i] + 1, lead + dateIndex[i] + 1, 1)
    dateIndex_analog = dateIndex_analog0
    ed = np.zeros(num_analogDates)
    for i in np.arange(a):
        ed += np.dot(np.power(data_normalized[dateIndex_target,:] - data_normalized[dateIndex_analog, :], 2), weight)
        dateIndex_target -= b
        dateIndex_analog -= b
    ed_ind = ed.argsort()
    if weight_type == "dist":
        K = np.sqrt(ed[ed_ind[range(k)]])
        K /= sum(K)
    for i in range(k):
        kexp[day, i] = data[dateIndex_analog0[ed_ind[i]] + df, precipCol].mean()
        exp[day] += kexp[day, i]*K[i]
    obs[day] = -1
    abs_err[day] = -1
    if (dateIndex_target0 + f) <= num_samples:
        obs[day] = data[dateIndex_target0 + df, precipCol].mean()
        abs_err[day] = abs(obs[day]-exp[day])
        square_errors[m[day]-1] += np.power(obs[day]-exp[day], 2)
mainBox = np.hstack((exp, obs, abs_err))
filename = filepath.split("\\")[-1].split(" ")[0]
resCols = []
for i in range(1, kmax+1):
    resCols.append(f"k = {i}")
dataframe_out = pd.DataFrame(mainBox, columns=["Expected", "Observed", "Abs. Err."], index=index_ea)
dataframe_rmse = pd.DataFrame(np.sqrt(square_errors), columns = ["RMSE"], index=index_sea)
dataframe_k = pd.DataFrame(kexp, columns = resCols, index = index_ea)
with pd.ExcelWriter(f'{filename} testing k{k} w{w} f{f}.xlsx') as writer:
    dataframe_out.to_excel(writer, sheet_name = "Results")
    dataframe_rmse.to_excel(writer, sheet_name = "RMSE")
    dataframe_k.to_excel(writer, sheet_name = "k Nearest Neighbors")
