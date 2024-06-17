import tkinter as tk
from Messages import * 
import pandas as pd
import numpy as np
import json
import time
from tkinter.filedialog import askopenfilename, askdirectory
from InputHandler import removeImpossibleDates
import concurrent.futures as cf
import threading as thr
import os
from plfv2 import parloop_func, parloop_getk, ci_and_cl
import matplotlib.colors as mcolors
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

def threader_station_file():
    t1=thr.Thread(target=open_station_file) 
    t1.start()
def threader_fc_folder():
    t1=thr.Thread(target=make_fc_folder) 
    t1.start()
def threader_fc_loader():
    t1=thr.Thread(target=load_fc_folder) 
    t1.start()
def threader_draw():
    t1 = thr.Thread(target = draw_graphs)
    t1.start()
def threader_figs():
    t1 = thr.Thread(target = save_figs)
    t1.start()
def threader_fail_data():
    t1 = thr.Thread(target = save_alone)
    t1.start()
"""def threader_fc_tester():
    t1 = thr.Thread(target = test_ab)
    t1.start()
def test_ab():
    return 0
"""
def event_draw(event):
    if not str.isdigit(var_klimit.get()):#or not str.isdigit(var_firstyear.get()) or not str.isdigit(var_lastyear.get()):
        return
    threader_draw()

def superdict(i):
    gd = {}; gd["filepath"] = filepath; gd["dates"] = dates; gd["data"] = data; gd["num_tarDays"] = num_tarDays; gd["num_samples"] = num_samples; gd["num_var"] = num_var
    gd["weight"] = weight; gd["data_normalized_original"] = data_normalized_original;  gd["lag"] = lag; gd["lead"] = lead;  gd["f"] = f; gd["windowSize"] = windowSize
    gd["df"] = df; gd["data_len"] = data_len; gd["DY"] = DY; gd["kTemp"] = kTemp; gd["kmax"] = kmax; gd["d"] = d; gd["m"] = m; gd["y"] = y
    gd["b"] = range_b[i]; gd["amin_b"] = amin_[i]; gd["amax_b"] = amax_[i]; gd["precipCol"] = precipCol; gd["abmax"] = abmax; gd["weight_type"] = weight_type
    return gd
def minidict(filepath, i):
    gd = {}; gd["filepath"] = filepath; gd["i"] = i
    return gd        

def make_dir_if_new(name):
    if not os.path.exists(name):
        os.makedirs(name)
def update_notice(new_message):
    v.set(new_message)
def abelize_buttons(new_message, enabled):
    update_notice(new_message)
    state = "normal" if enabled else "disabled"
    state_entry = "normal" if enabled and var_figsdrawn.get() else "disabled"
    btn_station["state"] = state
    btn_params["state"] = state
    btn_ab_folder["state"] = state
    btn_load["state"] = state
    rdo_cl["state"] = state
    rdo_rmse["state"] = state
    chk_alone["state"] = state_entry
    ent_k["state"] = state_entry
    """ent_fy["state"] = state_entry
    ent_ly["state"] = state_entry"""
    btn_savefigs["state"] = state_entry
    btn_savealone["state"] = state_entry

def open_station_file():
    abelize_buttons(MSG_PLEASE_WAIT, False)
    global filepath, dates, data
    filepath = askopenfilename(
        filetypes=[("Station Data (*.xlsx)", ".xlsx")]
    )
    if not filepath:
        abelize_buttons(MSG_BAD_FILE, True)
        return
    book = pd.read_excel(filepath)
    book = book.dropna()
    dates_data = book.to_numpy()
    if isinstance(dates_data[0,0], str):
        dates_data = np.delete(dates_data, 0, 0)
    dates = dates_data[:, :3].astype(int)
    data = dates_data[:, 3:]
    have_data.set(True)
    abelize_buttons(MSG_STATION_GET, True)

def open_param_file():
    global precipCol, amin, amax, bmin, bmax, abmin, abmax, f, kTemp, w, test_target_years, test_target_months, test_target_days, weight_type, d, m, y, amin_, amax_, num_b, num_ab, range_b, num_tarDays, num_samples, num_var, kmax, data_normalized_original, windowSize, lag, lead, weight, data_len, df, DY, kTemp, lend, leny, max_workers
    if not have_data.get():
        update_notice(MSG_GIB_DATA)
        return
    update_notice(MSG_PLEASE_WAIT)
    filepath = askopenfilename(
        filetypes=[("Parameters (*.txt)", ".txt")]
    )
    if not filepath:
        update_notice(MSG_BAD_FILE)
        return
    with open(filepath) as f:
        params = json.loads(f.read())
    for name in var_names:
            globals()[name] = params[name]
    if test_target_months[0] < 0:
        try:
            test_target_months = range(-test_target_months[0], test_target_months[1] + 1)
        except:
            test_target_months = range(1, -test_target_months[0] + 1)
    if test_target_years[0] < 0:
        test_target_years = range(-test_target_years[0], test_target_years[1] + 1)
    y, m, d = np.meshgrid(test_target_years, test_target_months, test_target_days)
    y, m, d = removeImpossibleDates(y.flatten(), m.flatten(), d.flatten())
    num_tarDays = len(d)
    if num_tarDays==0:
        update_notice(ERROR_DATES)
        return
    if all(weight_type != x for x in dist_weight_type_list):
        update_notice(ERROR_WEIGHT)
        return
    if not (isinstance(kTemp, int) or kTemp == "root_analogues"):
        update_notice(ERROR_K)
        return
    precipCol -= 1
    num_samples, num_var = np.shape(data)
    data_normalized_original = np.zeros(data.shape)
    weight = np.ones(num_var)
    #weight[precipCol] = 3
    for col in range(num_var):
        data_normalized_original[:, col] = data[:,col] - np.average(data[:,col])
        data_normalized_original[:, col] /= np.std(data[:,col])
    range_b = np.arange(bmin, bmax+1).astype(int)
    num_b = len(range_b)
    amin_ = np.maximum(np.ceil(abmin/range_b), amin).astype(int) - 1
    amax_ = np.minimum(np.floor(abmax/range_b), amax).astype(int)
    num_ab = np.sum(amax_ - amin_)
    lag = -w-1
    lead = w
    windowSize = 2*w+1
    df = np.arange(1,f+1)
    data_len = len(data)
    lend = len(test_target_days)
    leny = len(test_target_years)
    DY = len(test_target_days)*len(test_target_years)
    if isinstance(kTemp, int):
        kmax = kTemp
    else:
        kmax = int(np.floor(np.sqrt((np.amax(y) - np.amin(dates[:,0]))*windowSize)))
    var_klimit.set(kmax)
    var_firstyear.set(y[0])
    var_lastyear.set(y[-1])
    have_params.set(True)
    update_notice(MSG_PARAM_GET)

def make_fc_folder():
    global obs, obs_all, exp, kexp, sqr_err, ci, cl, chosen_pairs, lend, leny, DY, direc, index_ea
    if not have_data.get():
        update_notice(MSG_GIB_DATA)
        return
    if not have_params.get():
        update_notice(MSG_GIB_PARAM)
        return
    if not str.isdigit(var_klimit.get()):
        return
    if var_chooser.get() == 0:
        update_notice(MSG_GIB_METHOD)
        return
    def get_time(secs):
        hrs = np.floor(secs/3600)
        secs -= 3600*hrs
        mins = np.floor(secs/60)
        secs -= 60*mins
        return hrs.astype(int), mins.astype(int), np.floor(secs).astype(int)
    hrs, mins, secs = get_time((0.000033*num_samples + 0.00208*num_tarDays)*num_ab + 12*kmax)
    abelize_buttons(f"Proceeding with training. Please allow {hrs} hr. {mins} min. {secs} sec.", False)
    timerStart = time.time()
    header = []
    for b in range(num_b):
        for a in range(amin_[b]+1,amax_[b]+1):
            header.append(f"({a};{range_b[b]})")
    index_ea = []
    for i in range(num_tarDays):
        index_ea.append(f"{y[i]}-{m[i]}-{d[i]}")
    index_sea = ["Jan.", "Feb", "Mar.", "Apr.", "May", "June", "July", "Aug.", "Sep.", "Oct.", "Nov", "Dec."]
    exp = np.empty((num_tarDays, 0))
    kexp = np.empty((num_tarDays, 0, kmax))
    sqr_err = np.empty((NUM_MONTHS,0))
    ci = np.empty((NUM_MONTHS, 0, kmax))
    cl = np.empty((NUM_MONTHS, 0, kmax))
    dmy = np.empty((num_tarDays, 0, kmax))
    mink = np.empty((num_tarDays, 0, kmax))
    maxk = np.empty((num_tarDays, 0, kmax))
    num_o = np.shape(data)[0]
    obs_all = np.zeros((num_o - f, 2))
    obs_all[:, 0] = dates[:(num_o-f), 1]
    for i in range(num_o-f):
        obs_all[i, 1] = data[i + df, precipCol].mean()
    with cf.ProcessPoolExecutor(max_workers = max_workers) as executor:
        # Submit tasks to the executor
        ans = executor.map(parloop_func, [superdict(i) for i in range(num_b)], chunksize = 1)
        for dicti in ans:
            exp = np.hstack((exp, dicti["exp"]))
            ci = np.hstack((ci, dicti["ci"]))
            cl = np.hstack((cl, dicti["cl"]))
            sqr_err = np.hstack((sqr_err, dicti["sq_err"]))
            dmy = np.hstack((dmy, dicti["dmy"]))
            kexp = np.hstack((kexp, dicti["kexp"]))
            #Remove ALL 6 Quotation Marks before and after if you want these data
            """mink = np.hstack((mink, dicti["mink"]))
            maxk = np.hstack((maxk, dicti["maxk"]))"""
            obs = dicti["obs"]
    chosen_pairs_real = []
    chosen_pairs = np.zeros((NUM_MONTHS, kmax+1), dtype = int)
    for i in range(NUM_MONTHS):
        if i+1 not in test_target_months:
            chosen_pairs[i, :] = -1
            cpr = ["-1" for _ in range(kmax+1)]
        else:
            cpr = ["" for _ in range(kmax+1)]
            chosen_pairs[i, 0] = np.argmin(sqr_err[i,:])
            cpr[0] = header[chosen_pairs[i, 0]]
            for j in range(kmax):
                chosen_pairs[i, j+1] = np.argmax(cl[i,:, j])
                cpr[j+1] = header[chosen_pairs[i, j+1]]
        chosen_pairs_real.append(cpr)      
    lend = len(test_target_days)
    leny = len(test_target_years)
    other = [lend, leny, DY, f, kmax, max_workers]
    stationname = filepath.split("/")[-1].split(" ")[0]
    h2 = ["by RMSE", "by Conf. Level for k = 1"]
    for i in range(2, kmax+1):
        h2.append(f"k = {i}")
    direc = "/".join(filepath.split("/")[:-1])
    direc = f"{direc}/{stationname}v2 f{f} w{w} k_Type{kTemp} weight_Type{weight_type}"
    dataframe_ea = pd.DataFrame(exp, columns=header, index=index_ea)
    dataframe_sea = pd.DataFrame(sqr_err, columns=header, index=index_sea)
    dataframe_obs = pd.DataFrame(obs, columns=["No (a;b)"], index=index_ea)
    dataframe_all = pd.DataFrame(obs_all[:, 1], columns=["No (a;b)"], index = obs_all[:, 0])
    #dataframe_ci = pd.DataFrame(ci, columns=header, index=index_sea)
    #dataframe_cl = pd.DataFrame(cl, columns = header, index = index_sea)
    dataframe_cab = pd.DataFrame(chosen_pairs, columns=h2, index=index_sea)
    dataframe_cabr = pd.DataFrame(chosen_pairs_real, columns=h2, index=index_sea)
    dataframe_other = pd.DataFrame(other, index = ["lend", "leny", "DY", "f", "kmax", "max_workers"])
    make_dir_if_new(direc)
    with pd.ExcelWriter(f'{direc}/{FIL_NAME}.xlsx') as writer:
        dataframe_ea.to_excel(writer, sheet_name = SHT_EXP)
        dataframe_sea.to_excel(writer, sheet_name = SHT_ERR)
        dataframe_obs.to_excel(writer, sheet_name = SHT_OBS)
        dataframe_all.to_excel(writer, sheet_name = SHT_ALLO)
        #dataframe_ci.to_excel(writer, sheet_name = SHT_CI)
        #dataframe_cl.to_excel(writer, sheet_name = SHT_CL)
        dataframe_cabr.to_excel(writer, sheet_name = SHT_RCAB)
        dataframe_cab.to_excel(writer, sheet_name = SHT_CAB)
        dataframe_other.to_excel(writer, sheet_name = SHT_OTHER)
    make_dir_if_new(f"{direc}/{FOL_K}")
    make_dir_if_new(f"{direc}/{FOL_CI}")
    make_dir_if_new(f"{direc}/{FOL_CL}")
    make_dir_if_new(f"{direc}/{FOL_DMY}")
    #Remove ALL 6 Quotation Marks before and after if you want this data
    """make_dir_if_new(f"{direc}/{FOL_MINK}")
    make_dir_if_new(f"{direc}/{FOL_MAXK}")"""
    for i in range(kmax):
        dataframe_k = pd.DataFrame(kexp[:, :, i], columns=header, index=index_ea)
        dataframe_k.to_csv(f'{direc}/{FOL_K}/{FIL_K}{i+1}.csv')
        dataframe_ci = pd.DataFrame(ci[:, :, i], columns=header, index=index_sea)
        dataframe_cl = pd.DataFrame(cl[:, :, i], columns = header, index = index_sea)
        dataframe_dmy = pd.DataFrame(dmy[:, :, i], columns = header, index = index_ea)
        dataframe_ci.to_csv(f'{direc}/{FOL_CI}/{FIL_CI}{i+1}.csv')
        dataframe_cl.to_csv(f'{direc}/{FOL_CL}/{FIL_CL}{i+1}.csv')
        dataframe_dmy.to_csv(f'{direc}/{FOL_DMY}/{FIL_DMY}{i+1}.csv')
        #Remove ALL 6 Quotation Marks before and after if you want this data
        """dataframe_min = pd.DataFrame(mink[:, :, i], columns = header, index = index_ea)
        dataframe_max = pd.DataFrame(maxk[:, :, i], columns = header, index = index_ea)
        dataframe_min.to_csv(f'{direc}/{FOL_MINK}/{FIL_MINK}{i+1}.csv')
        dataframe_max.to_csv(f'{direc}/{FOL_MAXK}/{FIL_MAXK}{i+1}.csv')"""
    timerEnd = time.time()
    have_results.set(True)
    e = dataframe_obs.index
    index_ea = e
    draw_graphs()
    hrs, mins, secs = get_time(timerEnd - timerStart)
    abelize_buttons(f"Procedure Completed in {hrs} hr. {mins} min. {secs} sec.", True)

def load_fc_folder():
    global obs, obs_all, kexp, ci, cl, chosen_pairs, DY, lend, leny, f, kmax, direc, index_ea, max_workers
    if var_chooser.get() == 0:
        update_notice(MSG_GIB_METHOD)
        return
    abelize_buttons(MSG_PLEASE_WAIT, False)
    direc = askdirectory()
    if not direc:
        abelize_buttons(MSG_BAD_DIR, True)
        return
    try: 
        book = pd.ExcelFile(f"{direc}/{FIL_NAME}.xlsx")
    except:
        abelize_buttons(ERR_FIL, True)
        return
    try:
        exp = book.parse(SHT_EXP).to_numpy()
        obs = book.parse(SHT_OBS).to_numpy()
        obs_all = book.parse(SHT_ALLO).to_numpy()
        chosen_pairs = book.parse(SHT_CAB).to_numpy()
        other = book.parse(SHT_OTHER).to_numpy()
    except:
        abelize_buttons(ERR_SHT, True)
        return
    lend = int(other[0, 1]); leny = int(other[1, 1]); DY = int(other[2, 1]); f = int(other[3, 1]); kmax = int(other[4, 1]); max_workers = int(other[5, 1])
    var_klimit.set(kmax)
    num_tarDays, num_ab = np.shape(exp)
    kexp = -np.ones((num_tarDays, num_ab - 1, kmax))
    ci= np.zeros((num_tarDays, num_ab - 1, kmax))
    cl= np.zeros((num_tarDays, num_ab - 1, kmax))
    with cf.ProcessPoolExecutor(max_workers = max_workers) as executor:
        # Submit tasks to the executor
        try:
            filepath_k = f"{direc}/{FOL_K}/{FIL_K}"
            ans1 = executor.map(parloop_getk, [minidict(filepath_k, i) for i in range(kmax)])
            """filepath_ci = f"{direc}/{FOL_CI}/{FIL_CI}"
            ans2 = executor.map(parloop_getk, [minidict(filepath_ci, i) for i in range(kmax)])
            filepath_cl = f"{direc}/{FOL_CL}/{FIL_CL}"
            ans3 = executor.map(parloop_getk, [minidict(filepath_cl, i) for i in range(kmax)])"""
        except:
            abelize_buttons(ERR_FIL, True)
            return
        i = 0
        for ks in ans1:
            kexp[:, :, i] = ks[:, 1:]
            i += 1
        """i = 0
        for cis in ans2:
            ci[:, :, i] = cis[:, 1:]
            i += 1
        i = 0
        for clss in ans3:
            cl[:, :, i] = clss[:, 1:]
            i += 1"""
    have_results.set(True)
    index_ea = obs[:, 0]
    obs = obs[:, 1]
    chosen_pairs = chosen_pairs[:, 1:]
    var_firstyear.set(index_ea[0].split("-")[0])
    var_lastyear.set(index_ea[-1].split("-")[0])
    abelize_buttons(MSG_LOD, True)
    draw_graphs()
    

def draw_graphs():
    global index_eaf, g_low, g_high, ob
    if not have_results.get():
        return
    if var_chooser.get() == 0:
        update_notice(MSG_GIB_METHOD)
        return
    abelize_buttons(MSG_PREP_GRAPHS, False)
    klimit = int(var_klimit.get())
    pairs = chosen_pairs[:, klimit*(var_chooser.get() - 1)]
    knn_min = np.zeros((DY, NUM_MONTHS)); knn_max = np.zeros((DY, NUM_MONTHS)); obse = np.zeros((DY, NUM_MONTHS))
    ind = np.arange(DY)
    histox = []; ciset = []; clset = []; obrset = []; histxticks = []; netchange = []
    plotxticks = np.arange(int(np.ceil(lend/2)), np.sum(pairs>-1)*DY, np.sum(pairs>-1)*lend)
    plotxticklabels = sorted(set([int(a.split("-")[0])for a in index_ea]))
    for i in range(NUM_MONTHS):
        if pairs[i] > -1:
            ci1 = 0
            cl1 = 0
            for j in range(DY):
                knn_min[j, i], knn_max[j, i], ci1, cl1 = ci_and_cl(kexp[ind[j], pairs[i], range(klimit)], obs[ind[j]], ci1, cl1)
            obse[:, i] = obs[ind]
            ind += DY
            histox.append(index_sea[i])
            ciset.append(ci1*f/DY)
            clset.append(cl1/DY)
            histxticks.append("CL: %.2f%% %s" % (clset[-1]*100, index_sea[i]))
            x = obs_all[obs_all[:, 0] == i + 1, 1]
            obrset.append((np.percentile(x, 50*(1+clset[-1])) - np.percentile(x, 50*(1-clset[-1])))*f)
            chang = (ciset[-1]/obrset[-1])*100 - 100
            netchange.append("%s%.2f%%" % ("+" if chang >= 0 else "-", abs(chang)))

    x = np.arange(len(obs), dtype = np.float64); g_low = np.arange(len(obs), dtype = np.float64); g_high = np.arange(len(obs), dtype = np.float64); ob = np.arange(len(obs), dtype = np.float64);
    ind_i = np.arange(lend)
    ind_iI = np.arange(lend)
    for I in range(leny):
        for i in range(NUM_MONTHS):
            if pairs[i] > -1:
                g_low[ind_iI] = knn_min[ind_i, i]
                g_high[ind_iI] = knn_max[ind_i, i]
                ob[ind_iI] = obse[ind_i, i]
                ind_iI += len(ind_i)
        ind_i += len(ind_i)
    if var_isolate.get():
        failures = [i or j for i, j in zip(g_low > ob, ob > g_high)]
        x = x[failures]
        ob = ob[failures]
        g_low = g_low[failures]
        g_high = g_high[failures]
        index_eaf = index_ea[failures]
    bigplot.clear()
    bighist.clear()
    bigplot.plot(x, g_low*f, "k", linewidth = 0.2, label = "GEM Lower Bound")
    bigplot.plot(x, g_high*f, "k", linewidth = 0.2, label = "GEM Upper Bound")
    bigplot.fill_between(x, g_low*f, g_high*f)
    bigplot.scatter(x, ob*f, 0.5, mcolors.CSS4_COLORS["black"], label = "Observed")
    bigplot.set_xlabel("Sample Days by Year")
    bigplot.set_ylabel(f"Total Precipitation after {f} Days")
    bigplot.set(xlim = (0, len(obs)))
    bigplot.legend(loc='upper center', ncols=3, bbox_to_anchor = (0.5, 1.2))
    bigplot.set_xticks(plotxticks, plotxticklabels)
    bigplot.set_title(PLT_GRAPH_TITLE)
    bigplot.tick_params(axis="x", labelrotation = 45, labelsize = 5)
    canvas1.draw()
    histdict = {"Average GEM Dist." : ciset, "Observed Dist. Equivalent Confidence" : obrset}
    width = 0.25
    multiplier = 0
    x = np.arange(len(histox))
    for name, data in histdict.items():
        offset = width * multiplier
        bighist.bar(x + offset, data, width, label = name)
        multiplier += 1
    rects = bighist.patches
    j = 0
    for i in range(len(netchange)):
        heights = [rects[i].get_height(), rects[i+len(netchange)].get_height()]
        xpos = (rects[i].get_x() + rects[i + len(netchange)].get_x())/2
        ypos = np.max(heights) + 0.01
        bighist.text(xpos, ypos, netchange[i], ha="center", va="bottom", fontsize = "xx-small")
    bighist.legend(loc='upper center', ncols=2, bbox_to_anchor = (0.5, 1.2))
    bighist.set_title(PLT_HISTO_TITLE)
    bighist.set_xticks(range(len(x)), labels=histxticks)
    bighist.tick_params(axis="x", labelrotation = 45, labelsize = 6)
    canvas2.draw()
    var_figsdrawn.set(True)
    abelize_buttons(MSG_DONE_GRAPHS, True)

def save_figs():
    if not var_figsdrawn.get():
        return
    fig1.savefig(f"{direc}/Chart kmax{var_klimit.get()}.png")
    fig2.savefig(f"{direc}/Histogram kmax{var_klimit.get()}.png")
def save_alone():
    if not var_figsdrawn.get() or not var_isolate.get():
        return
    dataframe_ea = pd.DataFrame(np.transpose([ob, g_low, g_high]), columns=["Observed", "GEM Lower Bound", "GEM Upper Bound"], index=index_eaf)
    with pd.ExcelWriter(f'{direc}/{FAIL_NAME}{var_klimit.get()}.xlsx') as writer:
        dataframe_ea.to_excel(writer)



if __name__ == "__main__":
    window = tk.Tk()
    window.title("GEM Program 2.0 (Now with GUI)")
    window.rowconfigure(3)
    window.columnconfigure(9)

    #Window Grid Frame
    frm_buttons = tk.Frame(window, relief=tk.GROOVE, bd=2)
    frm_buttons.grid(row=0, column=0, sticky="ns")
    frm_other = tk.Frame(window, relief=tk.GROOVE, bd=2)
    frm_other.grid(row=0,column=1,sticky="w")
    frm_entry = tk.Frame(frm_buttons)
    frm_entry.grid(row = 31, column = 0)
    frm_fy = tk.Frame(frm_buttons)
    frm_fy.grid(row = 41, column = 0)
    frm_ly = tk.Frame(frm_buttons)
    frm_ly.grid(row = 51, column = 0)

    #Notice Board to update user on what's happened
    v = tk.StringVar()
    label_notice = tk.Label(frm_other, textvariable = v)
    label_notice.grid(row=0, column=0, sticky="nsew")
    v.set("Notice Board")

    #Buttons to allow the user to interact with things
    have_data = tk.BooleanVar()
    btn_station = tk.Button(frm_buttons, text=LBL_STATION, command=threader_station_file)
    btn_station.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    have_params = tk.BooleanVar()
    btn_params = tk.Button(frm_buttons, text=LBL_PARAM, command=open_param_file)
    btn_params.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
    #btn_ab_tester = tk.Button(frm_buttons, text = LBL_TEST_FC, command=threader_fc_tester)
    #btn_ab_tester.grid(row=21, column=0, sticky="ew", padx=5, pady=5)
    have_results = tk.BooleanVar()
    btn_ab_folder = tk.Button(frm_buttons, text=LBL_MAKE_FC, command=threader_fc_folder)
    btn_ab_folder.grid(row=11, column=0, sticky="ew", padx=5, pady=5)
    
    btn_load = tk.Button(frm_buttons, text=LBL_LOAD_FC, command=threader_fc_loader)
    btn_load.grid(row=21, column=0, sticky="ew", padx=5, pady=5)

    var_klimit = tk.StringVar()
    ent_label = tk.Label(frm_entry, text = "k = ")
    ent_label.grid(row = 0, column = 0)
    ent_k = tk.Entry(frm_entry, textvariable = var_klimit, state = "disabled")
    ent_k.bind("<Return>", event_draw)
    ent_k.grid(row = 0, column = 1, sticky="ew", padx=5, pady=5)

    var_firstyear = tk.StringVar()
    """fy_label = tk.Label(frm_fy, text = "First Year:")
    fy_label.grid(row = 0, column = 0)
    ent_fy = tk.Entry(frm_fy, textvariable = var_firstyear)
    ent_fy.bind("<Return>", event_draw)
    ent_fy.grid(row = 0, column = 1, sticky="ew", padx=5, pady=5)"""

    var_lastyear = tk.StringVar()
    """ly_label = tk.Label(frm_ly, text = "First Year:")
    ly_label.grid(row = 0, column = 0)
    ent_ly = tk.Entry(frm_ly, textvariable = var_lastyear)
    ent_ly.bind("<Return>", event_draw)
    ent_ly.grid(row = 0, column = 1, sticky="ew", padx=5, pady=5)"""

    var_chooser = tk.IntVar(value = AB_CONF)
    rdo_cl = tk.Radiobutton(frm_buttons, text = "GEM via highest overall confidence", variable = var_chooser, value = AB_CONF, command=threader_draw)
    rdo_cl.grid(row=61, column=0, sticky="ew", padx=5, pady=5)

    rdo_rmse = tk.Radiobutton(frm_buttons, text = "GEM via lowest overall RMSE", variable = var_chooser, value = AB_RMSE, command = threader_draw)
    rdo_rmse.grid(row=71, column=0, sticky="ew", padx=5, pady=5)

    var_figsdrawn = tk.BooleanVar()
    btn_savefigs = tk.Button(frm_buttons, text = "Save Figures", command = threader_figs)
    btn_savefigs.grid(row = 81, column = 0, sticky = "ew", padx = 5, pady = 5)

    var_isolate = tk.BooleanVar()
    chk_alone = tk.Checkbutton(frm_buttons, text="Isolate Failures", variable = var_isolate, command = threader_draw)
    chk_alone.grid(row=91, column=0, sticky="ew", padx=5, pady=5)
    btn_savealone = tk.Button(frm_buttons, text = "Save Failure Data", command = threader_fail_data)
    btn_savealone.grid(row = 101, column = 0, sticky = "ew", padx = 5, pady = 5)
    

    #Data and Distributions
    fig1 = Figure(figsize=(6, 5), dpi=100)
    bigplot = fig1.add_subplot()
    bigplot.set_xlabel("Days")
    bigplot.set_ylabel("Precip")
    fig1.subplots_adjust(top = 0.85, bottom = 0.15)
    canvas1 = FigureCanvasTkAgg(fig1, master=frm_other)  # A tk.DrawingArea.
    canvas1.draw()
    canvas1.get_tk_widget().grid(row=1, column=0, rowspan = 2, columnspan=4)

    #Confidence Levels
    fig2 = Figure(figsize=(6, 5), dpi=100)
    bighist = fig2.add_subplot()
    bighist.set_xlabel("Months")
    bighist.set_ylabel("Interval")
    fig2.subplots_adjust(top = 0.85, bottom=0.15)
    canvas2 = FigureCanvasTkAgg(fig2, master=frm_other)  # A tk.DrawingArea.
    canvas2.draw()
    canvas2.get_tk_widget().grid(row=1, column=5, rowspan = 2, columnspan=4)
    
    window.mainloop()
