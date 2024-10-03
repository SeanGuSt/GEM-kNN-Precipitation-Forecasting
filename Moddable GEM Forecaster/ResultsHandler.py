import numpy as np
import pandas as pd
import os
from Messages import *
from GEMObj import GEMObj
def initialDict(self):
    bigDicti = {}
    bigDicti["exp"] = np.empty((self.num_tarDays, 0))
    bigDicti["kexp"] = np.empty((self.num_tarDays, 0, self.k))
    bigDicti["dmy"] = np.empty((self.num_tarDays, 0, self.k))
    return bigDicti
def stackResults(self: GEMObj, dicti: dict, bigDicti: dict):
    bigDicti["exp"] = np.hstack((bigDicti["exp"], dicti["exp"]))
    bigDicti["dmy"] = np.hstack((bigDicti["dmy"], dicti["dmy"]))
    bigDicti["kexp"] = np.hstack((bigDicti["kexp"], dicti["kexp"]))
    return bigDicti

def saveResults(self: GEMObj, bigDicti: dict, train: bool):
    FNS_norm = bigDicti["FNS_norm"]
    FNS_distWeight = bigDicti["FNS_distWeight"]
    FNS_dist = bigDicti["FNS_dist"]
    FNS_castWeight = bigDicti["FNS_castWeight"]
    FNS_cast = bigDicti["FNS_cast"]
    stationname = self.filepath.split("/")[-1].split(" ")[0]#Name of station (It is assumed the station's name has no spaces)
    direc = "/".join(self.filepath.split("/")[:-1])#Get the directory where the station is found. This is where the new file is.
    direc = f"{direc}/{stationname}GEMObj f{self.f} w{self.w} norm{FNS_norm} distW{FNS_distWeight} dist{FNS_dist} castW{FNS_castWeight} cast{FNS_cast}"
    make_dir_if_new(direc)
    header = getDataFrameHeader(self)
    index_ea = [f"{self.d[i]}/{self.m[i]}/{self.y[i]}" for i in range(self.num_tarDays)]
    index_sea = ["Jan.", "Feb", "Mar.", "Apr.", "May", "June", "July", "Aug.", "Sep.", "Oct.", "Nov", "Dec."]
    make_dir_if_new(f"{direc}/{FOL_K}")
    make_dir_if_new(f"{direc}/{FOL_DMY}")
    #Let num_ab be the number of valid (a, b) pairs input by the parameter file. Then, the following dimensions are:
    exp = bigDicti["exp"]#self.num_tarDays x num_ab
    kexp = bigDicti["kexp"]#self.num_tarDays x num_ab x self.k
    dmy = bigDicti["dmy"]#self.num_tarDays x num_ab x self.k
    chosen_pairs = bigDicti["chosen_pairs"]#self.num_tarDays x num_ab
    chosen_pairs_real = bigDicti["chosen_pairs_real"]#self.num_tarDays x num_ab
    GEMC_compare = bigDicti["GEMC_compare"]#self.num_tarDays x num_ab
    with pd.ExcelWriter(f'{direc}/{FIL_NAME}.xlsx') as writer:
        dataframe_ea = pd.DataFrame(exp, columns=header, index=index_ea)
        dataframe_ea.to_excel(writer, sheet_name = SHT_EXP)
        if train:
            dataframe_score = pd.DataFrame(GEMC_compare, columns = header, index=index_ea)
            dataframe_cab = pd.DataFrame(chosen_pairs, columns=["(a, b) Pair"], index=index_sea)
            dataframe_cabr = pd.DataFrame(chosen_pairs_real, columns=["(a, b) Pair"], index=index_sea)
            dataframe_score.to_excel(writer, sheet_name = "Points")
            dataframe_cabr.to_excel(writer, sheet_name = SHT_RCAB)
            dataframe_cab.to_excel(writer, sheet_name = SHT_CAB)
    for i in range(self.k):
        dataframe_k = pd.DataFrame(kexp[:, :, i], columns=header, index=index_ea)
        dataframe_k.to_csv(f'{direc}/{FOL_K}/{FIL_K}{i+1}.csv')
        dataframe_dmy = pd.DataFrame(dmy[:, :, i], columns = header, index = index_ea)
        dataframe_dmy.to_csv(f'{direc}/{FOL_DMY}/{FIL_DMY}{i+1}.csv')

def getDataFrameHeader(self: GEMObj):
    header = []
    for b in range(self.bmin, self.bmax+1):
        amin_b = np.maximum(np.ceil(self.abmin/b), self.amin).astype(int) - 1
        amax_b = np.minimum(np.floor(self.abmax/b), self.amax).astype(int)
        for a in range(amin_b+1,amax_b+1):
            header.append(f"({a};{b})")
    return header
def make_dir_if_new(name: str):
    if not os.path.exists(name):
        os.makedirs(name)