import pandas as pd
import numpy as np
import json
tryAgain_standard = "I'm sorry, please try again: "
def input_defaultsFile():
    filepath = "C:/Users/scg2600\OneDrive - University of Texas at Arlington/Documents/Oklahoma Weather Station Data/defaultsTrain.txt"#input("Please give the full path to your parameters file: ")
    while True:
        try:
            print("One moment...")
            with open(filepath) as f:
                data = f.read()
            return json.loads(data)
        except:
            filepath = input("File not found, or invalid. Check that the path and filename are correct, then try again.\n(To find the full path, you can right click the file, then click Properties>Security>Object Name):  ")
            continue

def input_getFile():
    filepath = "C:/Users/scg2600/OneDrive - University of Texas at Arlington/Documents/Oklahoma Weather Station Data/Chandler daily 1902-2009.xlsx"#input("Please give the full path to your file: ")
    while True:
        try:
           print("One moment...")
           book = pd.read_excel(filepath)
           break
        except:
            filepath = input("File not found. Check that the path and filename are correct, then try again.\n(To find the full path, you can right click the file, then click Properties>Security>Object Name):  ")
    book = book.dropna()
    dates_data = book.to_numpy()
    if isinstance(dates_data[0,0], str):
        dates_data = np.delete(dates_data, 0, 0)
    dates = dates_data[:, :3].astype(int)
    data = dates_data[:, 3:]
    return filepath, dates, data

def removeImpossibleDates(y: list[int], m: list[int], d: list[int]):
    toRemove = []
    for i in range(len(d)):
        if (np.isin(m[i], [4, 6, 9, 11]) and d[i]==31) or (m[i]==2 and ((d[i]==29 and y[i]%4!=0) or d[i] > 29)):
            toRemove.append(i)
    return np.delete(y, toRemove), np.delete(m, toRemove), np.delete(d, toRemove)

def input_getResponse(initialMessage: str, acceptedAnswers: list[str], tryAgainMessage = "Not an accepted answer. Please try again."):
    clarification = "\n(Type "
    for i in range(len(acceptedAnswers)):
        if i < len(acceptedAnswers)-1:
            clarification = clarification + acceptedAnswers[i] + ", or "
        else:
            clarification = clarification + acceptedAnswers[i] + ", then press return)\n"
    userInput = input(initialMessage + clarification).lower()
    while userInput not in acceptedAnswers:
        userInput = input(tryAgainMessage + clarification).lower()
    return userInput
