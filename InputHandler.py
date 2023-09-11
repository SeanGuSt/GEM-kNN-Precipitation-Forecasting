import pandas as pd
import numpy as np
import json
tryAgain_standard = "I'm sorry, please try again: "
def input_trainingFile():
    filepath = input("Please give the full path to your training results file: ")
    while True:
        try:
            print("One moment...")
            book = pd.read_excel(filepath, 'Chosen Pairs')
        except:
            filepath = input("File not found. Check that the path and filename are correct, and the sheet is named 'Chosen Pairs', then try again.")
            continue
        try:
            abPairs =  book.loc[:, "Chosen"]
            a = np.zeros((12, 1))
            b = np.zeros((12, 1))
            for i in range(12):
                abPair = abPairs[i].replace('(','').replace(')','').split(", ")
                a[i] = int(abPair[0])
                b[i] = int(abPair[1])
            return a, b
        except:
            filepath = input("Make sure there is a column named 'Chosen' in your 'Chosen Pairs' sheet, then try again.")
def input_defaultsFile():
    filepath = input("Please give the full path to your parameters file: ")
    while True:
        try:
            print("One moment...")
            with open(filepath) as f:
                data = f.read()
            return json.loads(data)
        except:
            filepath = input("File not found. Check that the path and filename are correct, then try again.\n(To find the full path, you can right click the file, then click Properties>Security>Object Name):  ")
            continue
def input_getFile():
    filepath = input("Please give the full path to your file: ")
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
def removeImpossibleDates(y,m,d):
    toRemove = []
    for i in range(len(d)):
        if (np.isin(m[i], [4, 6, 9, 11]) and d[i]==31) or (m[i]==2 and ((d[i]==29 and y[i]%4!=0) or d[i] > 29)):
            toRemove.append(i)
    return np.delete(y, toRemove), np.delete(m, toRemove), np.delete(d, toRemove)
def input_getInt(initialMessage, tryAgainMessage, intMin = 0, intMax = 0):
    clarification = ""
    if intMin != intMax:
        clarification = f"(Between {intMin} and {intMax}.): "
    userInput = input(initialMessage + clarification)
    while True:
        try:
            iui = int(userInput)
        except:
            userInput = input(tryAgainMessage)
        else:
            if iui >= intMin and iui <= intMax:
                return iui
            else:
                userInput = input(tryAgainMessage)
def input_getArray(initialMessage, tryAgainMessage, intMin, intMax):
    clarification = f"(Between {intMin} and {intMax}.): "
    userInput = input(initialMessage + clarification)
    while True:
        try: 
            newInput = np.array([int(i) for i in userInput.split()])
        except:
            userInput = input(tryAgainMessage)
        else:
            if all(x >= intMin and x <= intMax for x in np.abs(newInput)):
                if any(newInput<0):
                    if len(newInput) == 1:
                        return np.arange(1,-newInput[0]+1)
                    elif len(newInput) == 2:
                        return np.arange(-newInput[0],newInput[1]+1)
                    else:
                        userInput = input(tryAgainMessage)
                else:
                    return newInput
            else:
                userInput = input(tryAgainMessage)
def input_getYN(initialMessage, tryAgainMessage):
    accepted_answers = ["y", "n"]
    clarification = "\nPress Y or N: "
    userInput = input(initialMessage + clarification).lower()
    while userInput not in accepted_answers:
        userInput = input(tryAgainMessage).lower()
    return userInput
