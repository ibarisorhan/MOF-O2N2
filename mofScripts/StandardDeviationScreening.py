# -*- coding: utf-8 -*-
"""

@author: Baris
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import pathlib



cwd = str(pathlib.Path(__file__).parent.absolute()) #current working directory

mypath = cwd + "/Outputs/" 
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for ofc in onlyfiles:
    feat = ofc.split("_")[0]
    print(feat)
    file = mypath + ofc
    print(file)
    pred = pd.read_excel(file,sheet_name = "Predictions")
    act = pd.read_excel(file,sheet_name = "Actual")
    mofs = pd.read_excel(file,sheet_name = "TestIndicesAndMofs")
    name = cwd + "/MOFdata.csv"
    og  = pd.read_csv(name)
    
    count = 0 
    dd = {}
    for i in range(5):
        #5 split states
        count+=1
        aves = []
        for j in range(len(pred)):
            ent = sum(pred.iloc[j,(i*3)+1:(i*3)+4]) / 3
            aves.append(ent)
        dd["Ave_"+str(count)] = aves
    df = pd.DataFrame(dd)
    selected = []
    notsel = []
    for i in range(5):
        diff = df.iloc[:,i] - act.iloc[:,i+1]
        std = (sum(diff**2)/len(diff))**.5
        for j in range(len(diff)):
            if abs(diff[j]) <= 2*std:
                """
                SET STANDARD DEVIATION NUMBER IN LINE ABOVE
                """
                s = mofs.iloc[j,i+1]
                selected.append(s)
            else:
                notsel.append(s)
    otp = og[og["MOFRefcodes"].isin(selected)]
    new_path = cwd + "/2STD/" + feat
    otp.to_excel(new_path+".xlsx")
