# -*- coding: utf-8 -*-
"""


          _____                   _______                   _____          
         /\    \                 /::\    \                 /\    \         
        /::\____\               /::::\    \               /::\    \        
       /::::|   |              /::::::\    \             /::::\    \       
      /:::::|   |             /::::::::\    \           /::::::\    \      
     /::::::|   |            /:::/~~\:::\    \         /:::/\:::\    \     
    /:::/|::|   |           /:::/    \:::\    \       /:::/__\:::\    \    
   /:::/ |::|   |          /:::/    / \:::\    \     /::::\   \:::\    \   
  /:::/  |::|___|______   /:::/____/   \:::\____\   /::::::\   \:::\    \  
 /:::/   |::::::::\    \ |:::|    |     |:::|    | /:::/\:::\   \:::\    \ 
/:::/    |:::::::::\____\|:::|____|     |:::|    |/:::/  \:::\   \:::\____\
\::/    / ~~~~~/:::/    / \:::\    \   /:::/    / \::/    \:::\   \::/    /
 \/____/      /:::/    /   \:::\    \ /:::/    /   \/____/ \:::\   \/____/ 
             /:::/    /     \:::\    /:::/    /             \:::\    \     
            /:::/    /       \:::\__/:::/    /               \:::\____\    
           /:::/    /         \::::::::/    /                 \::/    /    
          /:::/    /           \::::::/    /                   \/____/     
         /:::/    /             \::::/    /                                
        /:::/    /               \::/____/                                 
        \::/    /                 ~~                                       
         \/____/                                                           
                                                                           
                                   
                                                           

  ____               _         _   
 / ___|   ___  _ __ (_) _ __  | |_ 
 \___ \  / __|| '__|| || '_ \ | __|
  ___) || (__ | |   | || |_) || |_ 
 |____/  \___||_|   |_|| .__/  \__|
                       |_| 
                       

Note: Does not include if name == '__main__' statement

"""




#----   Importing Necessary Libraries   ----
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from pandas import ExcelWriter
import time
import random
import pathlib


#----   Input Parameters   ----
feature_iteration_list = ['O2uptakemolkg','N2uptakemolkg',  'SelfdiffusionofO2cm2s', 'SelfdiffusionofN2cm2s',  'HenrysconstantO2', 'HenrysconstantN2', 'SelfdiffusionofO2cm2sinfDilute', 'SelfdiffusionofN2cm2sinfDilute']

feature_thresholds = {'O2uptakemolkg': 1.2,
 'N2uptakemolkg': 0.98,
 'SelfdiffusionofO2cm2s' : 0.0006,
 'SelfdiffusionofN2cm2s' : 0.0005,
 'HenrysconstantO2' : 1* (10**-5),
 'HenrysconstantN2': (7.9 * (10**-6)),
 'SelfdiffusionofO2cm2sinfDilute': 0.0005,
 'SelfdiffusionofN2cm2sinfDilute': 0.0006}



for feat_loop_index in range(len(feature_iteration_list)):
    Features_to_predict = [feature_iteration_list[feat_loop_index]]
    print("\n\n\n\n\n----------------------------\n{}\n----------------------------".format(feature_iteration_list[feat_loop_index]))
    outputfilename = Features_to_predict[0] +"_Thresholded"
    inputfilename = "MOFdata.csv"
    
    cwd = str(pathlib.Path(__file__).parent.absolute())
    input_file = cwd  + "/"+inputfilename
    

    
    tt = time.ctime().split(" ")
    tt2 = tt[3].split(":")
    hms = tt[1] + tt[2] +"_" +tt2[0] + tt2[1] + tt2[2]
    dateTimeExt =  hms
    
    output_file = cwd + "/Outputs/"+ outputfilename + "_" + dateTimeExt + ".xlsx"
    backup_output_directory = cwd+ "/Outputs/Backup/"
    regression_type = "Random Forests"

    
    Notes= "Standard Scaler was used for all predictions.In this file the log transform was NOT applied to any columns."
        
    sheet_names_of_output = ["R2Averages", "R2Expanded", "Predictions", "Actual", "ABOUT","TestIndicesAndMofs", "TrainPreds", "TrainActual"]
    xstart_index, xend_index = 0,-9
    ystart_index = -9
    
    Geometric_features = ['LCD', 'PLD', 'LFPD', 'Volume', 'ASA_m2_g','ASA_m2_cm3', 'NASA_m2_g', 'NASA_m2_cm3', 'AV_VF', 'AV_cm3_g','NAV_cm3_g']
    
    Features_From_New = ["AV_VF","ASA_m2_cm3","ASA_m2_g","PLD","LCD"]
    
    Non_repeating_features = ['LCD', 'PLD', 'Volume', 'ASA_m2_g','NASA_m2_g','AV_VF', 'AV_cm3_g', 'NAV_cm3_g']
    
    FeatsFromPaper = [' H', 'C', 'N', 'F', 'Cl', 'Br', 'V', 'Cu', 'Zn', 'Zr','metal type', ' total degree of unsaturation', 'metalic percentage',' oxygetn-to-metal ratio', 'electronegtive-to-total ratio',' weighted electronegativity per atom', ' nitrogen to oxygen ']
    
    combination =  FeatsFromPaper + Non_repeating_features
    
    number_of_estimators = 100
    regression_states = [random.randint(0, 1000000) for i in range(3)]
    train_ratio, test_ratio = 0.8, 0.2
    
    print(combination)
    
    
    #----   Setting Up Dataframes   ----
    dfcolumns = combination +  ["MOFRefcodes"]
    df2 = pd.read_csv(input_file)
    temp = [ i for i in df2.columns]
    for i in temp:
        if i != "MOFRefcodes":
            df2[i] = df2[i].astype(float)
    THRESHOLD = feature_thresholds[feature_iteration_list[feat_loop_index]]
    df = df2[df2[feature_iteration_list[feat_loop_index]] <=THRESHOLD]
    X = df[dfcolumns]
    Y = df[feature_iteration_list]
    
    
    
    #----   Train Test Splits (Changed From Original)   ----
    """
    Since Sklearn's K-Fold CrossValidation ends up with varying test set sizes, to keep uniformity the train-test
    splits were manually done with potentially small ovelaps in the final two groups
    """
    num_entries = len(X)
    entry_indices = list(range(num_entries))
    entries_per_split = int(num_entries/5)
    
    if num_entries%5 >0:
        entries_per_split +=1
    print(entries_per_split)
    split1, split2 ,split3, split4, split5 = entry_indices[:entries_per_split],entry_indices[entries_per_split:entries_per_split*2],entry_indices[entries_per_split*2:entries_per_split*3],entry_indices[entries_per_split*3:entries_per_split*4],entry_indices[-entries_per_split:]
    split_states =[split1, split2 ,split3, split4, split5]
    
    
    
    
    #----   Defining Functions   ----
    def CorrCoefficient(X, Y):
        """Take X and Y lists and return the Pearson Coefficient."""
        r=np.sum((X-np.average(X))*(Y-np.average(Y)))/math.sqrt(np.sum((X-np.average(X))**2)*np.sum((Y-np.average(Y))**2))
        calc = r**2
        return calc
    
    def test_combo(xr,xe,yr_current,ye_current, average, standard_deviation, selected_combo,reg):
        print("Number of Estimators: ",reg.n_estimators, "    RandomState: " ,reg.random_state)
        ave, dev = average, standard_deviation
        combination = selected_combo
        xr_ss, xe_ss =xr[combination],xe[combination]
        xr_current, xe_current = ssx.fit_transform(xr_ss), ssx.transform(xe_ss)
        reg.fit(xr_current,yr_current.ravel())
        prediction = reg.predict(xe_current)
        Predictions = []
        for pred in prediction:
            t = (pred * dev) + ave
            Predictions.append(t)
        return Predictions
    
    
    
    
    #----   Initialising Dictionaries & Main Loop   ----
    ssx, ssy = StandardScaler(), StandardScaler()
    PredictionDictionary = {}
    ActualValuesDictionary  = {}
    TrainDict = {}
    TrainPred  = {}
    R2Dictionary = {}
    MofAndIndexLibrary  = {}
    TrainMofLibrary = {}
    CombinationR2Dictionary = {}
    TrainR2 = []
    split_state_count = 0
    for ycolumn in Features_to_predict:
        CombinationR2Dictionary[ycolumn] = []
        
        for split_state in split_states:
            test_split = split_state
            train_split = list(range(num_entries))
            for test_index in test_split:
                train_split.remove(test_index)
            xr,xe,yr,ye = X.iloc[train_split,:],X.iloc[test_split,:],Y.iloc[train_split,:],Y.iloc[test_split,:]
            
            
            split_state_count += 1
            for regression_state in regression_states:
                selected_target = ycolumn
                print('\n\n-----  Split State: {:10} Regression State: {:10}  -----'.format(split_state_count, regression_state))
                rf = RandomForestRegressor(random_state= regression_state, n_estimators = number_of_estimators)
                
                
                #----   Formating Target Variable   ----
                yr_ss, ye_ss = np.array(yr[selected_target]).reshape(-1,1),np.array(ye[selected_target]).reshape(-1,1)
                yr_current, ye_current = ssy.fit_transform(yr_ss), ssy.transform(ye_ss)
                yr_current, ye_current = yr_current.reshape(-1,1), ye_current.reshape(-1,1)
                yr_unscaled, ye_unscaled = np.array(yr[selected_target]).reshape(-1,1), np.array(ye[selected_target]).reshape(-1,1)
                ave, dev  = np.average(yr_unscaled), np.std(yr_unscaled)
                
                
                #----   Testing Combination 1   ----
                print()
                selection = combination
                name = str(selected_target) + "_Combination_" + str(split_state_count) + "_" + str(regression_state)
                print(name)
                print("Length of Combination: ",len(selection))
                
                Predictions= test_combo(xr, xe, yr_current, ye_current, ave, dev, selection,rf)
                correlation = CorrCoefficient(Predictions, ye_unscaled.reshape(1,-1))
                PredictionDictionary[name] = Predictions
                R2Dictionary[name] = [correlation]
                CombinationR2Dictionary[ycolumn].append(correlation)
                print('Number of Features Tested: {:5} R2 Correlation: {:5}'.format(len(selection), correlation))
                
                
                #----   Writing Actual Values to Dictionary   ----
                ActualName = str(selected_target)  +  "_Actual_" + str(split_state_count) 
                ActualValuesDictionary[ActualName] = ye_unscaled.reshape(1,-1)[0]
    
    
                #----   Saving Test Indices and Mofs   ----
                MAILkey1 = "Actual_" + str(split_state_count) +"_MOFcode"
                MofAndIndexLibrary[MAILkey1] = [mrfc for mrfc in xe["MOFRefcodes"]]
                TMkey =  "Train_" + str(split_state_count) +"_MOFcode"
                TrainMofLibrary[TMkey] = [mrfc for mrfc in xr["MOFRefcodes"]]
                
                
                #----   Train Predictions    ----
                print()
                selection = combination
                name = str(selected_target) + "_Combination_" + str(split_state_count) + "_" + str(regression_state)            
                Predictions= test_combo(xr, xr, yr_current, yr_current, ave, dev, selection,rf)
                correlation = CorrCoefficient(Predictions, yr_unscaled.reshape(1,-1))
                TrainPred[name] = Predictions
                TrainR2.append(correlation)
                print('Train R2 Correlation: {:5}'.format(correlation))
                
                
                #----   Writing Train Values to Dictionary   ----
                ActualName = str(selected_target)  +  "_Actual_" + str(split_state_count) 
                TrainDict[ActualName] = yr_unscaled.reshape(1,-1)[0]
    
                
                
    #----   Getting Average R2 values into Single Dictionary   ----
    AveragedR2Dictionary = {"Feature":[],"Combination": []}
    for rowname in Features_to_predict:
        AveragedR2Dictionary["Feature"].append(rowname)
        AveragedR2Dictionary["Combination"].append(np.mean(CombinationR2Dictionary[rowname]))
    
    
    
    #----   Creating Hyperparameters Dictionary   ----
    HP = {
          "Input File: ": [input_file],
          "Output File: " : [output_file],
          "Regression Type: ": [regression_type],
          "Descriptors: ": [[j for j in X.columns]],
          "Combination: ": [combination],
          "Target Variables: ": [Features_to_predict],
          "Number of Estimators: ": [number_of_estimators],
          "Split States :": [str(split_states)],
          "Regression Random States: ": [str(regression_states)],
          "Train Ratio: ":[train_ratio],
          "Test Ratio: ": [test_ratio],
          "Number of Entries: ":[len(X)],
          "Train R2s:":[str(TrainR2)],
          "NOTES: ": [Notes]
          }
    
    

            
    #----   Converting Dictionaries to Dataframes and Writing Output   ----
    backup_data = [AveragedR2Dictionary, R2Dictionary, PredictionDictionary, ActualValuesDictionary, HP, MofAndIndexLibrary]
    
    for i in range(len(backup_data)):
        backup_output_directory = cwd+ "/Outputs/Backup/"+ sheet_names_of_output[i]+"_"+ dateTimeExt+".txt"
        backup_file = open(backup_output_directory,"a+")
        backup_file.write(str(backup_data[i]))
        backup_file.close()
    
    HPDF = pd.DataFrame(HP)
    HPDF = HPDF.set_index("Input File: ")
    HPDF = HPDF.transpose()
    
    AveR2 = pd.DataFrame(AveragedR2Dictionary)
    AveR2= AveR2.set_index("Feature")
    AveR2 = AveR2.transpose()
    
    Expanded_R2 = pd.DataFrame(R2Dictionary)
    Expanded_R2 = Expanded_R2.transpose()
    PredictionsDF = pd.DataFrame(PredictionDictionary)
    ActualValuesDF = pd.DataFrame(ActualValuesDictionary)
    
    MAILdf = pd.DataFrame(MofAndIndexLibrary)
    TrainMofs = pd.DataFrame(TrainMofLibrary)
    TD = pd.DataFrame(TrainDict)
    TP = pd.DataFrame(TrainPred)
    
    
    with ExcelWriter(output_file) as writer:
        AveR2.to_excel(writer, sheet_name = sheet_names_of_output [0])
        Expanded_R2.to_excel(writer, sheet_name = sheet_names_of_output [1])
        PredictionsDF.to_excel(writer, sheet_name = sheet_names_of_output [2])
        ActualValuesDF.to_excel(writer, sheet_name = sheet_names_of_output [3])
        HPDF.to_excel(writer, sheet_name = sheet_names_of_output [4])
        MAILdf.to_excel(writer,sheet_name = sheet_names_of_output [5])
        TP.to_excel(writer,sheet_name = sheet_names_of_output[6])
        TD.to_excel(writer,sheet_name = sheet_names_of_output[7])
        TrainMofs.to_excel(writer,sheet_name = "TrainMOFs")
    
 
print("\n\nDone!")
    
    

