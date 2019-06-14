# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 05:58:18 2018
this file creates subsets of features

1-4) n-gram precision , n= 1,2,3,4
5-8) n-gram recal,      n= 1,2,3,4
9-12) n-gram f-measure, n=1,2,3,4
13)maximum ration of hypothesis length over the set of reference captions( realted to brevity penalty)
14) Rogue-L score
15) Meteor score
16) WMD score
17-20) Blue 1,2,3,4 scores
21) SPICE score
22)cider

@author: 22161668
"""

import numpy as np
import json
import os

path='D:/NNeval_classification/data/all_features'
data_path='D:/NNeval_classification/data'


def subset(features):
    
    ImgId=features.keys()
    new={}
    for ids in ImgId:
        
         new[ids]=[
                   features[ids][0], #1-gram precision
                   features[ids][1], #2-gram precision
                   features[ids][2], #3-gram precision
                   features[ids][3], #4-gram precision
###         
                   features[ids][4], #1-gram recal
#                   features[ids][5], #2-gram recal
#                   features[ids][6], #3-gram recal
 #                  features[ids][7], #4-gram recal
#         
#                   features[ids][8], #1-gram f-measure
#                   features[ids][9], #2-gram f-measure
#                   features[ids][10], #3-gram f-measure
#                   features[ids][11], #4-gram f-measure
         
#                   features[ids][12], # length ratio
                   features[ids][13], #Rogue-L
                   features[ids][14], # Meteor
                   features[ids][15], # wmd 
#                   features[ids][16], # blue1
#                   features[ids][17], # blue2
#                   features[ids][18], # blue3
#                   features[ids][19], # blue4
                   features[ids][20], # SPICE
                   features[ids][21]/10,# CIDER
                   
#                  ((features[ids][22])+1)*0.5, # MOWE W2V
                  ((features[ids][23])+1)*0.5, # MOWE_FAST
                   
#                   features[ids][24], # DP_Text
#                   features[ids][25], # DP_lemma
#                   features[ids][26], # DP_dep # average overlap between terminal nodes according to grammatical relationship
#                   features[ids][27], # DP_pos # average overlap between terminal nodes according to grammatical category
##                   
#                   features[ids][28], # HWCM_TEXT1
#                   features[ids][29], # HWCM_TEXT2
#                   features[ids][30], # HWCM_TEXT3
#                   features[ids][31], # HWCM_TEXT4
#                   
#                   features[ids][32], # HWCM_lemma1
                  features[ids][33], # HWCM_lemma2
                  features[ids][34], # HWCM_lemma3
                   features[ids][35], # HWCM_lemma4              
#####                   
##                    features[ids][36], # HWCM_DEP1
                    features[ids][37], # HWCM_DEP2
                   features[ids][38] # HWCM_DEP3                  
#                   features[ids][39] # HWCM_DEP4
###                   
#                   features[ids][40], # HWCM_pos1
#                 features[ids][41] # HWCM_pos2
#                  features[ids][42] # HWCM_pos3
#                 features[ids][43] # HWCM_pos4
#                ((features[ids][44])+1)*0.5 # MOWE_glove
###  -----------------------------------mean-------------------------------
                   
#                   features[ids][45+0], #1-gram precision
#                   features[ids][45+1], #2-gram precision
#                   features[ids][45+2], #3-gram precision
#                   features[ids][45+3], #4-gram precision
###         
#                   features[ids][45+4], #1-gram recal
#                   features[ids][45+5], #2-gram recal
#                   features[ids][45+6], #3-gram recal
 #                  features[ids][45+7], #4-gram recal
#         
#                   features[ids][45+8], #1-gram f-measure
#                   features[ids][45+9], #2-gram f-measure
#                   features[ids][45+10], #3-gram f-measure
#                   features[ids][45+11], #4-gram f-measure
         
#                   features[ids][45+12], # length ratio
#                   features[ids][45+13], #Rogue-L
#                    features[ids][45+14], # Meteor
#                   features[ids][45+15], # wmd 
#                   features[ids][45+16], # blue1
#                   features[ids][45+17], # blue2
#                   features[ids][45+18], # blue3
#                   features[ids][45+19], # blue4
#                   features[ids][45+20], # SPICE
#                   features[ids][45+21]/10,# CIDER
                   
#                   ((features[ids][45+22])+1)*0.5, # MOWE
#                  ((features[ids][45+23])+1)*0.5, # MOWE_FAST
                   
#                   features[ids][45+24], # DP_Text
#                   features[ids][45+25], # DP_lemma
#                   features[ids][45+26], # DP_dep # average overlap between terminal nodes according to grammatical relationship
#                   features[ids][45+27], # DP_pos # average overlap between terminal nodes according to grammatical category
##                   
#                   features[ids][45+28], # HWCM_TEXT1
#                   features[ids][45+29], # HWCM_TEXT2
#                   features[ids][45+30], # HWCM_TEXT3
#                   features[ids][45+31], # HWCM_TEXT4
#                   
#                   features[ids][45+32], # HWCM_lemma1
#                  features[ids][45+33], # HWCM_lemma2
#                  features[ids][45+34], # HWCM_lemma3
#                   features[ids][45+35], # HWCM_lemma4              
#####                   
##                    features[ids][45+36], # HWCM_DEP1
#                    features[ids][45+37], # HWCM_DEP2
#                   features[ids][45+38] # HWCM_DEP3                  
#                   features[ids][45+39] # HWCM_DEP4
###                   
#                   features[ids][45+40], # HWCM_pos1
#                  features[ids][45+41], # HWCM_pos2
#                  features[ids][45+42], # HWCM_pos3
#                 features[ids][45+43] # HWCM_pos4
#                ((features[ids][89])+1)*0.5, # MOWE_glove
#-----------------------------------min -----------------------
#                   features[ids][90+0], #1-gram precision
#                   features[ids][90+1], #2-gram precision
#                   features[ids][90+2], #3-gram precision
#                   features[ids][90+3], #4-gram precision
###         
#                   features[ids][90+4], #1-gram recal
#                   features[ids][90+5], #2-gram recal
#                   features[ids][90+6], #3-gram recal
 #                  features[ids][90+7], #4-gram recal
#         
#                   features[ids][90+8], #1-gram f-measure
#                   features[ids][90+9], #2-gram f-measure
#                   features[ids][90+10], #3-gram f-measure
#                   features[ids][90+11], #4-gram f-measure
         
#                   features[ids][90+12], # length ratio
#                   features[ids][90+13], #Rogue-L
#                    features[ids][90+14], # Meteor
#                   features[ids][90+15], # wmd 
#                   features[ids][90+16], # blue1
#                   features[ids][90+17], # blue2
#                   features[ids][90+18], # blue3
#                   features[ids][90+19], # blue4
#                   features[ids][90+20], # SPICE
#                   features[ids][90+21]/10,# CIDER
                   
#                   ((features[ids][90+22])+1)*0.5, # MOWE
#                  ((features[ids][90+23])+1)*0.5, # MOWE_FAST
                   
#                   features[ids][90+24], # DP_Text
#                   features[ids][90+25], # DP_lemma
#                   features[ids][90+26], # DP_dep # average overlap between terminal nodes according to grammatical relationship
#                   features[ids][90+27], # DP_pos # average overlap between terminal nodes according to grammatical category
##                   
#                   features[ids][90+28], # HWCM_TEXT1
#                   features[ids][90+29], # HWCM_TEXT2
#                   features[ids][90+30], # HWCM_TEXT3
#                   features[ids][90+31], # HWCM_TEXT4
#                   
#                   features[ids][90+32], # HWCM_lemma1
#                  features[ids][90+33], # HWCM_lemma2
#                  features[ids][90+34], # HWCM_lemma3
#                   features[ids][90+35], # HWCM_lemma4              
#####                   
##                    features[ids][90+36], # HWCM_DEP1
#                    features[ids][90+37], # HWCM_DEP2
#                   features[ids][90+38] # HWCM_DEP3                  
#                   features[ids][90+39] # HWCM_DEP4
###                   
#                   features[ids][90+40], # HWCM_pos1
#                  features[ids][90+41], # HWCM_pos2
#                  features[ids][90+42], # HWCM_pos3
#                 features[ids][90+43] # HWCM_pos4]
                                    ]

         
    return new
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#    
with open (os.path.join(path, "f8k_features.json")) as f:
    f8k=json.load(f)
    
    for i in f8k: 
        if any(np.isnan(f8k[i])): 
            f8k[i]=list(np.nan_to_num(np.array(f8k[i])))
            
    f8k=subset(f8k)
    
with open (os.path.join(data_path, "test/f8k_features.json"),'w') as f:
    json.dump(f8k,f)
##""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
#
with open (os.path.join(path, "compcoco_features.json")) as f:
    compcoco=json.load(f)

    for i in compcoco: 
        if any(np.isnan(compcoco[i])): 
            compcoco[i]=list(np.nan_to_num(np.array(compcoco[i])))

    compcoco=subset(compcoco)

with open (os.path.join(data_path, "test/compcoco_features.json"),'w') as f:
    json.dump(compcoco,f)
#
##""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
with open (os.path.join(path, "test_pascal_features1.json")) as f:
    test_pascal1=json.load(f)

    for i in test_pascal1: 
        if any(np.isnan(test_pascal1[i])): 
            test_pascal1[i]=list(np.nan_to_num(np.array(test_pascal1[i])))
            
    test_pascal1=subset(test_pascal1)

with open (os.path.join(data_path, "test_pascal/test_pascal_features1.json"),'w') as f:
    json.dump(test_pascal1,f)
#
##""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
with open (os.path.join(path, "test_pascal_features2.json")) as f:
    test_pascal2=json.load(f)

    for i in test_pascal2: 
        if any(np.isnan(test_pascal2[i])): 
            test_pascal2[i]=list(np.nan_to_num(np.array(test_pascal2[i])))
            
    test_pascal2=subset(test_pascal2)
    
with open (os.path.join(data_path, "test_pascal/test_pascal_features2.json"),'w') as f:
    json.dump(test_pascal2,f)
# #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
with open (os.path.join(path, "test_abstract_features1.json")) as f:
    test_abstract1=json.load(f)

    for i in test_abstract1: 
        if any(np.isnan(test_abstract1[i])): 
            test_abstract1[i]=list(np.nan_to_num(np.array(test_abstract1[i])))
            
    test_abstract1=subset(test_abstract1)

with open (os.path.join(data_path, "test_abstract/test_abstract_features1.json"),'w') as f:
    json.dump(test_abstract1,f)
#
##""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
with open (os.path.join(path, "test_abstract_features2.json")) as f:
    test_abstract2=json.load(f)

    for i in test_abstract2: 
        if any(np.isnan(test_abstract2[i])): 
            test_abstract2[i]=list(np.nan_to_num(np.array(test_abstract2[i])))
            
    test_abstract2=subset(test_abstract2)
    
with open (os.path.join(data_path, "test_abstract/test_abstract_features2.json"),'w') as f:
    json.dump(test_abstract2,f)    
##""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""   
##""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
with open (os.path.join(path, "train_features.json")) as f:
    train=json.load(f) 
    
    for i in train: 
        if any(np.isnan(train[i])): 
            train[i]=list(np.nan_to_num(np.array(train[i])))
            
    train=subset(train)
    
with open (os.path.join(data_path, "train/train_features.json"),'w') as f:
    json.dump(train,f)
##""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#        
with open (os.path.join(path, "val_features.json")) as f:
    val=json.load(f) 
    
    for i in val: 
        if any(np.isnan(val[i])): 
            val[i]=list(np.nan_to_num(np.array(val[i])))
    val=subset(val)
    
with open (os.path.join(data_path, "val/val_features.json"),'w') as f:
    json.dump(val,f)
##"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
#    
with open (os.path.join(path, "valcomp_full_features.json")) as f:
    valcompfull=json.load(f)
    
    for i in valcompfull: 
        if any(np.isnan(valcompfull[i])): 
            valcompfull[i]=list(np.nan_to_num(np.array(valcompfull[i])))
            
    valcompfull=subset(valcompfull)
    
with open (os.path.join(data_path, "valcomp_full/valcomp_full_features.json"),'w') as f:
    json.dump(valcompfull,f)
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/replace-person_features_c.json")) as f:
    replacepersonc=json.load(f) 

    for i in replacepersonc: 
        if any(np.isnan(replacepersonc[i])): 
            replacepersonc[i]=list(np.nan_to_num(np.array(replacepersonc[i])))
            
    replacepersonc=subset(replacepersonc)
    
with open (os.path.join(hodosh_path, "replace-person/replace-person_features_c.json"),'w') as f:
    json.dump(replacepersonc,f)
#    
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/replace-person_features_d.json")) as f:
    replacepersond=json.load(f) 

    for i in replacepersond: 
        if any(np.isnan(replacepersond[i])): 
            replacepersond[i]=list(np.nan_to_num(np.array(replacepersond[i])))
            
    replacepersond=subset(replacepersond)
    
with open (os.path.join(hodosh_path, "replace-person/replace-person_features_d.json"),'w') as f:
    json.dump(replacepersond,f)
#
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/replace-scene_features_c.json")) as f:
    replacescenec=json.load(f) 

    for i in replacescenec: 
        if any(np.isnan(replacescenec[i])): 
            replacescenec[i]=list(np.nan_to_num(np.array(replacescenec[i])))

    replacescenec=subset(replacescenec)
    
with open (os.path.join(hodosh_path, "replace-scene/replace-scene_features_c.json"),'w') as f:
    json.dump(replacescenec,f)
#
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/replace-scene_features_d.json")) as f:
    replacescened=json.load(f) 

    for i in replacescened: 
        if any(np.isnan(replacescened[i])): 
            replacescened[i]=list(np.nan_to_num(np.array(replacescened[i])))

    replacescened=subset(replacescened)
    
with open (os.path.join(hodosh_path, "replace-scene/replace-scene_features_d.json"),'w') as f:
    json.dump(replacescened,f)
#
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/share-person_features_c.json")) as f:
    sharepersonc=json.load(f) 

    for i in sharepersonc: 
        if any(np.isnan(sharepersonc[i])): 
            sharepersonc[i]=list(np.nan_to_num(np.array(sharepersonc[i])))

    sharepersonc=subset(sharepersonc)
    
with open (os.path.join(hodosh_path, "share-person/share-person_features_c.json"),'w') as f:
    json.dump(sharepersonc,f)
#
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/share-person_features_d.json")) as f:
    sharepersond=json.load(f) 
    
    for i in sharepersond: 
        if any(np.isnan(sharepersond[i])): 
            sharepersond[i]=list(np.nan_to_num(np.array(sharepersond[i])))

    sharepersond=subset(sharepersond)
    
with open (os.path.join(hodosh_path, "share-person/share-person_features_d.json"),'w') as f:
    json.dump(sharepersond,f)    
#
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/share-scene_features_c.json")) as f:
    sharescenec=json.load(f) 

    for i in sharescenec: 
        if any(np.isnan(sharescenec[i])): 
            sharescenec[i]=list(np.nan_to_num(np.array(sharescenec[i])))

    sharescenec=subset(sharescenec)
    
with open (os.path.join(hodosh_path, "share-scene/share-scene_features_c.json"),'w') as f:
    json.dump(sharescenec,f)
#
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/share-scene_features_d.json")) as f:
    sharescened=json.load(f)
    
    for i in sharescened: 
        if any(np.isnan(sharescened[i])): 
            sharescened[i]=list(np.nan_to_num(np.array(sharescened[i])))

    sharescened=subset(sharescened)
    
with open (os.path.join(hodosh_path, "share-scene/share-scene_features_d.json"),'w') as f:
    json.dump(sharescened,f)
#    
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/just-person_features_c.json")) as f:
    justpersonc=json.load(f) 
    
    for i in justpersonc: 
        if any(np.isnan(justpersonc[i])): 
            justpersonc[i]=list(np.nan_to_num(np.array(justpersonc[i])))

    justpersonc=subset(justpersonc)
    
with open (os.path.join(hodosh_path, "just-person/just-person_features_c.json"),'w') as f:
    json.dump(justpersonc,f)
#    
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/just-person_features_d.json")) as f:
    justpersond=json.load(f) 

    for i in justpersond: 
        if any(np.isnan(justpersond[i])): 
            justpersond[i]=list(np.nan_to_num(np.array(justpersond[i])))

    justpersond=subset(justpersond)
    
with open (os.path.join(hodosh_path, "just-person/just-person_features_d.json"),'w') as f:
    json.dump(justpersond,f)
#
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/just-scene_features_c.json")) as f:
    justscenec=json.load(f) 
    
    for i in justscenec: 
        if any(np.isnan(justscenec[i])): 
            justscenec[i]=list(np.nan_to_num(np.array(justscenec[i])))
    
    justscenec=subset(justscenec)
    
with open (os.path.join(hodosh_path, "just-scene/just-scene_features_c.json"),'w') as f:
    json.dump(justscenec,f)
#    
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/just-scene_features_d.json")) as f:
    justscened=json.load(f) 

    for i in justscened: 
        if any(np.isnan(justscened[i])): 
            justscened[i]=list(np.nan_to_num(np.array(justscened[i])))
  
    justscened=subset(justscened)
    
with open (os.path.join(hodosh_path, "just-scene/just-scene_features_d.json"),'w') as f:
    json.dump(justscened,f)
#
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
##['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'
#
hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/switch-people_features_c.json")) as f:
    switchpeoplec=json.load(f) 

    for i in switchpeoplec: 
        if any(np.isnan(switchpeoplec[i])): 
            switchpeoplec[i]=list(np.nan_to_num(np.array(switchpeoplec[i])))
  
    switchpeoplec=subset(switchpeoplec)
    
with open (os.path.join(hodosh_path, "switch-people/switch-people_features_c.json"),'w') as f:
    json.dump(switchpeoplec,f)
    
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#['replace-person', 'replace-scene', 'share-person', 'share-scene','just-person','just-scene','switch-people'

hodosh_path='D:/NNeval_classification/data/Hodosh_bt/'

with open (os.path.join(path, "hod/switch-people_features_d.json")) as f:
    switchpeopled=json.load(f) 

    for i in switchpeopled: 
        if any(np.isnan(switchpeopled[i])): 
            switchpeopled[i]=list(np.nan_to_num(np.array(switchpeopled[i])))
 
    switchpeopled=subset(switchpeopled)
    
with open (os.path.join(hodosh_path, "switch-people/switch-people_features_d.json"),'w') as f:
    json.dump(switchpeopled,f)


#----------------------------------------------------------------------------------------------

opi=['CS-EN-','DE-EN-','FI-EN-','RO-EN-','RU-EN-','TR-EN-']

MT_path='D:/NNeval_classification/data/MT'

for split in opi:
    with open (os.path.join(path, "MT/%sfeatures.json" %(split))) as f:
        mt=json.load(f) 

    for i in mt: 
        if any(np.isnan(mt[i])): 
            mt[i]=list(np.nan_to_num(np.array(mt[i])))
  
    mt=subset(mt)
    
    with open (os.path.join(MT_path, "%sfeatures.json" %(split)),'w') as f:
        json.dump(mt,f)   
        
        
#-----------------------------------------------------------------------------
        
opi=['CS-EN-','DE-EN-','FI-EN-','RU-EN-']

MT_path='D:/NNeval_classification/data/MT2'

for split in opi:
    with open (os.path.join(path, "MT2/%sfeatures.json" %(split))) as f:
        mt=json.load(f) 

    for i in mt: 
        if any(np.isnan(mt[i])): 
            mt[i]=list(np.nan_to_num(np.array(mt[i])))
  
    mt=subset(mt)
    
    with open (os.path.join(MT_path, "%sfeatures.json" %(split)),'w') as f:
        json.dump(mt,f)  
        
        
#---------------------------
        
system_path='D:/NNeval_classification/data/system'
humans='E:/coco_captioning_challenge/human/captions_val2014_rand_evalimgs.json'
jeffdonahue='E:/coco_captioning_challenge/jeffdonahue/captions_val2014_LRCN-2F-VGGNET-BEAM1_results.json'
junhua='E:/coco_captioning_challenge/junhua.mao/captions_val2014_rb_results.json'
karpathy='E:/coco_captioning_challenge/karpathy/captions_val2014_LSTM_results.json'
kelvin_xu='E:/coco_captioning_challenge/kelvin_xu/captions_val2014_attention_results.json'
kolarmartin='E:/coco_captioning_challenge/kolarmartin/captions_val2014_COSINE_results.json'
mmitchell='E:/coco_captioning_challenge/mmitchell/captions_val2014_oldVGG_results.json'
mRNN_share_JMao='E:/coco_captioning_challenge/mRNN_share.JMao/captions_val2014_05-29-10_results.json'
MSR_Captivator='E:/coco_captioning_challenge/MSR_Captivator/captions_val2014_d-me.dmsm.grnn.fc6.fc7.cider.pos.600best_results.json'
myamaguchi='E:/coco_captioning_challenge/myamaguchi/captions_val2014_bigramsearch_results.json'
NearestNeighbor='E:/coco_captioning_challenge/NearestNeighbor/captions_val2014_knncider_results.json'
OriolVinyals='E:/coco_captioning_challenge/OriolVinyals/captions_val2014_googlstm_results.json'
Q_Wu='E:/coco_captioning_challenge/Q.Wu/captions_val2014_attributes_results.json'
rakshithShetty='E:/coco_captioning_challenge/rakshithShetty/captions_val2014_CMME_results.json'
ryank='E:/coco_captioning_challenge/ryank/captions_val2014_MLBL_results.json'
TsinghuaBigeye='E:/coco_captioning_challenge/TsinghuaBigeye/captions_val2014_6717_results.json'

names=[('jeffdonahue',jeffdonahue),('junhua',junhua),('karpathy',karpathy),
    ('kolarmartin',kolarmartin),('mmitchell',mmitchell),('mRNN_share_JMao',mRNN_share_JMao),
    ('myamaguchi',myamaguchi),('NearestNeighbor',NearestNeighbor),('kelvin_xu',kelvin_xu),
    ('OriolVinyals',OriolVinyals),('Q_Wu',Q_Wu),('rakshithShetty',rakshithShetty),
    ('ryank',ryank),('TsinghuaBigeye',TsinghuaBigeye)]

for (name,_) in names:
    
    feature_path=os.path.join(path, "system/%s_features.json" %(name)) 
    with open(feature_path, 'r') as f:
            feature = json.load(f)
            
            for i in feature: 
                if any(np.isnan(feature[i])): 
                    feature[i]=list(np.nan_to_num(np.array(feature[i])))
            
            feature=subset(feature)
    
    with open (os.path.join(system_path, "%s_features.json" %(name)),'w') as f:
        json.dump(feature,f) 
        
        
#---------------------------------------------------------------------------------------------
#========================================================================================---------------------------------------------------------------------------
cc=[10,15,20,25,30,35,40,45]
for sp in cc:        
    with open (os.path.join(path, str(sp)+"test_pascal"+"/test_pascal_features1.json")) as f:
        test_pascal1=json.load(f)
    
        for i in test_pascal1: 
            if any(np.isnan(test_pascal1[i])): 
                test_pascal1[i]=list(np.nan_to_num(np.array(test_pascal1[i])))
                
        test_pascal1=subset(test_pascal1)
    
    with open (os.path.join(data_path, str(sp)+"test_pascal"+"/test_pascal_features1.json"),'w') as f:
        json.dump(test_pascal1,f)
    #
    ##""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""    
    with open (os.path.join(path, str(sp)+"test_pascal"+"/test_pascal_features2.json")) as f:
        test_pascal2=json.load(f)
    
        for i in test_pascal2: 
            if any(np.isnan(test_pascal2[i])): 
                test_pascal2[i]=list(np.nan_to_num(np.array(test_pascal2[i])))
                
        test_pascal2=subset(test_pascal2)
        
    with open (os.path.join(data_path, str(sp)+"test_pascal"+"/test_pascal_features2.json"),'w') as f:
        json.dump(test_pascal2,f)
#------------------------------------------------------------------------------------------------
crowd_path= 'D:/NNeval_classification/data/crowdflower'       
split='crowdflower'       
feature_path= os.path.join(path, "features_%s.json" %(split))

with open (feature_path,'r') as f:
    crowdflower=json.load(f) 

    for i in crowdflower: 
        if any(np.isnan(crowdflower[i])): 
            crowdflower[i]=list(np.nan_to_num(np.array(crowdflower[i])))
 
    crowdflower=subset(crowdflower)
    
with open (os.path.join(crowd_path, "features_%s.json" %(split)),'w') as f:
    json.dump(crowdflower,f)


print('All done')