# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 20:09:13 2018

@author: 22161668
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:48:38 2018

This files evaluated the scores of metrcis on two test sets, and a validation set
 and then gets the 3 types of correlation

blscore        (overall blue score)           [Blue1, Blue2, Blue3, Blue4]         4
blscores       ( n gram blue scores)          [[1gram],[2gram],[3-gram],[4-gram]]  4 x no of hyp
Rogue_L        (rogue overall score)          [score]                              1 
ROGUEL        (rogue individual scores)      [scores]                             1 x no of hyp
METEOR         (meteor scores)                [scores]                             1 x no of hyp
WMD            (wmd scores)                   [scores]                             1 x no of hyp
cider_all      (overall cider score)          [score]                              1 
CIDER          (cider individual scores)      [scores]                             1 x no of hyp
SPICE          (spice individual scores)      [scores]                             1 x no of hyp

@author: 22161668
"""

import pickle as pickle
import os
import json 
import time
import numpy as np
from nltk.corpus import stopwords
import pandas as pd
import scipy


#sys.path.append('D:/NNeval_classification/rogue_comps')

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from rg_comp import*
import gensim.models.keyedvectors as word2vec
from datetime import datetime
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def get_accuracies(scores1,Ids1, scores2, Ids2, base_dir):
    

    with open (os.path.join(base_dir,'test_pascal/test_pascal_judgement.json') ) as f:
        judgement=json.load(f)  
        
    with open (os.path.join(base_dir,'test_pascal/test_pascal_category.json') ) as f:
        category=json.load(f)  
        
    with open (os.path.join(base_dir,'test_pascal/test_pascal_id_to_labels.json') ) as f:
        labels=json.load(f) 

    for i in judgement:
        if judgement[i]==1:
            judgement[i]=0
        else:
            judgement[i]=1

    scores_1={}
    scores_2={}
    scores={}
#    label={}
    
    for i,ids in enumerate(Ids1):
        scores_1[ids]=scores1[i]
        
    for i,ids in enumerate(Ids2):
        scores_2[ids]=scores2[i]
    
    for ids in judgement:
        scores[ids]=[scores_1[ids],scores_2[ids]]
#        label[ids]=pascal_labels[ids]
    
    
    score_list={}
    
    for ids in judgement:
        score_list[ids]=np.argmax(scores[ids])
# cat HHC -1 , HHI -2 , MH- 3, MM 4        
    HHC=0   #341
    HHI=0   #284
    HM=0    #204
    MM=0    #171
    
    HHC_score=0   #341
    HHI_score=0   #284
    HM_score=0    #204
    MM_score=0    #171
    a={}
    a[1]=0
    a[2]=0
    a[3]=0
    a[4]=0
    a[5]=0

    for key in judgement:
#        value=pascal_labels[key]
        
        if score_list[key]==judgement[key]:
            yes=1
        else:
            yes=0
         
            
        if category[key]==1:
            HHC=HHC+1
            HHC_score=HHC_score+yes
        
        elif category[key]==2:
            HHI=HHI+1
            HHI_score=HHI_score+yes
        
        elif category[key]==3:
            HM=HM+1
            HM_score=HM_score+yes
                    
        else:
            MM=MM+1
            MM_score= MM_score+yes
            a[labels[key][score_list[key]]]=a[labels[key][score_list[key]]]+1
            
            
    print('Machine models', a[1], ', 2',a[2] , ', 3',a[3], ', 4',a[4], ', 5',a[5])
    #print ('{},{},{},{},{}'.format(a[1],a[2],a[3],a[4],a[5]))
    HHC_acc=float(HHC_score)/HHC   #341
    HHI_acc= float(HHI_score)/HHI   #284
    HM_acc=float(HM_score)/HM    #204
    MM_acc=float(MM_score)/MM 
    AVERAGE=(HHC_acc+HHI_acc+HM_acc+MM_acc)/4
    print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(HHC_acc,HHI_acc,HM_acc,MM_acc,AVERAGE,a[1],a[2],a[3],a[4],a[5]))

#    print ('HHC: {}\nHHI: {}\nHM: {}\nMM: {}\nAVG: {}'.format(HHC_acc,HHI_acc,HM_acc,MM_acc,AVERAGE))
    
    return [HHC_acc,HHI_acc,HM_acc,MM_acc, AVERAGE ]
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

data_path='D:/NNeval_classification/data'
glove_embedding= 'D:/NNeval_classification/wmd/GoogleNews-vectors-negative300.bin'
stop_words = stopwords.words('english')
stat_path = os.path.join(data_path, "train/train_stats.json")




option=['test_pascal']
check_flag=0 # flag to load wmd only once


for split in option:
    
    reference_path = os.path.join(data_path, "%s/%s_references.json" %(split,split))
    candidate_path1 = os.path.join(data_path, "%s/%s_captions1.json" %(split,split))
    meteor_score_path1 = os.path.join(data_path, "%s/%s_meteor_sc1.json" %(split,split))
    spice_score_path1 = os.path.join(data_path, "%s/%s_spice_sc1.json" %(split,split))

    candidate_path2 = os.path.join(data_path, "%s/%s_captions2.json" %(split,split))
    meteor_score_path2 = os.path.join(data_path, "%s/%s_meteor_sc2.json" %(split,split))
    spice_score_path2 = os.path.join(data_path, "%s/%s_spice_sc2.json" %(split,split))

    with open(reference_path, 'r') as f:
            ref = json.load(f)
            
    with open(candidate_path1, 'r') as f:
            cand1 = json.load(f)
            
    with open(candidate_path2, 'r') as f:
            cand2 = json.load(f)
            
    print ('tokenization...')
    tokenizer = PTBTokenizer()
    ref  = tokenizer.tokenize( ref)
    cand1 = tokenizer.tokenize(cand1)
    cand2 = tokenizer.tokenize(cand2)
    
    print( ' tokenzation done')
        
        #keys of all the dictionaries are stings 
    opts=[cand1,cand2]
    for cand in opts:
        
        hypo = cand
        
        if cand==cand1:
            meteor_score_path=meteor_score_path1
            spice_score_path=spice_score_path1
        else:
            meteor_score_path=meteor_score_path2
            spice_score_path=spice_score_path2
        
        print( ' start of score generation')
        
        assert(hypo.keys() == ref.keys())
        
        ImgId=hypo.keys() # for ensuring that all metrics get the same keys and return values in the same order
        
        time_now = datetime.now()
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
        # compute bleu score , n-gram precision , shortest and longest length ratios#########################
      
        
        blscore, blscores, ngprecision, lgratio=Bleu(4).compute_score(ref,hypo,ImgId)
        
        print('blue scores done\n time:  {}'.format(datetime.now()-time_now))
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
       
        # longest Common subsequence##################
        Rogue_L,ROGUEL= Rouge().compute_score(ref,hypo,ImgId)
        
        print('--Rogue_l done--\n time:  {}'.format(datetime.now()-time_now))
        #######################################################################################################    
            
        ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        # Meteor scores 
        METEOR=[]
        with open(meteor_score_path) as file:
            mt_file=json.load(file)
            
        for ids in ImgId:
            METEOR.append(mt_file[str(ids)]) # meteor scores json was saved with str as keys 
               
        print(' Meteor_done\n time:  {}'.format(datetime.now()-time_now))
        #############################################################################################
        
        '''''''''''''''''''''''''''''''''17'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
        print( '--wmd starting--')
        if(check_flag==0):
            wmd_model = word2vec.KeyedVectors.load_word2vec_format(glove_embedding, binary=True)
            #model2 = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            wmd_model.init_sims(replace=True) # 
            check_flag=1
        
        WMD=[]
        WMD_similarity=[]
        
        _count=0
        
        for id_ref in ImgId:
            c1=hypo[id_ref][0]
            c1= c1.lower().split()
            c1 = [w_ for w_ in c1 if w_ not in stop_words]
            
            distance=[]
            
            for refs in ref[id_ref]:
                c2=refs
                c2= c2.lower().split()
                c2 = [w_ for w_ in c2 if w_ not in stop_words]
                temp= wmd_model.wmdistance(c1, c2)
                
                if (np.isinf(temp)):
                    temp=10
                    print('found INF at {}, replacing with 10'.format(id_ref))
                
                distance.append(temp)
                
            wmd_dis= min(distance)
    #        wmd_similarity= 1. / (1. + (min(distance)))
            wmd_similarity=np.exp(-wmd_dis)    
            WMD.append(wmd_dis)
            WMD_similarity.append(wmd_similarity)
                
            if(_count%1000==0):
                print(_count) 
                
            _count=_count+1 
            
        print(' wmd_done\n time:  {}'.format(datetime.now()-time_now))
        
        #######################################################################################################
    #    '''''''''''''''''''''''''''''''18'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #    print('--starting Cider--')
        df_mode='corpus'
        
        CIDer=Cider(df=df_mode) # using coco-val-df as tf idf'
        
        cider_all, CIDER=CIDer.compute_score(ref,hypo,ImgId)
    
        print('--Cider done--\n time:  {}'.format(datetime.now()-time_now))
        
        #######################################################################################################    
        '''''''''''''''''''''''''''''''19'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
       # Spice scores 
        SPICE=[]
        with open(spice_score_path) as file:
            sp_file=json.load(file)
            
        for ids in ImgId:
            SPICE.append(sp_file[str(ids)]) # meteor scores json was saved with str as keys 
               
        print(' Spice_done\n time:  {}'.format(datetime.now()-time_now))
            
        #######################################################################################################
        
        mcs_lin=np.sum([SPICE,CIDER,METEOR],0)
        cs_lin=np.sum([SPICE,CIDER],0)
        
        
        BLUE1=blscores[0]
        BLUE2=blscores[1]
        BLUE3=blscores[2]
        BLUE4=blscores[3]
        
        if cand==cand1:
            ImgId1=ImgId
            BLUE1_1=BLUE1    
            BLUE2_1=BLUE2 
            BLUE3_1=BLUE3
            BLUE4_1=BLUE4      
            ROGUEL_1=ROGUEL
            METEOR_1=METEOR
            CIDER_1=CIDER
            SPICE_1=SPICE
            WMD_1=WMD
            WMD_similarity_1=WMD_similarity
            mcs_lin_1=mcs_lin
            cs_lin_1=cs_lin
        else:
            ImgId2=ImgId
            BLUE1_2=BLUE1    
            BLUE2_2=BLUE2 
            BLUE3_2=BLUE3
            BLUE4_2=BLUE4      
            ROGUEL_2=ROGUEL
            METEOR_2=METEOR
            CIDER_2=CIDER
            SPICE_2=SPICE
            WMD_2=WMD
            WMD_similarity_2=WMD_similarity
            mcs_lin_2=mcs_lin
            cs_lin_2=cs_lin

       
    metrics=[(BLUE1_1,BLUE1_2,'BLUE1'), (BLUE2_1,BLUE2_2,'BLUE2'), (BLUE3_1,BLUE3_2,'BLUE3'), 
             (BLUE4_1,BLUE4_2,'BLUE4'), (ROGUEL_1,ROGUEL_2,'ROGUE-L'), (METEOR_1,METEOR_2,'METEOR'), 
             (CIDER_1,CIDER_2,'CIDER'), (SPICE_1,SPICE_2,'SPICE'), (WMD_1,WMD_2,'WMD'),
             (mcs_lin_1, mcs_lin_2,'mcs_lin'),(cs_lin_1, cs_lin_2,'cs_lin'),
             (WMD_similarity_1, WMD_similarity_2,'WMD_similarity')]  
    
    accuracy={} 
    for scores1,scores2,name in metrics:
        print(name)
        accuracy[name]=get_accuracies(scores1, ImgId1, scores2, ImgId2, data_path)
        
        print('{}, {}, {}, {}, {}'.format(accuracy[name][0],
                                                               accuracy[name][1],accuracy[name][2],
                                                               accuracy[name][3],accuracy[name][4]))   
        
    accuracy_path=os.path.join(data_path, "accuracy/%s_accuracy.json" %(split))    
    with open(accuracy_path, 'w') as f:
        json.dump(accuracy,f)    
        
    print( '-- All {} correlations saved--\n time:  {}'.format(split, datetime.now()-time_now))