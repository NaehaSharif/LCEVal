
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 12:37:33 2018
feature extraction for test set
@author: 22161668
# this is the code for extracting scores of various metrics

0-3) n-gram precision , n= 1,2,3,4
4-7) n-gram recal,      n= 1,2,3,4
8-11) n-gram f-measure, n=1,2,3,4
12) brevity penalty
13) Rogue-L score
14) Meteor score
15) WMD score
16-19) Blue 1,2,3,4 scores
20) SPICE score
21) Cider score
22) MOWE Sentence semantic similarity
23) MOWE FAST Sentence semantic similarity
24-27) DPscores
28-43) HWCM scores
44) MOWE glove
--variable info----

blscore        (overall blue score)           [Blue1, Blue2, Blue3, Blue4]         4
blscores       ( n gram blue scores)          [[1gram],[2gram],[3-gram],[4-gram]]  4 x no of hyp
ngprecision    (n-gram modified precision)    [[1gram],[2gram],[3-gram],[4-gram]]  4 x no of hyp
bp             (length ratio)                 [ratios]                             1 x no of hyp
rg_nprecision  (rogue n-gram precision)       [[1gram],[2gram],[3-gram],[4-gram]]  4 x no of hyp
rg_nrecall     (rogue n-gram recall )         [[1gram],[2gram],[3-gram],[4-gram]]  4 x no of hyp
rg_fmeaure     (rogue n-gram fmeasure)        [[1gram],[2gram],[3-gram],[4-gram]]  4 x no of hyp
Rogue_L        (rogue overall score)          [score]                              1 
Rogue_Lscores  (rogue individual scores)      [scores]                             1 x no of hyp
meteor_scores  (meteor scores)                [scores]                             1 x no of hyp
wmd_score      (wmd scores)                   [scores]                             1 x no of hyp
cider_scores   (cider_scores)                 [scores]                             1 x no of hyp
MOWE_scores    (semantic similarity scores)   [scores]                             1 x no of hyp
MOWE_fastscores(semantic similarity scores)   [scores]                             1 x no of hyp
DPscores       (DP scores)                    [[txt],[lemma],[dep],[pos]]          4 x no of hyp
HWCM TEXTscores(hwcm scores)                  [[1chain],[2chain],[3chain],[4chain]]4 x no of hyp
HWCM LEMMAscores(hwcm scores)                 [[1chain],[2chain],[3chain],[4chain]]4 x no of hyp
HWCM DEPscores(hwcm scores)                   [[1chain],[2chain],[3chain],[4chain]]4 x no of hyp
HWCM POSscores(hwcm scores)                   [[1chain],[2chain],[3chain],[4chain]]4 x no of hyp
MOWE_glovescore(semantic similarity scores)   [scores]                             1 x no of hyp

saves the follwoing files:
Then goes the mean for all and then min for all concerned
--------------------------------------------------------------  
  
f8k_features.json
compcoco_features.json

@author: 22161668 naeha sharif
"""
import pickle as pickle
import os
import subprocess
import sys
import io
import json 
import time
import numpy as np
from nltk.corpus import stopwords
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from rg_comp import*
import gensim.models.keyedvectors as word2vec
from datetime import datetime
from scipy import spatial
import collections
import re
import gensim.models
import math
import operator
import spacy 
import syntactic 



data_path='D:/NNeval_classification/data'
#glove_embedding= 'D:/NNeval_classification/wmd/GoogleNews-vectors-negative300.bin'
stop_words = stopwords.words('english')
stat_path = os.path.join(data_path, "train/train_stats.json")
nlp = spacy.load('en_core_web_sm')
DP=syntactic.syntacticDP()


embeddings =gensim.models.KeyedVectors.load_word2vec_format( "GoogleNews-vectors-negative300.bin" , binary=True ) 
print( 'word to vec loaded') 
fasttext = gensim.models.KeyedVectors.load_word2vec_format('wiki.en.vec')
print( 'fastetxt loaded') 


#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def _parse_sentence(s): # this function parses a given string
    s = s.replace('.', '') # returns a copy of the string in which the occurrences of old have been replaced with new, s.replace(old, new)
    s = s.replace(',', '')
    s = s.replace('"', '')
    s = s.replace("'", '')
    s = s.lower() # coverts the string to lower case
    s = re.sub("\s\s+", " ", s) # \s\s+ you are looking for a space followed by 1 or more spaces in the string and trying to replace it 
    s = s.split(' ') #split or breakup a string and add the data to a string array using a defined separator.
    return s

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def dot_product2(v1, v2):
    return sum(map(operator.mul, v1, v2))

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def vector_cos(v1, v2):
    prod = dot_product2(v1, v2)
    len1 = math.sqrt(dot_product2(v1, v1))
    len2 = math.sqrt(dot_product2(v2, v2))
    return float(prod)/float(len1 * len2)

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



#option=['f8k','compcoco','test_pascal1','test_pascal2']
option=['test_pascal1','test_pascal2']
sp='28test_pascal'
#option=['test_pascal']  #run for test pascal feature 1 and feature 2

check_flag=0 # flag to load wmd only once
 
for split in option:
    
    if split=='test_pascal1':
        split='test_pascal'
        
        reference_path = os.path.join(data_path, "%s/%s_references.json" %(sp,split))
        candidate_path = os.path.join(data_path, "%s/%s_captions1.json" %(sp,split))
        spice_score_path = os.path.join(data_path, "%s/%s_spice_sc1.json" %(sp,split))  
        meteor_score_path = os.path.join(data_path, "%s/%s_meteor_sc1.json" %(sp,split))
        feature_path=os.path.join(data_path, "%s/%s_features1.json" %(sp,split))  
   
    elif split=='test_pascal2':
        split='test_pascal'
        
        reference_path = os.path.join(data_path, "%s/%s_references.json" %(sp,split))
        candidate_path = os.path.join(data_path, "%s/%s_captions2.json" %(sp,split))
        spice_score_path = os.path.join(data_path, "%s/%s_spice_sc2.json" %(sp,split))  
        meteor_score_path = os.path.join(data_path, "%s/%s_meteor_sc2.json" %(sp,split))
        feature_path=os.path.join(data_path, "%s/%s_features2.json" %(sp,split)) 
    
    else:
     
        reference_path = os.path.join(data_path, "test/test_references_%s.json" %(split))
        candidate_path = os.path.join(data_path, "test/test_captions_%s.json" %(split))
        meteor_score_path = os.path.join(data_path, "test/%s_meteor_sc.json" %(split))
        spice_score_path = os.path.join(data_path, "test/%s_spice_sc.json" %(split))
        feature_path=os.path.join(data_path, "test/%s_features.json" %(split)) 
            
    stat={}
        
        # load caption data
    with open(reference_path, 'r') as f:
            ref = json.load(f)
    with open(candidate_path, 'r') as f:
            cand = json.load(f)
            
    print ('tokenization...')
    tokenizer = PTBTokenizer()
    ref  = tokenizer.tokenize( ref)
    cand = tokenizer.tokenize(cand)
    hypo = cand
    print( ' tokenzation done')
        
    assert(hypo.keys() == ref.keys())
    ImgId=hypo.keys() # for ensuring that all metrics get the same keys and return values in the same order
        
   #==========================================================================================================
   #=============================creating vocab list from captions==============================================
    captions=[] 
    for v in hypo:
        captions.append(hypo[v][0])
        for k in ref[v]:
            captions.append(k)
            
    caps = [_parse_sentence(item) for item in captions]
    list_of_all_words = caps        
    list_of_all_words = [item for sublist in list_of_all_words for item in sublist]  
    counter = collections.Counter(list_of_all_words)
    print('---Total words in vocabulary: ', len(counter), '---') 
    
    vocab = counter.most_common(len(counter)) #Return a list of the n most common elements and their counts from the most common to the least.
    ## create word_to_idx, and idx_to_word
    vocab = [i[0] for i in vocab] # creating a list of vacabulary

    #===========================================================================   
#======================creating a subset of Glove matrix=====================================================    
    glove_file_840B300d='glove.840B.300d.txt'
    glove_matrix_840B300d={}
    emds=[(glove_file_840B300d, glove_matrix_840B300d,'glove_840B300d' )]

               
    for glove_file, glove_matrix, name in emds:
        with open(glove_file, 'r') as f:
            for line in f:
                line = line.strip()
                word = line.split(' ')[0]
                
                line = line.split(' ')[1:]
                word_vec = np.array([float(i) for i in line])
                if word in vocab:    
                    glove_matrix[word] = word_vec
                
        print('{} reading done...'.format(name))
                
    #=====================================================================================================================    
    print( ' start of feature extraction')
    time_now = datetime.now()
    #=====================================================================================================================

    '''''''''''''''''''''''''''''''''''''1-4, 13-14'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
    # compute bleu score , n-gram precision , shortest and longest length ratios#########################
    print('starting n-gram precison, shortest and longest ratios')
    
    blscore, blscores, ngprecision, lgratio=Bleu(4).compute_score(ref,hypo,ImgId)
    lgratio=ngprecision[0] 
    
    print('n-gram precison, shortest and longest ratios done\n time:  {}'.format(datetime.now()-time_now))
    #####################################################################################################
    
    '''''''''''''''''''''''''''''''''''''5-8,9-12'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    '''''''''''''''''''''''''''''''''''''1-4, 13-14'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
    # compute bleu score , n-gram precision , shortest and longest length ratios#########################
    print('starting n-gram precison, shortest and longest ratios')
    time_now = datetime.now()
    blscore, blscores, ngprecision, lgratio=Bleu(4).compute_score(ref,hypo,ImgId)
    lgratio=ngprecision[0]
    
    print('n-gram precison, shortest and longest ratios done\n time:  {}'.format(datetime.now()-time_now))
    #####################################################################################################
    
    '''''''''''''''''''''''''''''''''''''5-8,9-12'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # 1-gram recall, precision, and f-measure ############################################################
    print('starting n-gram precission, recall, f-score ')
    
    rg_nprecision= [[] for _ in range(4)]
    rg_nrecall= [[] for _ in range(4)]
    rg_fmeaure= [[] for _ in range(4)]
    
    rg_nprecision_mean= [[] for _ in range(4)]
    rg_nrecall_mean= [[] for _ in range(4)]
    rg_fmeaure_mean= [[] for _ in range(4)]
    
    rg_nprecision_min= [[] for _ in range(4)]
    rg_nrecall_min= [[] for _ in range(4)]
    rg_fmeaure_min= [[] for _ in range(4)]
    
    for id in ImgId:
        hypo_r=hypo[id]
        ref_r=ref[id]
        rouge_1 = [compute_rouge_n(hypo_r, [_r], 1) for _r in ref_r]
        
        r1_f, r1_p, r1_r = map(np.max, zip(*rouge_1))
        r1_f_mean, r1_p_mean, r1_r_mean = map(np.mean, zip(*rouge_1))
        r1_f_min, r1_p_min, r1_r_min = map(np.min, zip(*rouge_1))
        
        rg_nprecision[0].append(r1_p)
        rg_nrecall[0].append(r1_r)
        rg_fmeaure[0].append(r1_f)
        
        rg_nprecision_mean[0].append(r1_p_mean)
        rg_nrecall_mean[0].append(r1_r_mean)
        rg_fmeaure_mean[0].append(r1_f_mean)
        
        rg_nprecision_min[0].append(r1_p_min)
        rg_nrecall_min[0].append(r1_r_min)
        rg_fmeaure_min[0].append(r1_f_min)
        
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # 2-gram recall, precision, and f-measure ############################################################
        rouge_2 = [compute_rouge_n(hypo_r, [_r], 2) for _r in ref_r]
        
        r2_f, r2_p, r2_r =  map(np.max, zip(*rouge_2))
        r2_f_mean, r2_p_mean, r2_r_mean = map(np.mean, zip(*rouge_1))
        r2_f_min, r2_p_min, r2_r_min = map(np.min, zip(*rouge_1))
        
        rg_nprecision[1].append(r2_p)
        rg_nrecall[1].append(r2_r)
        rg_fmeaure[1].append(r2_f)
        
        rg_nprecision_mean[1].append(r2_p_mean)
        rg_nrecall_mean[1].append(r2_r_mean)
        rg_fmeaure_mean[1].append(r2_f_mean)
        
        rg_nprecision_min[1].append(r2_p_min)
        rg_nrecall_min[1].append(r2_r_min)
        rg_fmeaure_min[1].append(r2_f_min)

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' 
    # 3-gram recall, precision, and f-measure ############################################################
        rouge_3 = [compute_rouge_n(hypo_r, [_r], 3) for _r in ref_r]
        
        r3_f, r3_p, r3_r = map(np.max, zip(*rouge_3))
        r3_f_mean, r3_p_mean, r3_r_mean = map(np.mean, zip(*rouge_1))
        r3_f_min, r3_p_min, r3_r_min = map(np.min, zip(*rouge_1))
        
        rg_nprecision[2].append(r3_p)
        rg_nrecall[2].append(r3_r)
        rg_fmeaure[2].append(r3_f)
        
        
        rg_nprecision_mean[2].append(r3_p_mean)
        rg_nrecall_mean[2].append(r3_r_mean)
        rg_fmeaure_mean[2].append(r3_f_mean)
        
        rg_nprecision_min[2].append(r3_p_min)
        rg_nrecall_min[2].append(r3_r_min)
        rg_fmeaure_min[2].append(r3_f_min)

#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # 4-gram recall, precision, and f-measure ############################################################
        rouge_4 = [compute_rouge_n(hypo_r, [_r], 4) for _r in ref_r]
        
        r4_f, r4_p, r4_r = map(np.max, zip(*rouge_4))
        r4_f_mean, r4_p_mean, r4_r_mean = map(np.mean, zip(*rouge_1))
        r4_f_min, r4_p_min, r4_r_min = map(np.min, zip(*rouge_1))
        
        rg_nprecision[3].append(r4_p)
        rg_nrecall[3].append(r4_r)
        rg_fmeaure[3].append(r4_f)
        
        rg_nprecision_mean[3].append(r4_p_mean)
        rg_nrecall_mean[3].append(r4_r_mean)
        rg_fmeaure_mean[3].append(r4_f_mean)
        
        rg_nprecision_min[3].append(r4_p_min)
        rg_nrecall_min[3].append(r4_r_min)
        rg_fmeaure_min[3].append(r4_f_min)
    
    print('n-gram precission, recall, f-score done\ time:  {}'.format(datetime.now()-time_now))
    ######################################################################################################
      
    '''''''''''''''''''''''''''''''15'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print('--starting Rogue_L--')
    time_now = datetime.now()
    # longest Common subsequence##################
    Rogue_L,Rogue_Lscores= Rouge().compute_score(ref,hypo,ImgId)
    
    print('--Rogue_l done--\n time:  {}'.format(datetime.now()-time_now))
    #######################################################################################################    
        
    '''''''''''''''''''''''''''''''''16'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Meteor scores 
    meteor_scores=[]
    with open(meteor_score_path) as file:
        mt_file=json.load(file)
        
    for ids in ImgId:
        meteor_scores.append(mt_file[str(ids)]) # meteor scores json was saved with str as keys 
           
    print(' Meteor_done\n time:  {}'.format(datetime.now()-time_now))
    #############################################################################################
    
    '''''''''''''''''''''''''''''''''17'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print( '--wmd starting--')
    if(check_flag==0):
        wmd_model = embeddings
        wmd_model.init_sims(replace=True) # 
        check_flag=1
    
    wmd_score=[]
    wmd_score_mean=[]
    wmd_score_min=[]
    
    _count=0
    time_now = datetime.now()
    
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
                temp=1000
                #print('found INF at {}, replacing with 1000'.format(id_ref))
            
            distance.append(temp)
            
        wmd_dis=min(distance)
        wmd_similarity=np.exp(-wmd_dis)
        wmd_score.append(wmd_similarity)
        time_now1 = datetime.now()
        wmd_dis=np.mean(distance)
        wmd_similarity=np.exp(-wmd_dis)
        wmd_score_mean.append(wmd_similarity)
        
        wmd_dis=np.min(distance)
        wmd_similarity=np.exp(-wmd_dis)
        wmd_score_min.append(wmd_similarity)
        time_now2 = datetime.now()
            
#        if(_count%1000==0):
#            print(_count) 
#            
#        _count=_count+1 
        
    print(' wmd_done\n time:  {}'.format(datetime.now()-time_now))
    print(' wmd_middle\n time:  {}'.format(time_now2-time_now1))
    #######################################################################################################
#    '''''''''''''''''''''''''''''''18'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print('--starting Cider--')
    df_mode='coco-val-df'
    time_now = datetime.now()
    CIDer=Cider(df=df_mode) # using coco-val-df as tf idf'
    
    cider_all, cider_scores=CIDer.compute_score(ref,hypo,ImgId)

    print('--Cider done--\n time:  {}'.format(datetime.now()-time_now))
    
    #######################################################################################################    
    '''''''''''''''''''''''''''''''19'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
   # Spice scores 
    spice_scores=[]
    with open(spice_score_path) as file:
        sp_file=json.load(file)
        
    for ids in ImgId:
        spice_scores.append(sp_file[str(ids)]) # meteor scores json was saved with str as keys 
           
    print(' Spice_done\n time:  {}'.format(datetime.now()-time_now))
        
    #######################################################################################################
    
    #"""""""""""""""""""""""""""""""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #----------------------Sentence Embedding features -------------------------------------------------
    
    print('starting sentence embeddings')
    
    #  '''''''''''''''''''''''''''''''''17'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    print( '--Mowe glove starting--\n')
    _count=0 
    MOWE_glovescores=[] # mean of word embeddings
    MOWE_glovescores_mean=[]
    MOWE_glovescores_min=[]
    
    time_now = datetime.now()
    for id_ref in ImgId:
            
        newhyp=hypo[id_ref][0]
        newref=ref[id_ref]
            
        hyp_embed=[] 
        ref_embed={}
        temp=[]
            
        for words in _parse_sentence(newhyp):
            if words not in stopwords.words("english"):
                try:
                    w=list(glove_matrix[words])
                    hyp_embed.append(w) 
                except:
                    pass                           
        hyp_embed=np.mean(hyp_embed,0)
            
            
        distance=[]
            
        for s,refs in enumerate(newref):
            for rwords in _parse_sentence(refs):
                if rwords not in stopwords.words("english"):
                        try:
                            w=list(glove_matrix[rwords])
                            temp.append(w)
                        except:
                            pass
                            
            ref_embed[s]=np.mean(temp,0)
            temp=[]
            
            
        result=[]
            
        for r in ref_embed:
            
            if np.isnan(hyp_embed).any() or np.isnan(ref_embed[r]).any(): 
                result.append(-1)
                
            else:
                sentence1=hyp_embed
                sentence2=ref_embed[r]
                result.append(vector_cos(sentence1,sentence2))  
                
                
        MOWE_glovescores.append(max(result))  # mean of word embeddings  
        MOWE_glovescores_mean.append(np.mean(result))
        MOWE_glovescores_min.append(np.min(result))  
        
        if(_count%1000==0):
            print(_count) 
                
        _count=_count+1   
        
    print( 'glove MOWE done!')
    print(' glove MOWE \n time:  {}'.format(datetime.now()-time_now))
            
    #-------------------------------------word 2 vec----------------------------------------------------------------------------        

    #------------------------------ word2Vec starting-------------------------------------------------------------------------
    _count=0 
    MOWE_scores=[] # mean of word embeddings
    MOWE_scores_mean=[]
    MOWE_scores_min=[]    
    
    for id_ref in ImgId:
            
        newhyp=hypo[id_ref][0]
        newref=ref[id_ref]
            
        hyp_embed=[] 
        ref_embed={}
        temp=[]
            
        for words in _parse_sentence(newhyp):
            if words not in stopwords.words("english"):
                try:
                    w=list(embeddings[words])
                    hyp_embed.append(w) 
                except:
                    pass
                    

              
        hyp_embed=np.mean(hyp_embed,0)
            
            
        distance=[]
            
        for s,refs in enumerate(newref):
            for rwords in _parse_sentence(refs):
                if rwords not in stopwords.words("english"):
                        try:
                            w=list(embeddings[rwords])
                            temp.append(w)
                        except:
                            pass
                            
            ref_embed[s]=np.mean(temp,0)
            temp=[]
            
            
        result=[]
            
        for r in ref_embed:
            
            if np.isnan(hyp_embed).any() or np.isnan(ref_embed[r]).any(): 
                result.append(-1)
                
            else:
                sentence1=hyp_embed
                sentence2=ref_embed[r]
                result.append(vector_cos(sentence1,sentence2))  
                
                
        MOWE_scores.append(max(result))   
        MOWE_scores_mean.append(np.mean(result))
        MOWE_scores_min.append(np.min(result))
                
        if(_count%1000==0):
            print(_count) 
                
        _count=_count+1   
        
    print( 'W2V MOWE done!')
    print(' W2V MOWE \n time:  {}'.format(datetime.now()-time_now))
#----------------------------------------------------------------------------------------------------
    #######################################################################################################
    
    #"""""""""""""""""""""""""""""""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    #----------------------Sentence Embedding features -------------------------------------------------
      
    print('fast mowe starting')
    _count=0 
    MOWE_fastscores=[] # mean of word embeddings fasttext
    MOWE_fastscores_mean=[]    
    MOWE_fastscores_min=[]
    time_now = datetime.now()
    for id_ref in ImgId:
            
        newhyp=hypo[id_ref][0]
        newref=ref[id_ref]
            
        hyp_embed=[] 
        ref_embed={}
        temp=[]
            
        for words in _parse_sentence(newhyp):
            if words not in stopwords.words("english"):
                try:
                    w=list(fasttext[words])
                    hyp_embed.append(w) 
                except:
                    pass
                    

        hyp_embed=np.mean(hyp_embed,0)
            
            
        distance=[]
            
        for s,refs in enumerate(newref):
            for rwords in _parse_sentence(refs):
                if rwords not in stopwords.words("english"):
                        try:
                            w=list(fasttext[rwords])
                            temp.append(w)
                        except:
                            pass
                            
            ref_embed[s]=np.mean(temp,0)
            temp=[]
            
            
        result=[]
            
        for r in ref_embed:
            
            if np.isnan(hyp_embed).any() or np.isnan(ref_embed[r]).any(): 
                result.append(-1)
                
            else:
                sentence1=hyp_embed
                sentence2=ref_embed[r]
                result.append(vector_cos(sentence1,sentence2))  
                
                
        MOWE_fastscores.append(max(result))  # mean of word embeddings  
        MOWE_fastscores_mean.append(np.mean(result))
        MOWE_fastscores_min.append(np.min(result))
        
        
#        if(_count%1000==0):
#            print(_count) 
                
        _count=_count+1     
    print(' W2V MOWE fast done \n time:  {}'.format(datetime.now()-time_now))
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''   
          
        
    print( '--DP all starting--')
    _count=0
    DP_all_score=[[] for i in range(4)]
    DP_all_score_mean=[[] for i in range(4)]
    DP_all_score_min=[[] for i in range(4)]
    
    for id_ref in ImgId:
        c1=nlp(unicode(hypo[id_ref][0]))
        
        DP_all=[]

                
        for refs in ref[id_ref]:
            c2=nlp(unicode(refs))
            temp= DP.score_DP_all(c1, c2)
                
            DP_all.append(temp)
            
        DP_score= np.max(DP_all,0)
        DP_score_mean= np.mean(DP_all,0)
        DP_score_min= np.min(DP_all,0)
        
        for i,s in enumerate(DP_score):
            DP_all_score[i].append(s)
        
        for i,s in enumerate(DP_score_mean):
            DP_all_score_mean[i].append(s)
            
        for i,s in enumerate(DP_score_min):
            DP_all_score_min[i].append(s)
        
#        if(_count%1000==0):
#            print(_count) 
#            
        _count=_count+1 
        
    print(' DP_all done\n time:  {}'.format(datetime.now()-time_now))
    
# ===================================================================================================================       
    print( '--HWCM all starting--')
    _count=0
    HWCM_text_score=[[] for i in range(4)]
    HWCM_lemma_score=[[] for i in range(4)]
    HWCM_dep_score=[[] for i in range(4)]
    HWCM_pos_score=[[] for i in range(4)]
    
    #mean
    HWCM_text_score_mean=[[] for i in range(4)]
    HWCM_lemma_score_mean=[[] for i in range(4)]
    HWCM_dep_score_mean=[[] for i in range(4)]
    HWCM_pos_score_mean=[[] for i in range(4)]
    
    #min
    HWCM_text_score_min=[[] for i in range(4)]
    HWCM_lemma_score_min=[[] for i in range(4)]
    HWCM_dep_score_min=[[] for i in range(4)]
    HWCM_pos_score_min=[[] for i in range(4)]
    
    time_now = datetime.now()
    for id_ref in ImgId:
        c1=nlp(unicode(hypo[id_ref][0]))
        
        hwcm_text=[]
        hwcm_lemma=[]
        hwcm_dep=[]
        hwcm_pos=[]
        

                
        for refs in ref[id_ref]:
            c2=nlp(unicode(refs))
            temp= DP.score_HWCM_all(c1, c2)
                
            hwcm_text.append(temp[0])
            hwcm_lemma.append(temp[1])
            hwcm_dep.append(temp[2])
            hwcm_pos.append(temp[3])
            
            
        hwcm_text_score= np.max(hwcm_text,0)
        hwcm_lemma_score= np.max(hwcm_lemma,0)
        hwcm_dep_score= np.max(hwcm_dep,0)
        hwcm_pos_score= np.max(hwcm_pos,0)
        
        #mean
        hwcm_text_score_mean= np.mean(hwcm_text,0)
        hwcm_lemma_score_mean= np.mean(hwcm_lemma,0)
        hwcm_dep_score_mean= np.mean(hwcm_dep,0)
        hwcm_pos_score_mean= np.mean(hwcm_pos,0)
        #min
        hwcm_text_score_min= np.min(hwcm_text,0)
        hwcm_lemma_score_min= np.min(hwcm_lemma,0)
        hwcm_dep_score_min= np.min(hwcm_dep,0)
        hwcm_pos_score_min= np.min(hwcm_pos,0)
        
      
        for i,s in enumerate(hwcm_text_score):
            HWCM_text_score[i].append(s)
            
        for i,s in enumerate(hwcm_lemma_score):
            HWCM_lemma_score[i].append(s)
            
        for i,s in enumerate(hwcm_dep_score):
            HWCM_dep_score[i].append(s)     
            
        for i,s in enumerate(hwcm_pos_score):
            HWCM_pos_score[i].append(s)
            
        #mean
        for i,s in enumerate(hwcm_text_score_mean):
            HWCM_text_score_mean[i].append(s)
            
        for i,s in enumerate(hwcm_lemma_score_mean):
            HWCM_lemma_score_mean[i].append(s)
            
        for i,s in enumerate(hwcm_dep_score_mean):
            HWCM_dep_score_mean[i].append(s)     
            
        for i,s in enumerate(hwcm_pos_score_mean):
            HWCM_pos_score_mean[i].append(s)
         
        #min 
        for i,s in enumerate(hwcm_text_score_min):
            HWCM_text_score_min[i].append(s)
            
        for i,s in enumerate(hwcm_lemma_score_min):
            HWCM_lemma_score_min[i].append(s)
            
        for i,s in enumerate(hwcm_dep_score_min):
            HWCM_dep_score_min[i].append(s)     
            
        for i,s in enumerate(hwcm_pos_score_min):
            HWCM_pos_score_min[i].append(s)
            
         
                   
#        if(_count%1000==0):
#            print(_count) 
#            
#        _count=_count+1 
        
    print(' hwcm_all done\n time:  {}'.format(datetime.now()-time_now))
#        
    
    
#    
    print( '-- All {} features extracted--\n time:  {}'.format(split, datetime.now()-time_now))    
##########################################################################   
   
        
#    with open (stat_path) as f:
#            stat= json.load(f)
#            mean=stat['mean']
#            std= stat['std']


    features={}
    for i,ids in enumerate(ImgId):
        
            
          features[ids]=[ngprecision[0][i],
                         ngprecision[1][i],
                         ngprecision[2][i],
                         ngprecision[3][i],
                         rg_nrecall[0][i],
                         rg_nrecall[1][i],
                         rg_nrecall[2][i],
                         rg_nrecall[3][i],
                         rg_fmeaure[0][i],
                         rg_fmeaure[1][i],
                         rg_fmeaure[2][i],
                         rg_fmeaure[3][i],
                         lgratio[i],
                         Rogue_Lscores[i],
                         meteor_scores[i],
                         wmd_score[i],
                         blscores[0][i],
                         blscores[1][i],
                         blscores[2][i],
                         blscores[3][i],
                         spice_scores[i],
                         cider_scores[i],
                         MOWE_scores[i],
                         MOWE_fastscores[i],
                         DP_all_score[0][i],
                         DP_all_score[1][i],
                         DP_all_score[2][i],
                         DP_all_score[3][i],
                         HWCM_text_score[0][i],
                         HWCM_text_score[1][i],
                         HWCM_text_score[2][i],
                         HWCM_text_score[3][i],
                         HWCM_lemma_score[0][i],
                         HWCM_lemma_score[1][i],
                         HWCM_lemma_score[2][i],
                         HWCM_lemma_score[3][i],
                         HWCM_dep_score[0][i],
                         HWCM_dep_score[1][i],
                         HWCM_dep_score[2][i],
                         HWCM_dep_score[3][i],
                         HWCM_pos_score[0][i],
                         HWCM_pos_score[1][i],
                         HWCM_pos_score[2][i],
                         HWCM_pos_score[3][i],
                         MOWE_glovescores[i],
  #-------------------------mean----------------------------                       
                         rg_nprecision_mean[0][i],
                         rg_nprecision_mean[1][i],
                         rg_nprecision_mean[2][i],
                         rg_nprecision_mean[3][i],
                         rg_nrecall_mean[0][i],
                         rg_nrecall_mean[1][i],
                         rg_nrecall_mean[2][i],
                         rg_nrecall_mean[3][i],
                         rg_fmeaure_mean[0][i],
                         rg_fmeaure_mean[1][i],
                         rg_fmeaure_mean[2][i],
                         rg_fmeaure_mean[3][i],
                         lgratio[i],
                         Rogue_Lscores[i],
                         meteor_scores[i],
                         wmd_score_mean[i],
                         blscores[0][i],
                         blscores[1][i],
                         blscores[2][i],
                         blscores[3][i],
                         spice_scores[i],
                         cider_scores[i],
                         MOWE_scores_mean[i],
                         MOWE_fastscores_mean[i],
                         DP_all_score_mean[0][i],
                         DP_all_score_mean[1][i],
                         DP_all_score_mean[2][i],
                         DP_all_score_mean[3][i],
                         HWCM_text_score_mean[0][i],
                         HWCM_text_score_mean[1][i],
                         HWCM_text_score_mean[2][i],
                         HWCM_text_score_mean[3][i],
                         HWCM_lemma_score_mean[0][i],
                         HWCM_lemma_score_mean[1][i],
                         HWCM_lemma_score_mean[2][i],
                         HWCM_lemma_score_mean[3][i],
                         HWCM_dep_score_mean[0][i],
                         HWCM_dep_score_mean[1][i],
                         HWCM_dep_score_mean[2][i],
                         HWCM_dep_score_mean[3][i],
                         HWCM_pos_score_mean[0][i],
                         HWCM_pos_score_mean[1][i],
                         HWCM_pos_score_mean[2][i],
                         HWCM_pos_score_mean[3][i],
                         MOWE_glovescores_mean[i],
        
#------------------------------min-------------------------------------------------
                         rg_nprecision_min[0][i],
                         rg_nprecision_min[1][i],
                         rg_nprecision_min[2][i],
                         rg_nprecision_min[3][i],
                         rg_nrecall_min[0][i],
                         rg_nrecall_min[1][i],
                         rg_nrecall_min[2][i],
                         rg_nrecall_min[3][i],
                         rg_fmeaure_min[0][i],
                         rg_fmeaure_min[1][i],
                         rg_fmeaure_min[2][i],
                         rg_fmeaure_min[3][i],
                         lgratio[i],
                         Rogue_Lscores[i],
                         meteor_scores[i],
                         wmd_score_min[i],
                         blscores[0][i],
                         blscores[1][i],
                         blscores[2][i],
                         blscores[3][i],
                         spice_scores[i],
                         cider_scores[i],
                         MOWE_scores_min[i],
                         MOWE_fastscores_min[i],
                         DP_all_score_min[0][i],
                         DP_all_score_min[1][i],
                         DP_all_score_min[2][i],
                         DP_all_score_min[3][i],
                         HWCM_text_score_min[0][i],
                         HWCM_text_score_min[1][i],
                         HWCM_text_score_min[2][i],
                         HWCM_text_score_min[3][i],
                         HWCM_lemma_score_min[0][i],
                         HWCM_lemma_score_min[1][i],
                         HWCM_lemma_score_min[2][i],
                         HWCM_lemma_score_min[3][i],
                         HWCM_dep_score_min[0][i],
                         HWCM_dep_score_min[1][i],
                         HWCM_dep_score_min[2][i],
                         HWCM_dep_score_min[3][i],
                         HWCM_pos_score_min[0][i],
                         HWCM_pos_score_min[1][i],
                         HWCM_pos_score_min[2][i],
                         HWCM_pos_score_min[3][i],
                         MOWE_glovescores_min[i]]
#        
            
            
     
        
       
    with open(feature_path, 'w') as f:
        json.dump(features,f)    
        
    print( '-- All {} features saved--\n time:  {}'.format(split, datetime.now()-time_now))
    
    #----------------------------------------------------------------------------------------------
     #----------------------------------------------------------------------------------------------
     #----------------------------------------------------------------------------------------------
    
 