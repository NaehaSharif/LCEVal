# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:43:55 2018
syntactic feature extraction 

*FUNCTIONS* 
score_DP_all(doc, ref): # returns overlap between terminal nodes according to text, lemma, dep, pos
HWCM(doc) : # return 1-3 headword chains   
score_HWCM_all (doc, ref): # returns hwcm matches between cand and ref according to text, lemma, dep, pos
    
    
@author: 22161668
"""

import spacy
import numpy as np 
import collections 
from collections import namedtuple




#for token in doc:
#    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#          token.shape_, token.is_alpha, token.is_stop)
#    
    
class syntacticDP(object):
    
    def __init__(self):
    
        self.Tree = namedtuple("Tree", ["value", "children"])
        self.nodes=[]
        self.nodes2=[]
        self.nodes3=[]
        self.nodes4=[]

    def is_root(self, token):
        return token.dep_=="ROOT"
    
    
    def get_subtrees(self, head_token, extract=lambda x:x):
        return self.Tree(value = extract(head_token),
             children = [self.get_subtrees(child_token, extract) for child_token in head_token.children]
            )
    
    def get_trees(self, sentences, extract=lambda x:x):
        roots = filter(self.is_root, sentences) # one root per sentence
        return [self.get_subtrees(root, extract) for root in roots]   
    
    
    
    def print_tree(self, tree, indent_level=0):
        print("  "*indent_level + "├─" + str(tree.value))
        for child in tree.children:
            self.print_tree(child, indent_level+1)
            
    def tree_nodes(self, tree):
        
        self.nodes.append(str(tree.value))
        for child in tree.children:
           self.tree_nodes(child)
        
    def hwcm2(self, tree):
        for child in tree.children:
           self.nodes2.append((str(tree.value), str(child.value))) 
           self.hwcm2(child)           

    def hwcm3(self, tree):
        for child in tree.children:
            for hc in [self.nodes2[l][1] for l,a in enumerate([i[0] for i in self.nodes2]) if a==str(child.value)]:
                self.nodes3.append((str(tree.value), str(child.value),hc)) 
            self.hwcm3(child)   
            
    def hwcm4(self, tree):
        for child in tree.children:
            for hc in [self.nodes3[l][1:3] for l,a in enumerate([i[0] for i in self.nodes3]) if a==str(child.value)]:
                self.nodes4.append((str(tree.value), str(child.value), hc[0], hc[1] )) 
            self.hwcm4(child)   
                                 

    #print_tree(get_trees(doc)[0])
    #print_tree(get_trees(doc, lambda tok: tok.pos_)[0])
    #print_tree(get_trees(doc, lambda tok: (tok, tok.dep_, tok.pos_))[0])
    
    def get_leaves(self, node):
        
        if not node.children:
            yield node
    
        for child in node.children:
            for leaf in self.get_leaves(child):
                 yield leaf
                 
    
    def leaf_list(self,leaves):
        leaflist=[]
        for child in leaves:
            leaflist.append(str(child.value))
        return leaflist
    
    def leaf_list_all(self,leaves):
        leaflist=[]
        for child in leaves:
            leaflist.append((child.value))
        return leaflist
    

    
    def r_list_nc(self,r_list): # returns names and counts of r_list
        
        counter=collections.Counter(r_list)
        count=counter.most_common(len(counter))
        r_list_names=[i[0] for i in count]
        r_list_count=[i[1] for i in count]
        
        return  r_list_names, r_list_count
    
    
    def SCORE(self,h_list, r_list_names, r_list_count): 
        
        small = 1e-9
        h={}
        r={}
        for i, l in enumerate(r_list_names):
            r[l] = r_list_count[i]
            
        for l in h_list:
            h[l]=0
            
        for l in h_list:
            if l in r_list_names:
                h[l]=h[l]+1
                if h[l]>r[l]:
                    h[l]=r[l]
                    
        counth=0
        countr=np.sum(r_list_count)
        
        for x in h:
            counth=h[x]+counth
        
        score=float(counth)/float(countr+small)
        
        return score
#''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''    
    def score_DP_w(self,doc,ref): #overlap between terminal nodes according to words
        small = 1e-9
        
        hwords=self.get_trees(doc)[0]
        rwords=self.get_trees(ref)[0]
        h={}
        r={}
        h_leaves=self.get_leaves(hwords)
        h_list=self.leaf_list(h_leaves)
        
        r_leaves=self.get_leaves(rwords)
        r_list=self.leaf_list(r_leaves)
        
        
        counter=collections.Counter(r_list)
        r_list_count=counter.most_common(len(counter))
        r_list_names=[i[0] for i in r_list_count]
        r_list_count=[i[1] for i in r_list_count]
        
        
        
        for i, l in enumerate(r_list_names):
            r[l] = r_list_count[i]
            
        for l in h_list:
            h[l]=0
            
        for l in h_list:
            if l in r_list_names:
                h[l]=h[l]+1
                if h[l]>r[l]:
                    h[l]=r[l] # clippimg
                    
        counth=0
        countr=np.sum(r_list_count)
        for x in h:
            counth=h[x]+counth
        
        score=float(counth)/(countr+small)
        
        return score

    def score_DP_pos(self,doc,ref): # overlap between terminal nodes according to pos
        small = 1e-9
        tiny = 1e-15
        
        hwords=self.get_trees(doc, lambda tok: tok.pos_)[0]
        rwords=self.get_trees(ref, lambda tok: tok.pos_)[0]
        h={}
        r={}
        h_leaves=self.get_leaves(hwords)
        h_list=self.leaf_list(h_leaves)
        
        r_leaves=self.get_leaves(rwords)
        r_list=self.leaf_list(r_leaves)
        
        
        
        
        counter=collections.Counter(r_list)
        r_list_count=counter.most_common(len(counter))
        r_list_names=[i[0] for i in r_list_count]
        r_list_count=[i[1] for i in r_list_count]
        
        
        
        for i, l in enumerate(r_list_names):
            r[l] = r_list_count[i]
            
        for l in h_list:
            h[l]=0
            
        for l in h_list:
            if l in r_list_names:
                h[l]=h[l]+1
                if h[l]>r[l]:
                    h[l]=r[l] # clipping
                    
        counth=0
        countr=np.sum(r_list_count)
        for x in h:
            counth=h[x]+counth
        
        score=float(counth)/(countr+small)
        
        return score
    
    def score_DP_dep(self,doc,ref): #overlap between terminal nodes according to dep
        small = 1e-9
        tiny = 1e-15
        
        hwords=self.get_trees(doc, lambda tok: tok.dep_)[0]
        rwords=self.get_trees(ref, lambda tok: tok.dep_)[0]
        h={}
        r={}
        h_leaves=self.get_leaves(hwords)
        h_list=self.leaf_list(h_leaves)
        
        r_leaves=self.get_leaves(rwords)
        r_list=self.leaf_list(r_leaves)
        
        
        
        
        counter=collections.Counter(r_list)
        r_list_count=counter.most_common(len(counter))
        r_list_names=[i[0] for i in r_list_count]
        r_list_count=[i[1] for i in r_list_count]
        
        
        
        for i, l in enumerate(r_list_names):
            r[l] = r_list_count[i]
            
        for l in h_list:
            h[l]=0
            
        for l in h_list:
            if l in r_list_names:
                h[l]=h[l]+1
                if h[l]>r[l]:
                    h[l]=r[l]
                    
        counth=0
        countr=np.sum(r_list_count)
        for x in h:
            counth=h[x]+counth
        
        score=float(counth)/(countr+small)
      
        return score
    
    
    def score_DP_all(self,doc,ref): #overlap between terminal nodes according to text, lemma, dep, pos

        
        hwords=self.get_trees(doc, lambda tok: (tok.text, tok.lemma_, tok.dep_, tok.pos_))[0]
        rwords=self.get_trees(ref, lambda tok: (tok.text, tok.lemma_, tok.dep_, tok.pos_))[0]

        
        h_leaves=self.get_leaves(hwords)
        h_list=self.leaf_list_all(h_leaves) # list of tuples
        
        
        r_leaves=self.get_leaves(rwords)
        r_list=self.leaf_list_all(r_leaves)
        
            
        r_list_w=[i[0] for i in r_list]    
        r_list_lemma=[i[1] for i in r_list]
        r_list_dep=[i[2] for i in r_list]
        r_list_pos=[i[3] for i in r_list]
         
        h_list_w=[i[0] for i in h_list]
        h_list_lemma=[i[1] for i in h_list]
        h_list_dep=[i[2] for i in h_list]
        h_list_pos=[i[3] for i in h_list]
        
        
        r_list_names_w, r_list_count_w = self.r_list_nc(r_list_w)
        r_list_names_lemma, r_list_count_lemma = self.r_list_nc(r_list_lemma)
        r_list_names_dep, r_list_count_dep = self.r_list_nc(r_list_dep)
        r_list_names_pos, r_list_count_pos = self.r_list_nc(r_list_pos)
        
        scores_all=[self.SCORE(h_list_w, r_list_names_w, r_list_count_w), self.SCORE(h_list_lemma, r_list_names_lemma, r_list_count_lemma), 
                    self.SCORE(h_list_dep, r_list_names_dep, r_list_count_dep), self.SCORE(h_list_pos, r_list_names_pos, r_list_count_pos) ]


      
        return scores_all
    
    def HWCM(self,hwords): # return 1-3 head word chains
        
        self.nodes=[]
        self.nodes2=[]
        self.nodes3=[]
        self.nodes4=[]
        
            
        self.tree_nodes(hwords)
        h1chain=self.nodes            
            
        self.hwcm2(hwords)
        h2chain=self.nodes2

        self.hwcm3(hwords)
        h3chain=self.nodes3        
        
        
        self.hwcm4(hwords)
        h4chain=self.nodes4    
       
        return h1chain,h2chain,h3chain,h4chain
    
    def score_HWCM_all(self,doc,ref): #overlap between terminal nodes according to text, lemma, dep, pos

        
        hwords_text=self.get_trees(doc, lambda tok: tok.text)[0]
        hwords_lemma=self.get_trees(doc, lambda tok: tok.lemma_)[0]
        hwords_dep=self.get_trees(doc, lambda tok: tok.dep_)[0]
        hwords_pos=self.get_trees(doc, lambda tok: tok.pos_)[0]
        
        rwords_text=self.get_trees(ref, lambda tok: tok.text)[0]  
        rwords_lemma=self.get_trees(ref, lambda tok: tok.lemma_)[0] 
        rwords_dep=self.get_trees(ref, lambda tok: tok.dep_)[0] 
        rwords_pos=self.get_trees(ref, lambda tok: tok.pos_)[0] 
        
        
        
        score_text=self.HWCM_process(hwords_text,rwords_text)
        score_lemma=self.HWCM_process(hwords_lemma,rwords_lemma)
        score_dep=self.HWCM_process(hwords_dep,rwords_dep)
        score_pos=self.HWCM_process(hwords_pos,rwords_pos)
        
        return [score_text, score_lemma, score_dep, score_pos]
    
    
    def HWCM_process(self,hwords,rwords):   
        
         
         hwcm1, hwcm2, hwcm3, hwcm4 = self.HWCM(hwords)
         rwcm1, rwcm2, rwcm3, rwcm4 = self.HWCM(rwords)
         
         rwcm1_names, rwcm1_count = self.r_list_nc(rwcm1)
         overlap1=self.SCORE(hwcm1, rwcm1_names, rwcm1_count)
         
         rwcm2_names, rwcm2_count = self.r_list_nc(rwcm2)
         overlap2=self.SCORE(hwcm2, rwcm2_names, rwcm2_count)
         
         rwcm3_names, rwcm3_count = self.r_list_nc(rwcm3)
         overlap3=self.SCORE(hwcm3, rwcm3_names, rwcm3_count)
            
         
         rwcm4_names, rwcm4_count = self.r_list_nc(rwcm4)
         overlap4=self.SCORE(hwcm4, rwcm4_names, rwcm4_count)
         
         
         HWCM1=overlap1
         
         HWCM2= 0.5*(HWCM1+overlap2)
         
         HWCM3= 0.33*(HWCM1+overlap2+overlap3)
         
         HWCM4=0.25*(HWCM1+overlap2+overlap3+overlap4)
         
         return [HWCM1,HWCM2,HWCM3,HWCM4]
         
         
         
         
        