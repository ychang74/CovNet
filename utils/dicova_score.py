#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:02:04 2021
@author: Team DiCOVA, IISC, Bangalore
"""

import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def score(refs,sys_outs):
    """
    inputs::
    refs: reference in the format: 
    sys_outs: a list of scores (probability of being covid positive) for each wav-fileid 
    threshold (optional): a np.array(), like np.arrange(0,1,.01), sweeping for AUC
    
    outputs::
        
    """    
    thresholds=np.arange(0,1,0.0001)
    # Read the ground truth labels into a dictionary
    reference_labels={}
    for i in range(0, len(refs)):
        key = str(i)
        reference_labels[key]=refs[i]
    

    # Read the system scores into a dictionary
    sys_scores={}
    for i in range(0, len(sys_outs)):
        key = str(i)
        #sys_scores[key]=float(sys_outs[i][1])
        sys_scores[key] = float(sys_outs[i])

    
	# Ensure all files in the reference have system scores and vice-versa
    if len(sys_scores) != len(reference_labels):
        print("Expected the score file to have scores for all files in reference and no duplicates/extra entries")
        return None
    #%%

    # Arrays to store true positives, false positives, true negatives, false negatives
    TP = np.zeros((len(reference_labels),len(thresholds)))
    TN = np.zeros((len(reference_labels),len(thresholds)))
    keyCnt=-1
    for key in sys_scores: # Repeat for each recording
        keyCnt+=1
        sys_labels = (sys_scores[key]>=thresholds)*1	# System label for a range of thresholds as binary 0/1
        gt = reference_labels[key]
        
        ind = np.where(sys_labels == gt) # system label matches the ground truth
        if gt==1:	# ground-truth label=1: True positives 
            TP[keyCnt,ind]=1
        else:		# ground-truth label=0: True negatives
            TN[keyCnt,ind]=1
            
    total_positives = sum(reference_labels.values())	# Total number of positive samples
    total_negatives = len(reference_labels)-total_positives # Total number of negative samples
    
    TP = np.sum(TP,axis=0)	# Sum across the recordings
    TN = np.sum(TN,axis=0)
    
    TPR = TP/total_positives	# True positive rate: #true_positives/#total_positives
    TNR = TN/total_negatives	# True negative rate: #true_negatives/#total_negatives
	
    AUC = auc( 1-TNR, TPR )    	# AUC 

    ind = np.where(TPR>=0.8)[0]
    sensitivity = TPR[ind[-1]]
    specificity = TNR[ind[-1]]
	    
	# pack the performance metrics in a dictionary to save & return
	# Each performance metric (except AUC) is a array for different threshold values
	# Specificity at 90% sensitivity
    scores={'TPR':TPR,
            'FPR':1-TNR,
            'AUC':AUC,
            'sensitivity':sensitivity,
            'specificity':specificity,
			'thresholds':thresholds}

    return scores

'''
if __name__=='__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--ref_file','-r',required=True)
	parser.add_argument('--target_file','-t',required=True)
	args = parser.parse_args()
	main(args.ref_file,args.target_file,args.output_file)
	
'''
