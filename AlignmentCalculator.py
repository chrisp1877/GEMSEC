# Author: Chris Pecunies

import pandas as pd
import numpy as np

class AlignmentCalculator:
    
    
    # Takes in DataFrame from csv (n rows x l columns) as binding peptides, DataFrame from csv (mrows x l columns)
    # as peptides of interest, and constructs an AlignmentCalculator object with cluster keys, corresponding distance
    # matrices (in corresponding directory, with corresponding file name), peptides DataFrame (m x l), binding DataFrame (n x l)
    # !! Input csvs must have first row be index columns, not the first peptide sequence
    def __init__(self, p = None, b = None):
        self.cluster_keys = ["a", "b", "c", "e", "h", "i", "p", "r"]
        self.amino_acids = ["A", "R", "N", "D", "C", "E", "Q", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T","W","Y","V"]
        self.matrices = dict.fromkeys(self.cluster_keys)
        for key in self.cluster_keys:
            self.matrices[key] = pd.read_csv("cluster_Euclidean\improved\cluster_" + key + ".csv", index_col = 0)
        if p is not None and b is not None:
            if len(list(p.columns)) != len(list(b.columns)):
                raise ValueError("Peptide length must be equal to compare")  
            self.binders = b
            self.peptides = p
            self.num_of_peptides = len(list(self.peptides.index))
            self.num_of_binders = len(list(self.binders.index))
            self.length = len(list(self.peptides.columns))
            self.peptide_strings = [None] * self.num_of_peptides
            self.binder_strings = [None] * self.num_of_binders
            self.storePeptideAndBinderStrings()
    
    def storePeptideAndBinderStrings(self):
        for m in range(self.num_of_peptides):
            peptide_string = ''.join(list(self.peptides.iloc[m, :]))
            self.peptide_strings[m] = peptide_string
        for n in range(self.num_of_binders):
            binder_string = ''.join(list(self.binders.iloc[n, :]))
            self.binder_strings[n] = binder_string
            
            print("Peptide " + str(m) + " of " + str(self.num_of_peptides) + " done")
            
    #takes in one-letter string 1, one-letter string 2, compares and outputs distance from distance matrix of cluster given by user.
    #This method is non-dependent on peptide and binder list AlignmentCalculator is constructed with
    def calculateAASimilarity(self, AA1, AA2, cluster):
        return self.matrices[cluster].loc[AA1, AA2]

    #pass in string 1 (peptide 1) and string 2 (peptide 2) and string for cluster, outputs sum of distances from distrance matrix
    #This method is non-dependent on peptide and binder list AlignmentCalculator is constructed with
    def calculateSequenceSimilarity(self, PEP1, PEP2, cluster):
        score = 0
        for l in range(self.length):
            score += self.calculateAASimilarity(PEP1[l], PEP2[l], cluster)
        return score
    
    # Takes in petide of interest (n rows x 12 columns), binding peptides for comparison (m rows x 12 columns) and outputs
    # a dataframe outputting similarity scores for each of n peptides of interest for each of the m binding peptides for
    # comparison, returns m x n data frame with similarity scores between each peptide and binder
    def calculateAlignment(self, cluster):
        alignment_scores = pd.DataFrame(np.zeros(shape=(self.num_of_peptides, self.num_of_binders)))
        
        for m in range(self.num_of_peptides):
            peptide_string = ''.join(list(self.peptides.iloc[m, :]))
            for n in range(self.num_of_binders):
                binder_string = ''.join(list(self.binders.iloc[n, :]))
                alignment_scores.iloc[m,n] = self.calculateSequenceSimilarity(peptide_string, binder_string, cluster)
        alignment_scores.index = self.peptide_strings
        alignment_scores.columns = self.binder_strings
        return alignment_scores
    
    # Returns a m x n dataframe which provides the sum of scores for the calculateAlignment method
    # for each of the cluster keys
    def calculateTotalAlignment(self):
        alignment_scores = self.calculateAlignment(self.cluster_keys[0])
        for i in range(1, len(self.cluster_keys)):
            alignment_scores += self.calculateAlignment(self.cluster_keys[i])   
        return alignment_scores
    
    def calculateTotalSimilarityScores(self):
        similarity_scores = pd.DataFrame(np.zeros(shape=(self.num_of_peptides, len(self.cluster_keys))))
        similarity_scores.index = self.peptide_strings
        similarity_scores.columns = self.cluster_keys
        
        for m in range(self.num_of_peptides): #iterates thru rows of dataframe
            for key in self.cluster_keys: #iterates thru columns, keys, of dataframe
                totalScore = 0
                for n in range(self.num_of_binders): #calculates similarity score for given peptide and every binder, sums and puts in column
                    totalScore += self.calculateSequenceSimilarity(self.peptides.iloc[m,:], self.binders.iloc[n, :], key)  
                similarity_scores[key].iloc[m] = totalScore
            print("Peptide " + str(m) + " of " + str(self.num_of_peptides) + " done")
        return similarity_scores