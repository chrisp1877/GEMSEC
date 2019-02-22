# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:33:56 2019

@author: Chris
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

class AlignmentCalculator(object):
    
    #   Peptide and binder sequence length must be the same
    #   Format of peptides and binders csv:
    #   0  1  2  3  4  5  6  7  8  ...  n
    #   A  I  I  D  Y  I  A  Y  M  ...  S
    #   A  I  Y  D  T  M  Q  Y  V  ...  G

    def __init__(self, p_path = None, b_path = None, TSS_path = None):
        self.cluster_keys = ["a", "b", "c", "e", "h", "i", "p", "r"]
        self.matrices = dict.fromkeys(self.cluster_keys)
        for key in self.cluster_keys:
            # UNTRAINED distance matrices
            self.matrices[key] = pd.read_csv("./improved/cluster_" + key + ".csv", index_col = 0)
            #self.matrices[key] = pd.read_csv("./dist_matrices/Euclidean_AA_scaled_cluster_ " + key + ".csv", index_col = 0)
        self.__store_values(p_path, b_path)
        if TSS_path is not None:
            self.import_TSS(TSS_path)

    def __store_values(self, p_path, b_path):
        if p_path is not None:
            self.p = pd.read_csv(p_path)
            self.length = len(list(self.p.columns))
            self.peptides = [''.join(list(self.p.iloc[m, :])) for m in range(len(list(self.p.index)))]
        if b_path is not None:
            self.b = pd.read_csv(b_path)
            self.length = len(list(self.b.columns))
            self.binders = [''.join(list(self.b.iloc[n, :])) for n in range(len(list(self.b.index)))]
        
    def set_peptides(self, p_path):
        self.__store_values(p_path, None)
    
    def set_binders(self, b_path):
        self.__store_values(None, b_path)
    
    #   Format of TSS.csv file must be:
    #              a    b    c    ...  r   
    #   AIIDYIAYM  574  641  662  ...  599
    #   AIYDTMQYV  612  690  742  ...  640
    #   ALATFTVNI  719  782  829  ...  744  
      
    def import_TSS(self, TSS_path):
        self.tss_df = pd.read_csv(TSS_path)
        self.tss_df.set_index('Unnamed: 0', inplace = True)
        self.peptides = list(self.tss_df.index)
        
    def calculate_TSS(self):
        if self.p is None or self.b is None:
            raise Exception("List of peptides or binders not set")
        if len(list(self.p.columns)) != len(list(self.b.columns)):
            raise Exception("Sequence length of peptides and binders must be equal")
        np_ss = np.zeros(shape=(len(self.peptides), len(self.cluster_keys)))
        for m in range(len(self.peptides)):
            for i, key in enumerate(self.cluster_keys):
                total_score = 0
                for n in range(len(self.binders)):
                    total_score += sum(self.matrices[key].loc[self.peptides[m][l],
                                       self.binders[n][l]] for l in range(self.length)
                                       if self.peptides[m] is not self.binders[n])
                np_ss[m][i] = total_score
            print(str(m), str(len(self.peptides)))
        similarity_scores = pd.DataFrame(np_ss, index = self.peptides, columns = self.cluster_keys)
        self.tss_df = similarity_scores
        return similarity_scores
    
    #   Format of binding affinity data must be:
    #   Sequence   Binding Affinity
    #   FLIYFRSPL  -1.04139
    #   RLDPRLAPV  -1.04139
    #   FLMQIAILV  -1.04374
    
    def lin_reg_predict(self):
        if self.tss_df.empty:
            self.calculate_TSS()
        X = self.tss_df
        scaler = MinMaxScaler(feature_range=(-1,1),copy=True)
        X = pd.DataFrame(scaler.fit_transform(X, [-1,1]), index = X.index, columns = X.columns)
        Y = pd.read_csv('largest_mhc0_1.csv', header = None)
        Y.columns = Y.iloc[0]
        Y = Y.drop(0)
        Y.set_index('Sequence',inplace=True)
        reg = LinearRegression().fit(X, Y)
        prediction = reg.predict(X)
        X['prediction'] = prediction
        data = np.zeros([len(self.peptides), 2])
        data[:,0] = Y['Binding Affinity']
        data[:,1] = X['prediction']
        score = np.corrcoef(data[:,0], data[:,1])[0, 1]
        predictions = pd.DataFrame(data, index = X.index, columns = ['Binding Affinity', 'Predicted'])
        return predictions, score
    
    def elastic_net_predict(self):
        if self.tss_df.empty:
            self.calculate_TSS()
        X = self.tss_df
        scaler = MinMaxScaler(feature_range = (-1,1), copy = True)
        X = pd.DataFrame(scaler.fit_transform(X, [-1,1]), index = X.index, columns = X.columns)
        Y = pd.read_csv('largest_mhc0_1.csv', header = None)
        Y. columns = Y.iloc[0]
        Y = Y.drop(0)
        Y.set_index('Sequence', inplace = True)
        eNet = ElasticNet(tol=0.1)
        parametersGrid = {"max_iter": [1, 5, 10],
                      "alpha": [0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}
        grid = GridSearchCV(eNet, parametersGrid, scoring='r2', cv=10)
        grid.fit(X, Y)
        prediction = grid.predict(X)
        X['prediction'] = prediction
        data = np.zeros([len(self.peptides), 2])
        data[:,0] = Y['Binding Affinity']
        data[:,1] = X['prediction']
        score = np.corrcoef(data[:,0], data[:,1])[0, 1]
        predictions = pd.DataFrame(data, index = X.index, columns = ['Binding Affinity', 'Predicted'])
        return predictions, score
'''
To run with already populated TSS matrix:

from AlignmentCalculator import AlignmentCalculator
'''
ac = AlignmentCalculator()
ac.set_peptides('peptides_full.csv')
ac.set_binders('top_binders.csv')
predicted = ac.lin_reg_predict('largest_mhc0_1.csv')[0]
score = ac.lin_reg_predict('largest_mhc0_1.csv')[1]
