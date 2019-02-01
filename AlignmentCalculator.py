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

class AlignmentCalculator:
    p = pd.DataFrame()
    b = pd.DataFrame()
    peptides = []
    binders = []
    cluster_keys = []
    tss_df = pd.DataFrame()
    matrices = {}
    ba_linreg_predictions = pd.DataFrame()
    ba_linreg_score = 0
    
    def __init__(self, p_path = None, b_path = None):
        self.cluster_keys = ["a", "b", "c", "e", "h", "i", "p", "r"]
        self.dist_matrices = dict.fromkeys(self.cluster_keys)
        for key in self.cluster_keys:
            self.matrices[key] = pd.read_csv("./dist_matrices/Euclidean_AA_scaled_cluster_ " + key + ".csv", index_col = 0)
        self.__store_values(p_path, b_path)
    
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
                    total_score += sum(self.matrices[key].loc[self.peptides[m][l], \
                                       self.binders[n][l]] for l in range(self.length) \
                                       if self.peptides[m] is not self.binders[n])
                np_ss[m][i] = total_score
            print(str(m), str(len(self.peptides)))
        similarity_scores = pd.DataFrame(np_ss, index = self.peptides, columns = self.cluster_keys)
        self.tss_df = similarity_scores
        return similarity_scores
    
    def lin_reg_predict(self, binding_affinity_data_path):
        if self.tss_df.empty:
            self.calculate_TSS()
        X = self.tss_df
        scaler = MinMaxScaler(feature_range=(-1,1),copy=True)
        X = pd.DataFrame(scaler.fit_transform(X, [-1,1]), index = X.index, columns = X.columns)
        Y = pd.read_csv(binding_affinity_data_path, header = None)
        Y.columns = Y.iloc[0]
        Y = Y.drop(0)
        Y.set_index('sequence',inplace=True)
        reg = LinearRegression().fit(X, Y)
        prediction = reg.predict(X)
        X['prediction'] = prediction
        data = np.zeros([len(self.peptides), 2])
        data[:,0] = Y['Binding_affinity']
        data[:,1] = X['prediction']
        score = np.corrcoef(data[:,0], data[:,1])[0, 1]
        predictions = pd.DataFrame(data, index = X.index, columns = ['Binding Affinity', 'Predicted'])
        self.ba_linreg_predictions = predictions
        self.ba_linreg_score = score
        return predictions, score
    
#    def elastic_net_predict(self, binding_affinity_data_path):
#        if self.tss_df.empty:
#            self.calculate_TSS()
#        X = self.tss_df
#        scaler = MinMaxScaler(feature_range = (-1,1), copy = True)
#        X = pd.DataFrame(scaler.fit_transform(X, [-1,1]), index = X.index, columns = X.columns)
#        Y = pd.read_csv(binding_affinity_data_path, header = None)
#        Y. columns = Y.iloc[0]
#        Y = Y.drop(0)
#        Y.set_index('Sequence', inplace = True)
#        reg = ElasticNet().fit(X, Y)
#        prediction = reg.predict(X)
#        X['prediction'] = prediction