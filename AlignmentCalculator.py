# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:33:56 2019

@author: Chris
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

class AlignmentCalculator:
    p = pd.DataFrame()
    b = pd.DataFrame()
    p_strings = []
    b_strings = []
    cluster_keys = []
    tss_df = pd.DataFrame()
    matrices = {}
    num_of_peptides = 0
    num_of_binders = 0
    ba_linreg_predictions = pd.DataFrame()
    ba_linreg_score = 0
    
    def __init__(self, p = None, b = None):
        self.cluster_keys = ["a", "b", "c", "e", "h", "i", "p", "r"]
        self.dist_matrices = dict.fromkeys(self.cluster_keys)
        for key in self.cluster_keys:
            self.matrices[key] = pd.read_csv("./dist_matrices/Euclidean_AA_scaled_cluster_ " + key + ".csv", index_col = 0)
        if p is not None and b is not None:
            self.__store_values(p, b)
    
    def __store_values(self, p, b):
        if p is not None:
            self.p = pd.DataFrame(p)
            self.num_of_peptides = len(list(p.index))
            self.length = len(list(p.columns))
            self.p_strings = [''.join(list(p.iloc[m, :])) for m in range(self.num_of_peptides)]
        if b is not None:
            self.b = pd.DataFrame(b)
            self.num_of_binders = len(list(b.index))
            self.length = len(list(b.columns))
            self.b_strings = [''.join(list(b.iloc[n, :])) for n in range(self.num_of_binders)]
        
    def set_peptides(self, p_path):
        p = pd.read_csv(p_path)
        self.__store_vaues(p, None)
    
    def set_top_binders(self, b_path):
        b = pd.read_csv(b_path)
        self.__store_values(None, b)
    
    def calculate_TSS(self):
        if self.p is None or self.b is None:
            raise Exception("List of peptides or binders not set")
        if len(list(p.columns)) != len(list(b.columns)):
            raise Exception("Sequence length of peptides and binders must be equal")
        np_ss = np.zeros(shape=(self.num_of_peptides, len(self.cluster_keys)))
        for m in range(self.num_of_peptides):
            for i, key in enumerate(self.cluster_keys):
                total_score = 0
                for n in range(self.num_of_binders):
                    total_score += sum(self.matrices[key].loc[self.p_strings[m][l], \
                                       self.b_strings[n][l]] for l in range(self.length) \
                                       if self.p_strings[m] is not self.b_strings[n])
                np_ss[m][i] = total_score
            print(str(m), str(self.num_of_peptides))
        similarity_scores = pd.DataFrame(np_ss, index = self.p_strings, columns = self.cluster_keys)
        self.tss_df = similarity_scores
        return similarity_scores
    
    def predict_binding_affinity(self, binding_affinity_data_path):
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
        data = np.zeros([self.num_of_peptides, 2])
        data[:,0] = Y['Binding_affinity']
        data[:,1] = X['prediction']
        score = np.corrcoef(data[:,0], data[:,1])[0, 1]
        predictions = pd.DataFrame(data, index = X.index, columns = ['Binding Affinity', 'Predicted'])
        self.ba_linreg_predictions = predictions
        self.ba_linreg_score = score
        return predictions, score
    

p = pd.read_csv("./peptides_full.csv")
b = pd.read_csv("./top_binders.csv")

ac = AlignmentCalculator(p, b)
tss = ac.calculate_TSS()
predictions = ac.predict_binding_affinity('largest_mhc0_1.csv')