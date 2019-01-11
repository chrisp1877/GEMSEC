import pandas as pd
import numpy as np

def calculate_TSS(p,b):
    cluster_keys = ["a", "b", "c", "e", "h", "i", "p", "r"]
    matrices = dict.fromkeys(cluster_keys)
    for key in cluster_keys:
        matrices[key] = pd.read_csv("../cluster_Euclidean/improved/cluster_" + key + ".csv", index_col = 0)
    num_of_peptides = len(list(p.index))
    num_of_binders = len(list(b.index))
    length = len(list(p.columns))   
    p_strings = [''.join(list(p.iloc[m, :])) for m in range(num_of_peptides)]
    b_strings = [''.join(list(b.iloc[n, :])) for n in range(num_of_binders)]
   
    np_ss = np.zeros(shape=(num_of_peptides, len(cluster_keys)))
    for m in range(num_of_peptides):
        for i, key in enumerate(cluster_keys):
            total_score = 0
            for n in range(num_of_binders):
                total_score += sum(matrices[key].loc[p_strings[m][l], b_strings[n][l]] for l in range(length))
            np_ss[m][i] = total_score
    similarity_scores = pd.DataFrame(np_ss, index = p_strings, columns = cluster_keys)
    return similarity_scores