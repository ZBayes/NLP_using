import pandas as pd
import numpy as np
import jieba


def data_loader(path, file_type='excel', import_type='pd'):
    if file_type == 'excel' and import_type == 'pd':
        return pd.read_excel(path, header=None, index=None)
    elif file_type == 'npy':
        return np.load(path)
    else:
        return null


def word_cut(x):
    return jieba.cut(x)
