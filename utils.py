import os
import numpy as np
import pandas as pd
import deepchem as dc

import matplotlib.pyplot as plt


def load_data(target, split, i):
    
    d = np.load(os.path.join("./data", str(target), str(split), str(i)+".npz"))
    
    train_dataset = dc.data.NumpyDataset(X=d["x_tr"], y=d["y_tr"].reshape(-1,1))
    test_dataset = dc.data.NumpyDataset(X=d["x_te"], y=d["y_te"].reshape(-1,1))
    sim = d["sim"]
    
    return train_dataset, test_dataset, sim


def load_new_data(target, split, i):
    
    d = np.load(os.path.join("./new_data", str(target), str(split), str(i)+".npz"))
    
    train_dataset = dc.data.NumpyDataset(X=d["x_tr"], y=d["y_tr"].reshape(-1,1))
    test_dataset = dc.data.NumpyDataset(X=d["x_te"], y=d["y_te"].reshape(-1,1))
    sim = d["sim"]
    train_uids = d['uid_tr']
    test_uids = d['uid_te']
    sim_uids = d['sim_uids']
    
    return train_dataset, test_dataset, sim, train_uids, test_uids, sim_uids


def get_value(string):
    
    return float(string.split('±')[0].strip())


def highlight_min(data, color='yellow'):
    
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    data = data.apply(get_value).astype(float)
    
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        sub_data = data[filter(lambda x: 'unc' not in x, data.index)]
        is_max = data == sub_data.min()
        return [attr if v  else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.min().min()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
    
    
def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    #remove % and cast to float
    data = data.apply(get_value).astype(float)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)