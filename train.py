import deepchem as dc
import numpy as np
from itertools import product
from tqdm import tqdm
import os


def load_data(target, split, i):
    
    d = np.load(os.path.join("./data", str(target), str(split), str(i)+".npz"))
    
    train_dataset = dc.data.NumpyDataset(X=d["x_tr"], y=d["y_tr"].reshape(-1,1))
    test_dataset = dc.data.NumpyDataset(X=d["x_te"],  y=d["y_te"].reshape(-1,1))
    sim = d["sim"]
    
    return train_dataset, test_dataset, sim


def train_single(train_dataset, test_dataset, save_path):
    
    reg = dc.models.MultitaskRegressor(n_tasks=1, n_features=1024, layer_sizes=[500, 500, 200], uncertainty=True)
    reg.fit(train_dataset, nb_epoch=200)
    
    y_pred_real = reg.predict(test_dataset)
    y_pred_dropout, y_std = reg.predict_uncertainty(test_dataset)
    
    np.savez(file=save_path, 
             y_pred=y_pred_real.flatten(),
             y_drop_pred=y_pred_dropout.flatten(), 
             unc=y_std.flatten())

def train(data_dir):
    
    targets = os.listdir(data_dir)
    splits = ['bac', 'cv']
    split_ids = list(range(5))
    
    runs = list(product(targets, splits, split_ids))
    
    for target, split, split_id in tqdm(runs):
        
        save_path = os.path.join(data_dir, target, split, f"large_result_{split_id}.npz")
        train_dataset, test_dataset, _ = load_data(target=target, split=split, i=split_id)
   
        train_single(train_dataset=train_dataset, test_dataset=test_dataset, save_path=save_path)
    
    
if __name__ == "__main__":
    
    train('./data')