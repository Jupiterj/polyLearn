import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pandas as pd

from src.polylearn_linear import get_data
import json
from sklearn.model_selection import train_test_split
from src.morgan_ridge_analysis import generate_fingerprints
from rdkit import Chem
from tqdm import tqdm

import matplotlib.pyplot as plt


# This is the main function that trains a simple MLP on all the material features. The code is adapted from the model from class, but with modifications for our data. 
def train_nn(raw_file, ablation_file, prediction_list, property_list,epochs = 400,weight_decay = 0,lr=1e-3):
    df = get_data(raw_file)

    # For the sake of running this code, the property selection is hardcoded in, this can be improved on in a future iteration. 
    property_list = ['Melting temperature','Density','polymer_data.formula_weight']

    # Selects the properties used for prediction
    n = [0,1,2]

    random_state = None

    # # Determines the number of datapoints for each property
    # properties = np.array(df.columns.tolist())[2:]
    # d_points = np.count_nonzero(~np.isnan(df.to_numpy()[:,2:].astype(float)),axis = 0)
    # top_5_properties_arg = np.flip(np.argsort(d_points))[:10]
    # print('Top 10 properties:',properties[top_5_properties_arg])
    # print('Counts:',d_points[top_5_properties_arg])

    # Converts to numpy array
    array = df.to_numpy()[:,2:].astype(float)
    properties = df.columns.to_numpy()[2:]
    polymer_names = df['polymer_data.polymer_name'].to_numpy()
    polymer_smiles = df['polymer_data.smiles'].to_numpy()

    x = array[:,np.isin(properties,property_list)]
    y = array[:,np.isin(properties,prediction_list)]

    # Restricts polymer property array to polymers that have data on all the desired properties
    mask = np.logical_and(~np.isnan(x).any(axis=1),~np.isnan(y).any(axis=1))
    x = x[mask]
    y = y[mask]
    #print(y.shape[0],'polymers with desired properties')

    # Incorporation of SMILES data 
    #print('\n\nOnly with smiles data:')
    # load the ablation data for the fingerprinting
    with open(ablation_file, 'r') as f:
        data = json.load(f)

    # Use the parameters that gave the highest R^2 for predicting glass transition temperature
    max_idx = np.argmax([entry["r2"] for entry in data])
    par_list = ['radius','fpSize','fp_type']
    pars = {key:data[max_idx][key] for key in par_list}

    mol_list = [Chem.MolFromSmiles(smiles) for smiles in polymer_smiles[mask]]
    fp_data = generate_fingerprints(mol_list,**pars)
    x = np.concatenate((x[:,n], fp_data), axis=1)

    rng = np.random.default_rng(0)
    N = y.shape[0]
    d = x.shape[1] # number of "materials descriptors"

    #####################################################
    ## Copy and Paste ##
    #####################################################
    y_train,y_test,X_train,X_test = train_test_split(y,x,test_size = 0.2,random_state=random_state)

    # standardize the x and y values
    mu=X_train.mean(axis=0,keepdims=True)
    sigma=X_train.std(axis=0,keepdims=True)+1e-8
    X_train_std=(X_train-mu)/sigma
    X_test_std=(X_test-mu)/sigma

    train_ds=TensorDataset(torch.from_numpy(X_train_std).float(),torch.
        from_numpy(y_train).float())
    test_ds=TensorDataset(torch.from_numpy(X_test_std).float(),torch.from_numpy
                        (y_test).float())
    train_loader=DataLoader(train_ds,batch_size=64,shuffle=True)
    test_loader=DataLoader(test_ds,batch_size=64,shuffle=False)
    # -----------------------------
    # 4) Define a small MLP
    # -----------------------------
    class MLP(nn.Module):
        def __init__(self,d_in):
            super().__init__()
            self.net=nn.Sequential(
                nn.Linear(d_in,64),
                nn.ReLU(),
                nn.Linear(64,64),
                nn.ReLU(),
                nn.Linear(64,1),
            )
        def forward(self,x):
            return self.net(x)
    model=MLP(d)
    # define optimizer (Adam , lr=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)


    # define loss function (MSE)
    loss_fn = nn.MSELoss()
    #-----------------------------
    #5)Trainingloop
    #-----------------------------
    performance = []
    total_epochs = epochs
    p_bar = tqdm(total=total_epochs, desc="Training Neural Network")
    for epoch in range(1,total_epochs+1):
        model.train()
        total_loss=0.0
        for xb,yb in train_loader:
            # forwardpass
            pred= model(xb)

            # computeloss
            loss= loss_fn(pred, yb)

            # backward+step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()*xb.size(0)
        train_mse=total_loss/len(train_ds)
        
        #Evaluate MAE on test set
        model.eval()
        abs_err_sum=0.0
        square_err_sum = 0.0
        with torch.no_grad():
            for xb,yb in test_loader:
                pred=model(xb)
                abs_err_sum+=torch.abs(pred-yb).sum().item()
                square_err_sum+=torch.square(pred-yb).sum().item()
        test_mae=abs_err_sum/len(test_ds)
        test_rmse=(square_err_sum/len(test_ds))**0.5
        # if epoch%10==0:
        #     print(f"Epoch {epoch:02d}|Train MSE:{train_mse:.4f}|Test MAE:{test_mae:.4f}")
        result = {
            'epoch': epoch,
            'train_mse': train_mse,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        }
        p_bar.update(1)
        performance.append(result)
    p_bar.close()
    return performance
def fig4(performance):
    df = pd.DataFrame(performance)
    print(f"Epoch {df["epoch"].iloc[-1]:02d}|Train MSE:{df["train_mse"].iloc[-1]:.4f}|Test MAE:{df["test_mae"].iloc[-1]:.4f}|Test RMSE:{df["test_rmse"].iloc[-1]:.4f}")
    plt.figure()

    plt.plot(df["train_mse"], label="Training Loss")
    plt.plot(df["test_mae"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()

    plt.show()
