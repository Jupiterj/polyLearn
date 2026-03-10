import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


import json
from pandas import json_normalize
from sklearn import linear_model
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import mean_squared_error
from MorganAnalysis import generate_fingerprints,convert_smiles
from rdkit import Chem

#####################################################
## Copy and Paste ##
#####################################################
def get_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        
    # Import properties data into a pandas dataframe, keeping the polymer name as metadata
    # Ignores polymers without name (polymer blends)
    df= json_normalize(data,
                        ['polymer_data','property_summaries','properties'],
                        [['polymer_data','polymer_name'],['polymer_data','smiles'],['polymer_data','formula_weight']],
                        errors='ignore',
                        ).dropna(subset=['polymer_data.polymer_name'])
    df = df[['polymer_data.polymer_name','polymer_data.smiles','polymer_data.formula_weight','property_name','property_value_median']]

    # Averages values from duplicates
    df = df.groupby(['polymer_data.polymer_name','polymer_data.smiles','polymer_data.formula_weight','property_name'],sort=False,dropna=False).mean()
    df = df.reset_index()

    # Converts each property into its own column using the median as the value
    df = df.pivot(index=['polymer_data.polymer_name','polymer_data.smiles','polymer_data.formula_weight'], columns='property_name', values='property_value_median')

    # Reset the index to turn 'polymer_data.polymer_name' back into a column
    df = df.reset_index()

    # Returns dataframe
    return df

def kfold_val(x,y,alphas,_model,n_splits=10):
    alpha_scores = np.zeros_like(alphas)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for idx,alpha in enumerate(alphas):
        # Initialize the lasso_reg model
        model = _model(alpha=alpha)

        # Perform Cross Validation
        scores = cross_val_score(model, x, y, cv=kf, scoring='neg_root_mean_squared_error')
        alpha_scores[idx] = np.mean(scores) 
    alpha_best = alphas[np.where(alpha_scores==alpha_scores.max())[0][0]]
    # print('Alpha scores:',alpha_scores)
    print('Best alpha:',alpha_best)
    return alpha_best


# Main body
data_filename = 'new_smiles.json'
# data_filename = 'poly_info_Glass transition temperature.json'
# data_filename = 'poly_info_Radiation resistance.json'
df = get_data(data_filename)

# Selects the polymers used for prediction
property_list = ['Melting temperature','Density','polymer_data.formula_weight']
prediction_list = ['Glass transition temperature']

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
print(y.shape[0],'polymers with desired properties')

rng = np.random.default_rng(0)
N = y.shape[0]
d = len(n) # number of "materials descriptors"

#####################################################
## Copy and Paste ##
#####################################################
y_train,y_test,X_train,X_test = train_test_split(y,x[:,n],test_size = 0.1,random_state=random_state)

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
# TODO : define optimizer (Adam , lr=1e-3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# TODO : define loss function (MSE)
loss_fn = nn.MSELoss()
...
#-----------------------------
#5)Trainingloop
#-----------------------------
for epoch in range(1,501):
    model.train()
    total_loss=0.0
    for xb,yb in train_loader:
        #TODO:forwardpass
        pred= model(xb)

        #TODO:computeloss
        loss= loss_fn(pred, yb)

        #TODO:backward+step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*xb.size(0)
    train_mse=total_loss/len(train_ds)
    
    #Evaluate MAE on test set
    model.eval()
    abs_err_sum=0.0
    with torch.no_grad():
        for xb,yb in test_loader:
            pred=model(xb)
            abs_err_sum+=torch.abs(pred-yb).sum().item()
    test_mae=abs_err_sum/len(test_ds)
    if epoch%10==0:
        print(f"Epoch {epoch:02d}|TrainMSE:{train_mse:.4f}|Test MAE:{test_mae:.4f}")