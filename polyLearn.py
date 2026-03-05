# This is a machine learning model for predicting polymer materials properties
import pandas as pd
import numpy as np
import json
from pandas import json_normalize
from sklearn import linear_model
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from MorganAnalysis import generate_fingerprints,convert_smiles
from rdkit import Chem


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

random_state = 12

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


# standardize the x and y values
x = (x-np.mean(x,axis = 0))/np.std(x,axis = 0)
y = (y-np.mean(y,axis = 0))/np.std(y,axis = 0)


# Perfom Linear and Regularized Linear Regressions on full dataset
lin_reg = linear_model.LinearRegression().fit(y, x[:,n])
ridge_reg = linear_model.Ridge(alpha=0.1).fit(y, x[:,n])
lasso_reg = linear_model.Lasso(alpha=0.1).fit(y, x[:,n])
print('Linear Regression R^2',lin_reg.score(y,x[:,n]))
print('Ridge Regression R^2',ridge_reg.score(y,x[:,n]))
print('Lasso Regression R^2',lasso_reg.score(y,x[:,n]),end = '\n\n')

# Perfom test split to optimized alpha and to assess model generalizability
y_train,y_test,x_train,x_test = train_test_split(y,x[:,n],test_size = 0.1,random_state=random_state)

# Perfom k-fold cross validation to optimize alpha for both lasso and ridge regression
alphas = np.arange(0.01,2,0.01)

alpha_ridge = kfold_val(x_train,y_train,alphas,linear_model.Ridge,n_splits=10)
alpha_lasso = kfold_val(x_train,y_train,alphas,linear_model.Lasso,n_splits=10)

# Train model using optimized alpha values
lin_reg = linear_model.LinearRegression().fit(y_train, x_train)
ridge_reg = linear_model.Ridge(alpha=alpha_ridge).fit(y_train, x_train)
lasso_reg = linear_model.Lasso(alpha=alpha_lasso).fit(y_train, x_train)
print('\nTest splits:')
print('Linear Regression R^2',lin_reg.score(y_test,x_test))
print('Ridge Regression R^2',ridge_reg.score(y_test,x_test))
print('Lasso Regression R^2',lasso_reg.score(y_test,x_test))

# Incorporation of SMILES data into model
print('\n\nWith smiles data:')
# load the ablation data for the fingerprinting
with open("kfold_ablation_GTT.json", 'r') as f:
    data = json.load(f)

# Use the parameters that gave the highest R^2 for predicting glass transition temperature
max_idx = np.argmax([entry["r2"] for entry in data])
par_list = ['radius','fpSize','fp_type']
pars = {key:data[max_idx][key] for key in par_list}

mol_list = [Chem.MolFromSmiles(smiles) for smiles in polymer_smiles[mask]]
# mol_list,_ = convert_smiles(data_filename,prediction_list[0])
fp_data = generate_fingerprints(mol_list,**pars)


## Perform regression only on the smiles data
# x = np.concatenate((x[:,n], fp_data), axis=1)
x = fp_data


# Perfom Linear and Regularized Linear Regressions on full dataset
lin_reg = linear_model.LinearRegression().fit(y, x)
ridge_reg = linear_model.Ridge(alpha=0.1).fit(y, x)
lasso_reg = linear_model.Lasso(alpha=0.1).fit(y, x)
print('Linear Regression R^2',lin_reg.score(y,x))
print('Ridge Regression R^2',ridge_reg.score(y,x))
print('Lasso Regression R^2',lasso_reg.score(y,x),end = '\n\n')

# Perfom test split to optimized alpha and to assess model generalizability
y_train,y_test,x_train,x_test = train_test_split(y,x,test_size = 0.1,random_state=random_state)

# Perfom k-fold cross validation to optimize alpha for both lasso and ridge regression
alphas = np.arange(0.01,2,0.01)

# Restricts polymer property array to polymers that have data on all the desired properties
alpha_ridge = kfold_val(x_train,y_train,alphas,linear_model.Ridge,n_splits=10)
alpha_lasso = kfold_val(x_train,y_train,alphas,linear_model.Lasso,n_splits=10)

# Train model using optimized alpha values
lin_reg = linear_model.LinearRegression().fit(y_train, x_train)
ridge_reg = linear_model.Ridge(alpha=alpha_ridge).fit(y_train, x_train)
lasso_reg = linear_model.Lasso(alpha=alpha_lasso).fit(y_train, x_train)
print('\nTest splits:')
print('Linear Regression R^2',lin_reg.score(y_test,x_test))
print('Ridge Regression R^2',ridge_reg.score(y_test,x_test))
print('Lasso Regression R^2',lasso_reg.score(y_test,x_test))


## Combined data
x = np.concatenate((x[:,n], fp_data), axis=1)
# x = fp_data


# Perfom Linear and Regularized Linear Regressions on full dataset
lin_reg = linear_model.LinearRegression().fit(y, x)
ridge_reg = linear_model.Ridge(alpha=0.1).fit(y, x)
lasso_reg = linear_model.Lasso(alpha=0.1).fit(y, x)
print('Linear Regression R^2',lin_reg.score(y,x))
print('Ridge Regression R^2',ridge_reg.score(y,x))
print('Lasso Regression R^2',lasso_reg.score(y,x),end = '\n\n')

# Perfom test split to optimized alpha and to assess model generalizability
y_train,y_test,x_train,x_test = train_test_split(y,x,test_size = 0.1,random_state=random_state)

# Perfom k-fold cross validation to optimize alpha for both lasso and ridge regression
alphas = np.arange(0.01,2,0.01)

# Restricts polymer property array to polymers that have data on all the desired properties
alpha_ridge = kfold_val(x_train,y_train,alphas,linear_model.Ridge,n_splits=10)
alpha_lasso = kfold_val(x_train,y_train,alphas,linear_model.Lasso,n_splits=10)

# Train model using optimized alpha values
lin_reg = linear_model.LinearRegression().fit(y_train, x_train)
ridge_reg = linear_model.Ridge(alpha=alpha_ridge).fit(y_train, x_train)
lasso_reg = linear_model.Lasso(alpha=alpha_lasso).fit(y_train, x_train)
print('\nTest splits:')
print('Linear Regression R^2',lin_reg.score(y_test,x_test))
print('Ridge Regression R^2',ridge_reg.score(y_test,x_test))
print('Lasso Regression R^2',lasso_reg.score(y_test,x_test))
