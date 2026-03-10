# This is a machine learning model for predicting polymer materials properties
import pandas as pd
import numpy as np
import json
from pandas import json_normalize
from sklearn import linear_model
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import mean_squared_error
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


# standardize the x and y values
x = (x-np.mean(x,axis = 0))/np.std(x,axis = 0)
y = (y-np.mean(y,axis = 0))/np.std(y,axis = 0)


# Perfom Linear and Regularized Linear Regressions on full dataset
lin_reg = linear_model.LinearRegression().fit(x[:,n],y)
ridge_reg = linear_model.Ridge(alpha=0.1).fit(x[:,n],y)
lasso_reg = linear_model.Lasso(alpha=0.1).fit(x[:,n],y)
print('Full Dataset:')
print('Linear Regression R^2',lin_reg.score(x[:,n],y))
print('Ridge Regression R^2',ridge_reg.score(x[:,n],y))
print('Lasso Regression R^2',lasso_reg.score(x[:,n],y),end = '\n\n')


iterations = 100
R2_nosmiles_lin = np.zeros(iterations)
RMSE_nosmiles_lin = np.zeros(iterations)
R2_nosmiles_las = np.zeros(iterations)
RMSE_nosmiles_las = np.zeros(iterations)
R2_nosmiles_rid = np.zeros(iterations)
RMSE_nosmiles_rid = np.zeros(iterations)
R2_smiles_lin = np.zeros(iterations)
RMSE_smiles_lin = np.zeros(iterations)
R2_smiles_las = np.zeros(iterations)
RMSE_smiles_las = np.zeros(iterations)
R2_smiles_rid = np.zeros(iterations)
RMSE_smiles_rid = np.zeros(iterations)
R2_comb_lin = np.zeros(iterations)
RMSE_comb_lin = np.zeros(iterations)
R2_comb_las = np.zeros(iterations)
RMSE_comb_las = np.zeros(iterations)
R2_comb_rid = np.zeros(iterations)
RMSE_comb_rid = np.zeros(iterations)

# Perform repeated test splits with different random states to generate statistics on model performance
for training_idx in range(iterations):
    print('\nIteration',training_idx)
    # Perfom test split to optimized alpha and to assess model generalizability
    y_train,y_test,x_train,x_test = train_test_split(y,x[:,n],test_size = 0.1,random_state=random_state)

    # Perfom k-fold cross validation to optimize alpha for both lasso and ridge regression
    alphas = np.arange(0.01,2,0.01)

    alpha_ridge = kfold_val(x_train,y_train,alphas,linear_model.Ridge,n_splits=10)
    alpha_lasso = kfold_val(x_train,y_train,alphas,linear_model.Lasso,n_splits=10)
    # alpha_ridge = 0.14
    # alpha_lasso = 0.05

    # Train model using optimized alpha values
    lin_reg = linear_model.LinearRegression().fit(x_train,y_train)
    ridge_reg = linear_model.Ridge(alpha=alpha_ridge).fit(x_train,y_train)
    lasso_reg = linear_model.Lasso(alpha=alpha_lasso).fit(x_train,y_train)
    print('\nTest set:')
    print('Linear Regression R^2',lin_reg.score(x_test,y_test))
    print('Ridge Regression R^2',ridge_reg.score(x_test,y_test))
    print('Lasso Regression R^2',lasso_reg.score(x_test,y_test))

    R2_nosmiles_lin[training_idx] = lin_reg.score(x_test,y_test)
    RMSE_nosmiles_lin[training_idx] = np.sqrt(mean_squared_error(y_test, lin_reg.predict(x_test)))
    R2_nosmiles_rid[training_idx] = ridge_reg.score(x_test,y_test)
    RMSE_nosmiles_rid[training_idx] = np.sqrt(mean_squared_error(y_test, ridge_reg.predict(x_test)))
    R2_nosmiles_las[training_idx] = lasso_reg.score(x_test,y_test)
    RMSE_nosmiles_las[training_idx] = np.sqrt(mean_squared_error(y_test, lasso_reg.predict(x_test)))

    # Incorporation of SMILES data into model
    print('\n\nOnly with smiles data:')
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
    lin_reg = linear_model.LinearRegression().fit(x,y)
    ridge_reg = linear_model.Ridge(alpha=0.1).fit(x,y)
    lasso_reg = linear_model.Lasso(alpha=0.1).fit(x,y)
    print('Linear Regression R^2',lin_reg.score(x,y))
    print('Ridge Regression R^2',ridge_reg.score(x,y))
    print('Lasso Regression R^2',lasso_reg.score(x,y),end = '\n\n')

    # Perfom test split to optimized alpha and to assess model generalizability
    y_train,y_test,x_train,x_test = train_test_split(y,x,test_size = 0.1,random_state=random_state)

    # Perfom k-fold cross validation to optimize alpha for both lasso and ridge regression
    alphas = np.arange(0.01,2,0.01)

    # Restricts polymer property array to polymers that have data on all the desired properties
    alpha_ridge = kfold_val(x_train,y_train,alphas,linear_model.Ridge,n_splits=10)
    alpha_lasso = kfold_val(x_train,y_train,alphas,linear_model.Lasso,n_splits=10)
    # alpha_ridge = 0.07
    # alpha_lasso = 0.01

    # Train model using optimized alpha values
    lin_reg = linear_model.LinearRegression().fit(x_train,y_train)
    ridge_reg = linear_model.Ridge(alpha=alpha_ridge).fit(x_train,y_train)
    lasso_reg = linear_model.Lasso(alpha=alpha_lasso).fit(x_train,y_train)
    print('\nTest set:')
    print('Linear Regression R^2',lin_reg.score(x_test,y_test))
    print('Ridge Regression R^2',ridge_reg.score(x_test,y_test))
    print('Lasso Regression R^2',lasso_reg.score(x_test,y_test))

    R2_smiles_lin[training_idx] = lin_reg.score(x_test,y_test)
    RMSE_smiles_lin[training_idx] = np.sqrt(mean_squared_error(y_test, lin_reg.predict(x_test)))
    R2_smiles_rid[training_idx] = ridge_reg.score(x_test,y_test)
    RMSE_smiles_rid[training_idx] = np.sqrt(mean_squared_error(y_test, ridge_reg.predict(x_test)))
    R2_smiles_las[training_idx] = lasso_reg.score(x_test,y_test)
    RMSE_smiles_las[training_idx] = np.sqrt(mean_squared_error(y_test, lasso_reg.predict(x_test)))

    ## Combined data
    print('\nCombined data:')
    x = np.concatenate((x[:,n], fp_data), axis=1)
    # x = fp_data


    # Perfom Linear and Regularized Linear Regressions on full dataset
    lin_reg = linear_model.LinearRegression().fit(x,y)
    ridge_reg = linear_model.Ridge(alpha=0.1).fit(x,y)
    lasso_reg = linear_model.Lasso(alpha=0.1).fit(x,y)
    print('Linear Regression R^2',lin_reg.score(x,y))
    print('Ridge Regression R^2',ridge_reg.score(x,y))
    print('Lasso Regression R^2',lasso_reg.score(x,y),end = '\n\n')

    # Perfom test split to optimized alpha and to assess model generalizability
    y_train,y_test,x_train,x_test = train_test_split(y,x,test_size = 0.1,random_state=random_state)

    # Perfom k-fold cross validation to optimize alpha for both lasso and ridge regression
    alphas = np.arange(0.01,2,0.01)

    # Restricts polymer property array to polymers that have data on all the desired properties
    alpha_ridge = kfold_val(x_train,y_train,alphas,linear_model.Ridge,n_splits=10)
    alpha_lasso = kfold_val(x_train,y_train,alphas,linear_model.Lasso,n_splits=10)
    # alpha_ridge = 0.05
    # alpha_lasso = 0.01

    # Train model using optimized alpha values
    lin_reg = linear_model.LinearRegression().fit(x_train,y_train)
    ridge_reg = linear_model.Ridge(alpha=alpha_ridge).fit(x_train,y_train)
    lasso_reg = linear_model.Lasso(alpha=alpha_lasso).fit(x_train,y_train)
    print('\nTest set:')
    print('Linear Regression R^2',lin_reg.score(x_test,y_test))
    print('Ridge Regression R^2',ridge_reg.score(x_test,y_test))
    print('Lasso Regression R^2',lasso_reg.score(x_test,y_test))

    R2_comb_lin[training_idx] = lin_reg.score(x_test,y_test)
    RMSE_comb_lin[training_idx] = np.sqrt(mean_squared_error(y_test, lin_reg.predict(x_test)))
    R2_comb_rid[training_idx] = ridge_reg.score(x_test,y_test)
    RMSE_comb_rid[training_idx] = np.sqrt(mean_squared_error(y_test, ridge_reg.predict(x_test)))
    R2_comb_las[training_idx] = lasso_reg.score(x_test,y_test)
    RMSE_comb_las[training_idx] = np.sqrt(mean_squared_error(y_test, lasso_reg.predict(x_test)))

print('No Smiles:')
print('Linear R^2 =',np.mean(R2_nosmiles_lin),'+-',np.std(R2_nosmiles_lin))
print('Linear RMSE =',np.mean(RMSE_nosmiles_lin),'+-',np.std(RMSE_nosmiles_lin))
print('Ridge R^2 =',np.mean(R2_nosmiles_rid),'+-',np.std(R2_nosmiles_rid))
print('Ridge RMSE =',np.mean(RMSE_nosmiles_rid),'+-',np.std(RMSE_nosmiles_rid))
print('Lasso R^2 =',np.mean(R2_nosmiles_las),'+-',np.std(R2_nosmiles_las))
print('Lasso RMSE =',np.mean(RMSE_nosmiles_las),'+-',np.std(RMSE_nosmiles_las))
print('\nSmiles:')
print('Linear R^2 =',np.mean(R2_smiles_lin),'+-',np.std(R2_smiles_lin))
print('Linear RMSE =',np.mean(RMSE_smiles_lin),'+-',np.std(RMSE_smiles_lin))
print('Ridge R^2 =',np.mean(R2_smiles_rid),'+-',np.std(R2_smiles_rid))
print('Ridge RMSE =',np.mean(RMSE_smiles_rid),'+-',np.std(RMSE_smiles_rid))
print('Lasso R^2 =',np.mean(R2_smiles_las),'+-',np.std(R2_smiles_las))
print('Lasso RMSE =',np.mean(RMSE_smiles_las),'+-',np.std(RMSE_smiles_las))
print('\nCombined:')
print('Linear R^2 =',np.mean(R2_comb_lin),'+-',np.std(R2_comb_lin))
print('Linear RMSE =',np.mean(RMSE_comb_lin),'+-',np.std(RMSE_comb_lin))
print('Ridge R^2 =',np.mean(R2_comb_rid),'+-',np.std(R2_comb_rid))
print('Ridge RMSE =',np.mean(RMSE_comb_rid),'+-',np.std(RMSE_comb_rid))
print('Lasso R^2 =',np.mean(R2_comb_las),'+-',np.std(R2_comb_las))
print('Lasso RMSE =',np.mean(RMSE_comb_las),'+-',np.std(RMSE_comb_las))
