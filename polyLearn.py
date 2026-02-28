# This is a machine learning model for predicting polymer materials properties
import pandas as pd
import numpy as np
import json
from pandas import json_normalize
from sklearn import linear_model
from sklearn.model_selection import train_test_split,KFold,cross_val_score


def get_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        
    # Import properties data into a pandas dataframe, keeping the polymer name as metadata
    # Ignores polymers without name (polymer blends)
    df= json_normalize(data,
                        ['polymer_data','property_summaries','properties'],
                        [['polymer_data','polymer_name'],['polymer_data','smiles']],
                        errors='ignore',
                        ).dropna(subset=['polymer_data.polymer_name'])
    df = df[['polymer_data.polymer_name','polymer_data.smiles','property_name','property_value_median']]

    # Averages values from duplicates
    df = df.groupby(['polymer_data.polymer_name','polymer_data.smiles','property_name'],sort=False,dropna=False).mean()
    df = df.reset_index()

    # Converts each property into its own column using the median as the value
    df = df.pivot(index=['polymer_data.polymer_name','polymer_data.smiles'], columns='property_name', values='property_value_median')

    # Reset the index to turn 'polymer_data.polymer_name' back into a column
    df = df.reset_index()

    # Returns dataframe
    return df


# Main body
data_filename = 'poly_info_Glass transition temperature.json'
# data_filename = 'poly_info_Radiation resistance.json'
df = get_data(data_filename)

# Determines the number of datapoints for each property
properties = np.array(df.columns.tolist())[2:]
d_points = np.count_nonzero(~np.isnan(df.to_numpy()[:,2:].astype(float)),axis = 0)

top_5_properties_arg = np.flip(np.argsort(d_points))[:5]

print('Top 5 properties:',properties[top_5_properties_arg])
print('Counts:',d_points[top_5_properties_arg])

# Converts to numpy array
array = df.to_numpy()[:,2:].astype(float)
properties = df.columns.to_numpy()[2:]
polymer_names = df['polymer_data.polymer_name'].to_numpy()
polymer_smiles = df['polymer_data.smiles'].to_numpy()

# Check dimensions of all outputs
# print(array.shape)
# print(properties.shape)
# print(polymer_names.shape)
# print(polymer_smiles.shape)

# Selects the polymers used for prediction
# property_list = ['Melting temperature','Density','Heat of fusion']
property_list = ['Melting temperature','Density']
prediction_list = ['Glass transition temperature']

# Selects the properties used for prediction
# n = [0,1,2]
n = [0,1]

x = array[:,np.isin(properties,property_list)]
y = array[:,np.isin(properties,prediction_list)]

# Restricts polymer property array to polymers that have data on all the desired properties
mask = np.logical_and(~np.isnan(x).any(axis=1),~np.isnan(y).any(axis=1))
x = x[mask]
y = y[mask]

print(np.average(y))
print(y.shape[0],'polymers with desired properties')

# Perfom Linear and Regularized Linear Regressions on full dataset
lin_reg = linear_model.LinearRegression().fit(y, x[:,n])
ridge_reg = linear_model.Ridge(alpha=1.0).fit(y, x[:,n])
lasso_reg = linear_model.Lasso(alpha=1.0).fit(y, x[:,n])
print('Linear Regression R^2',lin_reg.score(y,x[:,n]))
print('Ridge Regression R^2',ridge_reg.score(y,x[:,n]))
print('Lasso Regression R^2',lasso_reg.score(y,x[:,n]),end = '\n\n')


# Perfom k-fold cross validation to optimize alpha for both lasso and ridge regression
alphas = np.array([0.1,0.5,1.0,2.0,5.0,100.0,200.0,100000])

alpha_scores_lasso = np.zeros_like(alphas)
k=10
kf = KFold(n_splits=k, shuffle=True, random_state=78)
for idx,alpha in enumerate(alphas):
    # Initialize the lasso_reg model
    model = linear_model.Lasso(alpha=alpha)

    # Perform Cross Validation
    scores = cross_val_score(model, x, y, cv=kf, scoring='neg_root_mean_squared_error')
    alpha_scores_lasso[idx] = np.mean(scores) 
print('alpha_scores_lasso:',alpha_scores_lasso)

alpha_scores_ridge = np.zeros_like(alphas)
k=10
kf = KFold(n_splits=k, shuffle=True, random_state=78)
for idx,alpha in enumerate(alphas):
    # Initialize the lasso_reg model
    model = linear_model.Ridge(alpha=alpha)

    # Perform Cross Validation
    scores = cross_val_score(model, x, y, cv=kf, scoring='neg_root_mean_squared_error')
    alpha_scores_ridge[idx] = np.mean(scores) 
print('alpha_scores_ridge:',alpha_scores_ridge)

alpha_lasso = alphas[np.argmax(alpha_scores_lasso)]
alpha_ridge = alphas[np.argmax(alpha_scores_ridge)]


# Perfom test split on optimized alpha to assess model generalizability
y_train,y_test,x_train,x_test = train_test_split(y,x,test_size = 0.1,random_state=12)
lin_reg = linear_model.LinearRegression().fit(y_train, x_train)
ridge_reg = linear_model.Ridge(alpha=alpha_ridge).fit(y_train, x_train)
lasso_reg = linear_model.Lasso(alpha=alpha_lasso).fit(y_train, x_train)
print('Test splits:')
print('Linear Regression R^2',lin_reg.score(y_test,x_test))
print('Ridge Regression R^2',ridge_reg.score(y_test,x_test))
print('Lasso Regression R^2',lasso_reg.score(y_test,x_test))
print(lin_reg.coef_)
print(ridge_reg.coef_)
print(lasso_reg.coef_)
