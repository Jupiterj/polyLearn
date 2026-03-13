# This is a linear rgession machine learning model for predicting polymer materials properties
import numpy as np
import json
from pandas import json_normalize
from sklearn import linear_model
from sklearn.model_selection import RepeatedKFold,KFold, GridSearchCV, cross_val_score,cross_validate,cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.morgan_ridge_analysis import generate_fingerprints
from rdkit import Chem
import matplotlib.pyplot as plt

def get_data(filename):
    # This file cleans up the extracted .json from PolyInfo
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

def compare_components(raw_file, ablation_file, prediction_list, property_list):
    # Main function for running the linear regression model and comparing the importance of different features for predicting the property of interest.
    df = get_data(raw_file)

    # For the sake of running this code, the property selection is hardcoded in, this can be improved on in a future iteration. 
    property_list = ['Melting temperature','Density','polymer_data.formula_weight']

    # Selects the properties used for prediction
    n = [
        [[0,1,2]], # only the materials properties
        [[0,1]],
        [[0,2]],
        [[1,2]],
        [slice(3, -1)], # only the smiles
        [slice(1, -1)] # all data
        ]

    labels = ['T_m, Density, Weight', 'T_m, Density', 'T_m, Weight', 'Density, Weight', 'SMILES only', 'All Features']
    random_state1 = 78
    random_state2 = 63


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

    # Include SMILES fingerprinting data
    # load the ablation data for the fingerprinting
    with open(ablation_file, 'r') as f:
        data = json.load(f)

    # Use the parameters that gave the highest R^2 for predicting glass transition temperature
    max_idx = np.argmax([entry["r2"] for entry in data])
    par_list = ['radius','fpSize','fp_type']
    pars = {key:data[max_idx][key] for key in par_list}

    mol_list = [Chem.MolFromSmiles(smiles) for smiles in polymer_smiles[mask]]
    fp_data = generate_fingerprints(mol_list,**pars)
    x = np.concatenate((x, fp_data), axis=1)

    # standardize the x and y values
    x = (x-np.mean(x,axis = 0))/np.maximum(np.std(x,axis = 0),+1e-8)
    y = (y-np.mean(y,axis = 0))/np.std(y,axis = 0)

    models = [
        ['Lasso',
        linear_model.Lasso(max_iter=10**6),
        {
            "model__alpha":np.logspace(-3, 3, 20,endpoint=True, base=10.0),
            #  "model__alpha":np.linspace(0.01, 2, 100)
            #  "alpha":np.linspace(99.01, 101, 10)
        },
        ],
        ['Ridge',
        linear_model.Ridge(max_iter=10**6),
        {
            "model__alpha":np.logspace(-3, 3, 20,endpoint=True, base=10.0),
            #    "model__alpha":np.linspace(9.01, 11, 100)
            },
            ]
    ]
    performance = []
    for idx,properties in enumerate(n):
        print('Components Evaluated:',labels[idx])
        for model_name,model,param_grid in models:
            print(model_name)
            x2 = np.concatenate([x[:,slices] for slices in properties], axis=1)
            # print(x2.shape)

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])
            # Perform repeated nested kfold cross validations to optimize hyper-parameters and test model generalizability
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state1)
            grid_search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error"
            )
            # Outer loop: model evaluation
            outer_cv = RepeatedKFold(
                n_splits=5,
                n_repeats=10,
                random_state=random_state2
            )
            scores = cross_validate(
                grid_search,
                x2,      # feature matrix
                y,      # target values
                cv=outer_cv,
                scoring={
                    "r2": "r2",
                    "mse": "neg_root_mean_squared_error"
                },
                return_estimator=True
            )


            all_preds = []
            all_actual = []

            for train_idx, test_idx in outer_cv.split(x2):

                X_train, X_test = x2[train_idx], x2[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_

                preds = best_model.predict(X_test)

                all_preds.extend(preds)
                all_actual.extend(y_test)
            # y_pred = cross_val_predict(model, x2, y, cv=outer_cv)
            all_preds = np.array(all_preds)
            all_actual = np.array(all_actual)

            # Compute R^2
            r2 = r2_score(all_actual,all_preds)

            fig, ax = plt.subplots(figsize=(5,5))

            # Scatter
            ax.scatter(all_actual, all_preds, color='red', s=15)

            # Perfect prediction line
            lims = [
                min(all_actual.min(), all_preds.min()),
                max(all_actual.max(), all_preds.max())
            ]

            ax.plot(lims, lims, 'k-', linewidth=2)

            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # Labels
            ax.set_xlabel("Experimental",fontsize = 14)
            ax.set_ylabel("Predicted",fontsize = 14)

            # Text box
            text = f"""Property: {labels[idx]}
            Regularization: {model_name}
            $R^2$ = {r2:.3f}"""

            ax.text(
                0.3, 0.05,
                text,
                transform=ax.transAxes,
                fontsize=13
            )

            plt.tight_layout()
            plt.show()

            print("R²: %.3f ± %.3f" % (scores["test_r2"].mean(), scores["test_r2"].std()))
            print("RMSE: %.3f ± %.3f" % (-scores["test_mse"].mean(), scores["test_mse"].std()))
            result = {
                'evaluation': labels[idx],
                'model': model_name,
                'r2': scores["test_r2"].mean(),
                'r2_std': scores["test_r2"].std(),
                'rmse': -scores["test_mse"].mean(),
                'rmse_std': scores["test_mse"].std()
                        }
            performance.append(result)
    return performance


