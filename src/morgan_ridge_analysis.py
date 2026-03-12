## This file serves as a hyperparameter ablation for the MorganFingerprint Generator built into RDKit. We first extract the relevant SMILES data and one desired property from a .json file. Then, we convert the SMILES data to a Mol object that is actionable by the RDKit API. We then conduct component ablation tests, assessing the R^2 and RMSE for a Regularized Ridge Regression model. The three hyperparameters that are given as options for component ablation are lambda/alpha for regularization algorithm, and radius and vector size for the Morgan algorithm. Additionally, values are simultaneously reported for three ways to represent fingerprints - in Bit form, count form, and heavy metal normalized count form. Time is recorded, as computational efficiency also becomes a critical aspect when these datasets get much larger. 

import json
from rdkit import Chem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time
from sklearn.decomposition import PCA
from tqdm import tqdm


# Start by putting data into format [MOL format, desired property]
def convert_smiles(file_name, property):
    with open(file_name, "r") as file:
        data = json.load(file)
    smiles_list = []
    prop_list = []
    mol_list = []
    for i in range(len(data["polymer_data"])):
        if "smiles" in data["polymer_data"][i]:
            smiles = data["polymer_data"][i]["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            prop = 0
            for j in range(len(data["polymer_data"][i]["property_summaries"])):
                for k in range(len(data["polymer_data"][i]["property_summaries"][j]["properties"])):
                    if data["polymer_data"][i]["property_summaries"][j]["properties"][k]["property_name"] == property and "property_value_median" in data["polymer_data"][i]["property_summaries"][j]["properties"][k]:
                        if mol is not None: 
                            prop =  data["polymer_data"][i]["property_summaries"][j]["properties"][k]["property_value_median"]
                            mol_list.append(mol)
                            prop_list.append(prop)
                            smiles_list.append(smiles)
                        else: 
                            print(f"There is an issue converting polymer {i+1} into MOL")

    prop_list = np.array(prop_list)

    # Verify that all SMILES can be converted into mol using RDKit
    id_check = [mol is not None for mol in mol_list]
    if False in id_check:
        print("There is at least one polymer that could not be converted from SMILES")
    else:
        "All polymers succcessfully converted from SMILES to Mol"
    return mol_list, prop_list

# This function takes in the converted Mol format list, and outputs a Morgan fingerprint via the Morgan algorithm with a radius and vector size. I also designed three different fingerprint representations -- bit, count, and heavy metal normalized count
def generate_fingerprints(mol_list, radius, fpSize, fp_type="count"):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    fp_list = np.zeros((len(mol_list), fpSize))

    # Decide which type of fingerprint we want to generate
    if fp_type == 'bit':
        for i in range(len(mol_list)):
            fp_list[i] = fpgen.GetFingerprintAsNumPy(mol_list[i])
    elif fp_type == 'count':
        for i in range(len(mol_list)):
            fp_list[i] = fpgen.GetCountFingerprintAsNumPy(mol_list[i])
    elif fp_type == 'normalized':
        size_normalization = np.array([mol.GetNumHeavyAtoms() for mol in mol_list])
        for i in range(len(mol_list)):
            count_fp = fpgen.GetCountFingerprintAsNumPy(mol_list[i])
            fp_list[i] = count_fp / size_normalization[i]
    return fp_list

# This function runs a full ablation study on the four hyperparameters we care about 1) alpha for ridge, radius and fpSize for the Morgan algorithm, and the actual fingerprint representation itself. I also built in an option to use different folds if we want to compare n-fold vs LeaveOneOut. 
def runablation(mol_list, prop_list, alpha_values, radius_values, fpSize_values, fp_types, foldmode):
    performance = []
    total_runs = len(alpha_values)*len(radius_values)*len(fpSize_values)*len(fp_types)
    pbar = tqdm(total=total_runs, desc="Running raw ablations")
    for i in range(len(alpha_values)):
        for j in range(len(radius_values)):
            for k in range(len(fpSize_values)):
                for l in range(len(fp_types)):
                    fp_data = generate_fingerprints(mol_list, radius_values[j], fpSize_values[k], fp_types[l])
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('ridge', linear_model.Ridge(alpha = alpha_values[i]))
                    ]) 
                    scoring = {
                        'r2': 'r2',
                        'rmse': 'neg_root_mean_squared_error'
                    }
                    start_time = time.time()
                    result = cross_validate(pipeline, fp_data, prop_list, scoring=scoring, cv=foldmode)
                    end_time = time.time()
                    r2_mean = np.mean(result['test_r2'])
                    rmse_mean = -1 * np.mean(result['test_rmse'])
                    r2_std = np.std(result['test_r2'])
                    rmse_std = np.std(result['test_rmse'])
                    # Store results
                    result = {
                        'alpha': alpha_values[i],
                        'radius': radius_values[j],
                        'fpSize': fpSize_values[k],
                        'fp_type': fp_types[l],
                        'r2': r2_mean,
                        'r2_std': r2_std,
                        'rmse': rmse_mean,
                        'rmse_std': rmse_std,
                        'time': end_time-start_time
                        }
                    pbar.update(1)
                    performance.append(result)
    pbar.close()
    return performance

# Similar function but now integrates a PCA component
def run_combined_PCAablation(mol_list, prop_list, alpha_values, radius_values, fpSize_values, fp_types, foldmode, n_components):
    performance = []
    n = 0
    total_runs = len(alpha_values)*len(radius_values)*len(fpSize_values)*len(fp_types)*len(n_components)
    pbar = tqdm(total=total_runs, desc="Running PCA ablations")
    for i in range(len(alpha_values)):
        for j in range(len(radius_values)):
            for k in range(len(fpSize_values)):
                for l in range(len(fp_types)):
                    for m in range(len(n_components)):
                        n +=1
                        fp_data = generate_fingerprints(mol_list, radius_values[j], fpSize_values[k], fp_types[l])
                        pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('pca', PCA(n_components=n_components[m])),
                            ('ridge', linear_model.Ridge(alpha = alpha_values[i]))
                        ])
                        scoring = {
                        'r2': 'r2',
                        'rmse': 'neg_root_mean_squared_error'
                        }
                        start_time = time.time()
                        result = cross_validate(pipeline, fp_data, prop_list, scoring=scoring, cv=foldmode)
                        end_time = time.time()
                        r2_mean = np.mean(result['test_r2'])
                        rmse_mean = -1 * np.mean(result['test_rmse'])
                        r2_std = np.std(result['test_r2'])
                        rmse_std = np.std(result['test_rmse'])
                        # Store results
                        result = {
                            'alpha': alpha_values[i],
                            'radius': radius_values[j],
                            'fpSize': fpSize_values[k],
                            'fp_type': fp_types[l],
                            'n_components': n_components[m],
                            'r2': r2_mean,
                            'r2_std': r2_std,
                            'rmse': rmse_mean,
                            'rmse_std': rmse_std,
                            'time': end_time-start_time
                            }
                        pbar.update(1)
                        performance.append(result)
                        #print("running ablation " + str(n) + " of " + str(len(alpha_values)*len(radius_values)*len(fpSize_values)*len(fp_types)*len(n_components)))
    pbar.close()
    return performance
