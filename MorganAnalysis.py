## This file serves as a hyperparameter ablation for the MorganFingerprint Generator built into RDKit. We first extract the relevant SMILES data and one desired property from a .json file. Then, we convert the SMILES data to a Mol object that is actionable by the RDKit API. We then conduct component ablation tests, assessing the R^2 and RMSE for a Regularized Ridge Regression model. The three hyperparameters that are given as options for component ablation are lambda/alpha for regularization algorithm, and radius and vector size for the Morgan algorithm. Additionally, values are simultaneously reported for three ways to represent fingerprints - in Bit form, count form, and heavy metal normalized count form. Time is recorded, as computational efficiency also becomes a critical aspect when these datasets get much larger. 

import json
from rdkit import Chem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time



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
                        prop =  data["polymer_data"][i]["property_summaries"][j]["properties"][k]["property_value_median"]
                        mol_list.append(mol)
                        prop_list.append(prop)
                        smiles_list.append(smiles)

    prop_list = np.array(prop_list)

    # Verify that all SMILES can be converted into mol using RDKit
    id_check = [mol is not None for mol in mol_list]
    if False in id_check:
        print("There is at least one polymer that could not be converted from SMILES")
    
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
    for i in range(len(alpha_values)):
        for j in range(len(radius_values)):
            for k in range(len(fpSize_values)):
                for l in range(len(fp_types)):
                    fp_data = generate_fingerprints(mol_list, radius_values[j], fpSize_values[k], fp_types[l])
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('ridge', linear_model.Ridge(alpha = alpha_values[i]))
                    ]) 
                    start_time = time.time()
                    prop_pred = cross_val_predict(pipeline, fp_data, prop_list, cv=foldmode)
                    end_time = time.time()
                    r2 = r2_score(prop_list, prop_pred)
                    rmse = root_mean_squared_error(prop_list, prop_pred)
                    # Store results
                    result = {
                        'alpha': alpha_values[i],
                        'radius': radius_values[j],
                        'fpSize': fpSize_values[k],
                        'fp_type': fp_types[l],
                        'r2': r2,
                        'rmse': rmse,
                        'time': end_time-start_time
                        }
                    performance.append(result)
    return performance

mol_list, prop_list = convert_smiles("poly_info_Radiation resistance.json", "Radiation resistance")
alpha_values = [0.1, 1, 10, 100, 1000, 10000]
radius_values = [1, 2, 3, 4, 5, 6]
fpSize_values = [1024, 2048, 4096]
fp_types = ["bit", "count", "normalized"]
foldmode = KFold(n_splits=5, shuffle=True, random_state = 2)
results = runablation(mol_list, prop_list, alpha_values, radius_values, fpSize_values, fp_types, foldmode)

with open("kfold_ablation.json", 'w') as f:
    json.dump(results, f)


