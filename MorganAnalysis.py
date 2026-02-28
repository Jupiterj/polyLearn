## This file serves as a hyperparameter ablation for the MorganFingerprint Generator built into RDKit. We first extract the relevant SMILES data and one desired property from a .json file. Then, we convert the SMILES data to a Mol object that is actionable by the RDKit API. We then conduct component ablation tests, assessing the LOOCV R^2 and RMSE for a Regularized Ridge Regression model. The three hyperparameters that are given as options for component ablation are lambda/alpha for regularization algorithm, and radius and vector size for the Morgan algorithm. Additionally, values are simultaneously reported for three ways to represent fingerprints - in Bit form, count form, and heavy metal normalized count form. Time is recorded, as computational efficiency also becomes a critical aspect when these datasets get much larger. 

import json
from rdkit import Chem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from sklearn import linear_model
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, LeaveOneOut
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

def ridgeablation(mol_list, prop_list, alpha, radius, fpSize):
    # Generate Fingerprints for each .mol polymer
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius = radius, fpSize = fpSize) # Chirality is default to false, bond types is default to true
    loo = LeaveOneOut() # we elect to do LOO cross validation
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', linear_model.Ridge(alpha = alpha))
    ]) 

    # Ablation 1 - Use Bit Fingerprints
    bit_list = np.zeros((len(mol_list),fpSize))
    for i in range(len(mol_list)):
        bit_list[i] = fpgen.GetFingerprintAsNumPy(mol_list[i])

    start = time.time()
    prop_pred = cross_val_predict(pipeline, bit_list, prop_list, cv=loo)
    end = time.time()
    r2 = r2_score(prop_list, prop_pred)
    rmse = root_mean_squared_error(prop_list, prop_pred)
    print(f"Bit Fingerprint Ablation \nLOOCV Overall R²: {r2:.4f}")
    print(f"LOOCV Overall RMSE: {rmse:.4f}")
    print(f"This calculation took {end-start} seconds")

    # Ablation 2 - Use Count Fingerprints
    count_list = np.zeros((len(mol_list),fpSize))
    for i in range(len(mol_list)):
        count_list[i] = fpgen.GetCountFingerprintAsNumPy(mol_list[i])

    start = time.time()
    prop_pred = cross_val_predict(pipeline, count_list, prop_list, cv=loo)
    end = time.time()
    r2 = r2_score(prop_list, prop_pred)
    rmse = root_mean_squared_error(prop_list, prop_pred)
    print(f"\nCount Fingerprint Ablation \nLOOCV Overall R²: {r2:.4f}")
    print(f"LOOCV Overall RMSE: {rmse:.4f}")
    print(f"This calculation took {end-start} seconds")

    # Ablation 3 - Use Density Normalized Count Fingerprints
    size_normalization = np.array([mol.GetNumHeavyAtoms() for mol in mol_list])
    normcount_list = np.zeros((len(mol_list),fpSize))
    for i in range(len(mol_list)):
        normcount_list[i] = count_list[i]/size_normalization[i]
    
    start = time.time()
    prop_pred = cross_val_predict(pipeline, normcount_list, prop_list, cv=loo)
    end = time.time()
    r2 = r2_score(prop_list, prop_pred)
    rmse = root_mean_squared_error(prop_list, prop_pred)
    print(f"\nHeavy-Atom-Normalized Fingerprint Ablation \nLOOCV Overall R²: {r2:.4f}")
    print(f"LOOCV Overall RMSE: {rmse:.4f}")
    print(f"This calculation took {end-start} seconds")

