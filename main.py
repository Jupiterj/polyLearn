## This is the main file for the project. It will be used to run all models and generate the results. It will also be used to generate the plots.
from src.fetch_data import fetch_poly_list_info, fetch_poly_smiles_info
from pathlib import Path
from src.select_data import select_params
from src.morgan_ridge_analysis import convert_smiles, runablation, run_combined_PCAablation
from sklearn.model_selection import KFold
from src.plot_morganRidge import fig1a, fig1b, table1, fig2a, fig2b

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

def main():
    ######
    # PART 1: Data Collection and Selection
    ######
    # The data is already collected and saved in the "sample_data/new_smiles.json" file. Please note that we are technically not allowed to scrape data, so the functions in the else statement may be unstable and not guaranteed to permanently work due to potential bans. If you would like to run the data collection yourself (i.e. on other properties), feel free to make your own account and edit the info in getlogin() in fetch_data.py.
    
    fetch_property_names = ["Glass transition temperature", "Density", "Melting point"]
    existing_file = SAMPLE_DATA_DIR / "new_smiles.json"

    if existing_file.exists():
        print("Data file already exists. Skipping data collection.")
    else:
        print("Proceeding with data collection from PolyInfo to new_smiles.json")
        fetch_poly_list_info(fetch_property_names, limit=50)
        fetch_poly_smiles_info(str(PROJECT_ROOT / "sample_data" / f"poly_info_{fetch_property_names[0]}.json"))
    
    try:
        selected_label, selected_properties, _ = select_params(existing_file)
    except Exception as e:
        selected_label = fetch_property_names[0]
        selected_properties = fetch_property_names[1:-1]
        selected_properties.append("SMILES")
        selected_properties.append("formula_weight")
        print("Your version of Python may not support the GUI, we default the selected properties to build our model off of as the initially fetched properties.")

    print("You have elected to build your model off of the label: " + ",".join(selected_label) + " and the features: " + ", ".join(selected_properties) + ".")

    ######
    # PART 2: Ridge Regression on Morgan Fingerprints
    ######
    # The following functions are used in the first part of our report discussion, where we explore the hyperparameters involved in the MorganFingerprint algorithm contained within RDKit and how they can be optimized for a ridge regression model.


    mol_list, labels = convert_smiles(existing_file, selected_label[0]) 
    # This function converts SMILES representation into .MOL format parsable by RDKIT
    kfold = KFold(n_splits=5, shuffle=True, random_state = 2)
    # Elected for 5-fold cross validation, with shuffling and a fixed random state for reproducibility.

    ## Fig 1(a) - Full Hyperparameter Sweep
    raw_ablation = runablation(mol_list, labels, alpha_values=[0.1, 1, 10, 100, 1000, 10000], radius_values=[1, 2, 3, 4, 5, 6],fpSize_values= [1024, 2048, 4096],fp_types=["bit", "count", "normalized"], foldmode=kfold)
    fig1a(raw_ablation)

    ## Fig 1(b) - Fingerprint Type Comparison
    raw_ablation_fp = runablation(mol_list, labels, alpha_values=[100], radius_values=[1, 2, 3, 4, 5, 6],fpSize_values= [1024, 2048, 4096],fp_types=["bit", "count", "normalized"], foldmode=kfold)
    fig1b(raw_ablation_fp)

    ## Generate data for Table 1
    table_data = runablation(mol_list, labels, alpha_values=[100], radius_values=[1, 2, 4, 6],fpSize_values= [1024, 2048, 4096],fp_types=["normalized"], foldmode=kfold)
    table1(table_data)

    ## Fig 2(a) - Full PCA Hyperparameter Sweep
    pca_ablation = run_combined_PCAablation(mol_list, labels, alpha_values=[0.1, 1, 10, 100, 1000, 10000], radius_values=[1, 2, 4, 6],fpSize_values= [1024, 2048, 4096],fp_types=["bit", "count", "normalized"], n_components=[2, 10, 50, 100], foldmode=kfold)
    fig2a(pca_ablation)

    ## Fig 2(b) - PC1 vs PC2 Visualization, with labels in three categories based on the percentiles of the property value distribution.
    fig2b(mol_list, labels)

    ######
    # PART 3: Linear Regression on all features
    ######
    # The following functions now integrate all features and runs a linear regression model to determine the most important features for predicting the property of interest.

if __name__ == "__main__":
    main()