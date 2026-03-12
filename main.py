## This is the main file for the project. It will be used to run all models and generate the results. It will also be used to generate the plots.
from src.fetch_data import fetch_poly_list_info, fetch_poly_smiles_info
from pathlib import Path
try:
    from src.select_data import select_params
except ImportError:
    select_params = None
from src.morgan_ridge_analysis import convert_smiles, runablation, run_combined_PCAablation
from src.polylearn_linear import compare_components
from sklearn.model_selection import KFold
from src.plot_morganRidge import fig1a, fig1b, table1, fig2a, fig2b
from src.polylearn_nn import train_nn,fig4
import json

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

def run_data_collection_selection(fetch_property_names):
    ""
    ############################
    # PART 1: Data Collection and Selection
    ############################
    # The data is already collected and saved in the "sample_data/new_smiles.json" file. Please note that we are technically not allowed to scrape data, so the functions in the else statement may be unstable and not guaranteed to permanently work due to potential bans. If you would like to run the data collection yourself (i.e. on other properties), feel free to make your own account and edit the info in getlogin() in fetch_data.py. NOTE that you will likely get banned after one go, so be warned!
    
    # Input:
    # fetch_property_names: a list of property names to query from PolyInfo. The first property in the list will be used as the label for the model, and the rest will be used as features, by default.
    # Output:
    # existing_file: the path to the existing file containing the data, if it exists. If it does not exist, the data will be collected and saved to this file.
    # selected_label: the label selected for the model, either through the GUI or defaulted to the first property in the list.
    # selected_properties: the features selected for the model, either through the GUI or defaulted to the rest of the properties in the list + SMILES + formula weight.

    existing_file = SAMPLE_DATA_DIR / "new_smiles.json"

    print("\n================== DATA COLLECTION AND SELECTION ==================")
    if existing_file.exists():
        print("Data file already exists. Skipping data collection.")
    else:
        print("Proceeding with data collection from PolyInfo to new_smiles.json")
        fetch_poly_list_info(fetch_property_names, limit=50)
        fetch_poly_smiles_info(str(PROJECT_ROOT / "sample_data" / f"poly_info_{fetch_property_names[0]}.json"))
    
    # Try to run the GUI selection if TKinter is available, otherwise default label to the first specified property and default features to otherproperties + SMILES + formula weight.
    try:
        selected_label, selected_properties, _ = select_params(existing_file)
    except Exception as e:
        selected_label = [fetch_property_names[0]]
        selected_properties = fetch_property_names[1:]
        selected_properties.append("SMILES")
        selected_properties.append("formula_weight")

    print("You have elected to build your model off of the label: " + ",".join(selected_label) + " and the features: " + ", ".join(selected_properties) + ".")

    return existing_file, selected_label, selected_properties

def ridge_fingerprint_analysis(existing_file, selected_label):
    ############################
    # PART 2: Ridge Regression on Morgan Fingerprints
    ############################
    # The following functions are used in the first part of our report discussion, where we explore the hyperparameters involved in the MorganFingerprint algorithm contained within RDKit and how they can be optimized for a ridge regression model.

    # Input:
    # existing_file: the path to the existing file containing the extracted data.
    # selected_label: the label selected for the model
    # Output:
    # ablation_path: the path to the file containing the results of the ablation study on the Morgan fingerprints, which will be used for the full linear model analysis.
    # Plots and tables used in the report discussion.

    print("\n================== RIDGE REGRESSION ON MORGAN FINGERPRINTS ==================")

    mol_list, labels = convert_smiles(existing_file, selected_label[0]) 
    # This function converts SMILES representation into .MOL format parsable by RDKIT
    kfold = KFold(n_splits=5, shuffle=True, random_state = 2)
    # Elected for 5-fold cross validation, with shuffling and a fixed random state for reproducibility.
    print("Outputting Fig 1(a)")
    ## Fig 1(a) - Full Hyperparameter Sweep
    raw_ablation = runablation(mol_list, labels, alpha_values=[0.1, 1, 10, 100, 1000, 10000], radius_values=[1, 2, 3, 4, 5, 6],fpSize_values= [1024, 2048, 4096],fp_types=["bit", "count", "normalized"], foldmode=kfold)
    fig1a(raw_ablation)
    # Save the ablation data for reference in the full model
    ablation_path = SAMPLE_DATA_DIR / "fp_raw_ablation.json"
    with open(ablation_path, 'w') as f:
        json.dump(raw_ablation, f)

    print("Outputting Fig 1(b)")
    ## Fig 1(b) - Fingerprint Type Comparison
    raw_ablation_fp = runablation(mol_list, labels, alpha_values=[100], radius_values=[1, 2, 3, 4, 5, 6],fpSize_values= [1024, 2048, 4096],fp_types=["bit", "count", "normalized"], foldmode=kfold)
    fig1b(raw_ablation_fp)

    print("Generating Data for Table 1")
    ## Generate data for Table 1
    table_data = runablation(mol_list, labels, alpha_values=[100], radius_values=[1, 2, 4, 6],fpSize_values= [1024, 2048, 4096],fp_types=["normalized"], foldmode=kfold)
    table1(table_data)

    ## Fig 2(a) - Full PCA Hyperparameter Sweep
    print("Outputting Fig 2(a)")
    pca_ablation = run_combined_PCAablation(mol_list, labels, alpha_values=[0.1, 1, 10, 100, 1000, 10000], radius_values=[1, 2, 4, 6],fpSize_values= [1024, 2048, 4096],fp_types=["bit", "count", "normalized"], n_components=[2, 10, 50, 100], foldmode=kfold)
    fig2a(pca_ablation)

    ## Fig 2(b) - PC1 vs PC2 Visualization, with labels in three categories based on the percentiles of the property value distribution.
    print("Outputting Fig 2(b)")
    fig2b(mol_list, labels)

    return ablation_path

def full_linear_model_analysis(existing_file, ablation_path, selected_label, selected_properties):
    ############################
    # PART 3: Linear Regression on all features
    ############################
    # The following functions now integrate all features and runs a linear regression model to determine the most important features for predicting the property of interest.
    
    # Input:
    # existing_file: the path to the existing file containing the extracted data.
    # ablation_path: the path to the file containing the results of the ablation study on the Morgan fingerprints
    # selected_label: the label selected for the model
    # selected_properties: the features selected for the model
    # Output:
    # Plots used in the report discussion.
    print("\n================== COMPONENT ABLATION ON LINEAR MODELS ==================")
    comp_results = compare_components(existing_file, ablation_path, selected_label, selected_properties)
    # This function prints the results of Lasso and Ridge regression models with different combinations of features, with a newly optimized alpha value for each model. 

    ####### TODO: @Jaden -- add your functions for plotting here.

def nn_model_analysis(existing_file, ablation_path, selected_label, selected_properties):
    ############################
    # PART 4: Simple Neural Network model on all features
    ############################
    # The following function runs a simple MLP on all features, with the same inputs as for the linear regression model. The purpose is to determine if a non-linear model could potentially perform better with the same features.

    # Input:
    # existing_file: the path to the existing file containing the extracted data.
    # ablation_path: the path to the file containing the results of the ablation study on the Morgan fingerprints
    # selected_label: the label selected for the model
    # selected_properties: the features selected for the model
    # Output:
    # Plots used in the report discussion.
    
    print("\n================== NEURAL NETWORK ON FULL COMPONENTS ==================")
    nn_results = train_nn(existing_file, ablation_path, selected_label, selected_properties,epochs = 400,weight_decay = 1,lr=1e-3)

    fig4(nn_results)


def main():
    # Set desired query properties
    fetch_property_names = ["Glass transition temperature", "Density", "Melting temperature"]
    # Collect and select data
    existing_file, selected_label, selected_properties = run_data_collection_selection(fetch_property_names)
    # Run initial ablations on only the morgan fingerprints to determine hyperparameters
    ablation_path = ridge_fingerprint_analysis(existing_file, selected_label)
    # Conduct component ablation for all components of the linear model
    full_linear_model_analysis(existing_file, ablation_path, selected_label, selected_properties)
    # Compare with a simple neural network model on the same features
    nn_model_analysis(existing_file, ablation_path, selected_label, selected_properties)


if __name__ == "__main__":
    main()