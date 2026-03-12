fetch_data.py - this script contains functions that are used to extract data from the PolyInfo database and save to .json

select_data.py - this script contains the GUI that can be used to manually select parameters from a .json file.

morgan_ridge_analysis.py - this script contains functions that convert SMILES structures to RDKit-processable MOL functions, then conducts hyperparamater ablation to determine optimal representations of the dataset polymers in vector form.

plot_morganRidge.py - this script contains all plots and data collection used for the initial Morgan Fingerprint optimization.

polylearn_linear.py - this script contains functions that implement, train, and evaluate linear regression models for all and components of the feature space

polylearn_nn.py - this script contains functions that implement, train, and evaluate a multi-layer perceptron neural network model for all features in the provided parameter space.