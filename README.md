## Accelerating Polymer Discovery *via* Glass-Transition Temperature Prediction

### Project Overview
Polymers are among the most difficult material classes to understand, owing to the complex interplay between their local chemical environments, global structures, and intramolecular interactions. This high degree of complexity makes it uniquely challenging to anticipate what a polymer's properties will be without experimental trial, which tends to be inefficient, expensive, and can lack repeatibility. One such property that is of particular interest in fields such as aerospace, electronics, and manufacturing is the glass transition temperature -- the temperature where a polymer rapidly transitions from being in a brittle, hard state to a soft, rubbery state. This project aims to accelerate the field of polymer discovery and design by training a machine learning model to predict glass transition temperature on a set of features that can be easily and conclusively extracted from a sample polymer. Namely, formula weight, density, melting temperature, and chemical structure. 

Our approach is as follows:  
**Data Extraction**: We manually scrape data via request queries to the PolyInfo database.  
**Data Selection and Processing**: Data is first cleaned (removing polymers that are blends and copolymers, or redundant). Then chemical names are converted into complex vectors via the Morgan Fingerprint (MF) algorithm. Prior to full model training, hyperparameter ablations are first conducted on the chemical structure datapoints to ensure that the MF itself is optimized. Principal Component Analysis (PCA) is tested to determine whether the MFs can be further reduced.  
**Linear Model Implementation and Ablation**: Ridge and Lasso regularized regression models are developed and alpha is optimized. Component ablation studies are conducted to determine whether specific properties have a larger impact on performance metrics compared to others.  
**Neural Network Extension**: As an extension to the main body of work, a simple neural network (multi-layer perceptron) is developed to compare the performance of non-linear models to linear models in this regression task.

### Getting Started  
***Data Preparation***  
  No data preparation is required to run our model. Clicking on main.py will run through the entire study. The model is built on sample_data/new_smiles.json; however, our code is built to be completely self-sufficient, so if you don't have that file, we will query for you and build it out. However, be warned that PolyInfo has a policy against manual scraping, so excessive use of the query functionalities will likely lead to a ban on the account used. If this happens, please contact the authors so we may delete our data. You are free to apply for your own account using your @stanford.edu account at [this website](https://polymer.nims.go.jp/).
  
***Package Installations***  
  Please run pip install -r requirements.txt prior to running main.py. RDKit is the speciality library used for cheminformatics.
  
***Running main.py***  
  The main script is completely self-sufficient. All outputs and monitoring can be done directly in terminal. Figures will pop up but will not save. Closing one figure will open the next one, in the order that they appear in the accompanying report. NOTE: TKinter GUI is an optional feature included in the script for data selection. The ability for this to run is dependant on your python version (3.9.7 works). 

Running main.py performs the following steps:
1. Load or collects data from PolyInfo
2. Cleans data and converts SMILES chemical structures to Morgan Fingerprint vectors via the RDKit library
3. Perform hyperparameter abalation studies with ridge regression on glass transition temperature for the Morgan Fingerprints algorithm to determine optimal parameters.
4. Using the optimized MF parameters, train Ridge and Lasso regression models.
5. Conduct component ablation studies on the various features included in the regression models.
6. Train a simple neural network (MLP) using the same feature set.
7. Generate the figures, table, and data discussed in the accompanying report.

***Repository Strucure***  
Please refer to the individual README.md in each folder for descriptions of each included file/script.

### Database  
PoLyInfo: https://polymer.nims.go.jp/ National Institute for Materials Science (NIMS), accessed 03-12-2026.

### Authors
Linda Lin, *ldlin@stanford.edu*  
Jaden Cramlet, *jadenjpt@stanford.edu*
