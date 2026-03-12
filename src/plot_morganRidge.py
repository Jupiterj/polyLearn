# This file is used to analyze and plot results used in MF section of the Results portion of the paper
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import src.morgan_ridge_analysis as mf


#######
#  FIGURE 1(a)
#######

def fig1a(data):
    val01 = []
    val1 = []
    val10 = []
    val100 = []
    val1000 = []
    val10000 = []
    for i in range(len(data)):
        if data[i]["alpha"] == 0.1:
            val01.append(data[i]["r2"])
        elif data[i]["alpha"] == 1:
            val1.append(data[i]["r2"])
        elif data[i]["alpha"] == 10:
            val10.append(data[i]["r2"])
        elif data[i]["alpha"] == 100:
            val100.append(data[i]["r2"])
        elif data[i]["alpha"] == 1000:
            val1000.append(data[i]["r2"])
        else:
            val10000.append(data[i]["r2"])

    meanlist = [np.mean(val01), np.mean(val1), np.mean(val10), np.mean(val100), np.mean(val1000), np.mean(val10000)]
    stdlist = [np.std(val01), np.std(val1), np.std(val10), np.std(val100), np.std(val1000), np.std(val10000)]
    alphalist = [0.1, 1, 10, 100, 1000, 10000]
    x_pos = np.arange(len(alphalist))
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['lines.linewidth'] = 3

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, meanlist, yerr=stdlist, capsize=10, color=colors, linewidth=3)
    plt.xticks(x_pos, [str(a) for a in alphalist], fontsize=25)
    plt.yticks(fontsize=25)
    # Set font to Calibri and increase font sizes
    plt.xlabel("Alpha", fontsize = 25)
    plt.ylabel("$R^2$ Score", fontsize = 25)
    plt.tight_layout()
    plt.show()

#########
#  FIGURE 1(b)
#########

def fig1b(data):
    val1 = []
    val2 = []
    val3 = []
    for i in range(len(data)):
        if data[i]["fp_type"] == "bit":
            val1.append(data[i]["r2"])
        elif data[i]["fp_type"] == "count":
            val2.append(data[i]["r2"])
        else:
            val3.append(data[i]["r2"])

    meanlist = [np.mean(val1), np.mean(val2), np.mean(val3)]
    stdlist = [np.std(val1), np.std(val2), np.std(val3)]
    fp_types = ["Bit", "Count", "Weighted"]
    x_pos = np.arange(len(fp_types))
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['font.size'] = 19
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['lines.linewidth'] = 3

    colors = ["#17BECF", "#E377C2", "#BCBD22"]
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, meanlist, yerr=stdlist, capsize=10, color=colors, linewidth=3)
    plt.xticks(x_pos, [str(a) for a in fp_types], fontsize=25)
    plt.yticks(fontsize=25)
    # Set font to Calibri and increase font sizes
    plt.xlabel("Vector Form", fontsize = 25)
    plt.ylabel("$R^2$ Score", fontsize = 25)
    plt.tight_layout()
    plt.show()

def table1(data):
    file_name = "table1_results.json"
    with open(file_name, "w") as file:
        json.dump(data, file, indent=2)

def fig2a(data):
    min_idx = np.argmin([entry["r2"] for entry in data])
    max_idx = np.argmax([entry["r2"] for entry in data])
    print(data[min_idx])
    print(data[max_idx])

    file_name = "pca_ablation_results.json"
    with open(file_name, "w") as file:
        json.dump(data, file, indent=2)
    val2 = []
    val10 = []
    val50 = []
    val100 = []

    for i in range(len(data)):
        if data[i]["n_components"] == 2:
            val2.append(data[i]["r2"])
        elif data[i]["n_components"] == 10:
            val10.append(data[i]["r2"])
        elif data[i]["n_components"] == 50:
            val50.append(data[i]["r2"])
        elif data[i]["n_components"] == 100:
            val100.append(data[i]["r2"])

    meanlist = [np.mean(val2), np.mean(val10), np.mean(val50), np.mean(val100)]
    stdlist = [np.std(val2), np.std(val10), np.std(val50), np.std(val100)]
    n_components = [2, 10, 50, 100]
    x_pos = np.arange(len(n_components))
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['font.size'] = 19
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['lines.linewidth'] = 3

    colors = ['#2A9D8F', '#E9C46A','#F4A261','#264653']
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, meanlist, yerr=stdlist, capsize=10, color=colors, linewidth=3)
    plt.xticks(x_pos, [str(a) for a in n_components], fontsize=25)
    plt.yticks(fontsize=25)
    # Set font to Calibri and increase font sizes
    plt.xlabel("# Components", fontsize = 25)
    plt.ylabel("$R^2$ Score", fontsize = 25)
    plt.tight_layout()
    plt.show()

def fig2b(mol_list, prop_list):
    # Bin colours
    low_thres = np.percentile(prop_list, 33)
    high_thres = np.percentile(prop_list, 66)
    colors = []
    for prop in prop_list:
        if prop < low_thres:
            colors.append("#001DDA")  
        elif prop < high_thres:
            colors.append("#0D7546")  
        else:
            colors.append("#CB4A1B")  

    pca = PCA()
    scale = StandardScaler()
    fdata = mf.generate_fingerprints(mol_list, radius = 1, fpSize=4096, fp_type="normalized")
    fdata_scaled = scale.fit_transform(fdata)
    fdata_pca = pca.fit_transform(fdata_scaled)

    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['lines.linewidth'] = 3

    plt.figure(figsize=(10, 6))
    plt.scatter(fdata_pca[:,0], fdata_pca[:,1], 50,c=colors)
    plt.xlabel("PC1", fontsize = 25)
    plt.ylabel("PC2", fontsize = 25)
    # plt.xlim((-10,10))
    # plt.ylim((-10,10))
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#001DDA', markersize=10, label='Low'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0D7546', markersize=10, label='Medium'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#CB4A1B', markersize=10, label='High')],
                    frameon=False)
    plt.tight_layout()
    plt.show()
