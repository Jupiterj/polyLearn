# This file is used to analyze and plot results from kfold_ablation_GTT.json
import json
import matplotlib.pyplot as plt


with open("kfold_ablation_GTT.json", "r") as f:
    data = json.load(f)

r2_values = [val['r2'] for val in data]
alpha_values = [val['alpha'] for val in data]
radius_values = [val['radius'] for val in data]
fpSize_values = [val['fpSize'] for val in data]
fp_types = [val['fp_type'] for val in data]
fp_values = [1 if val['fp_type'] == 'bit' else 2 if val['fp_type'] == 'count' else 3 for val in data]

# Selecting values for a specific radius
selected_alpha = 100
selected_radius = 1
selected_type = 3
filtered_r2_values = [r2 for r2, alpha, radius, type in zip(r2_values, alpha_values, radius_values, fp_values) if alpha == selected_alpha and radius == selected_radius and type == selected_type]
# filtered_radius_values = [radius for radius, alpha in zip(radius_values, alpha_values) if alpha == selected_alpha]
filtered_fpSize_values = [fpSize for fpSize, alpha, radius, type in zip(fpSize_values, alpha_values, radius_values, fp_values) if alpha == selected_alpha and radius == selected_radius and type == selected_type]
# filtered_fpType_values = [fpType for fpType, alpha, radius in zip(fp_values, alpha_values, radius_values) if alpha == selected_alpha and radius == selected_radius]



# Plotting r2 values for different alpha values
plt.figure(figsize=(10, 6))
plt.scatter(filtered_fpSize_values, filtered_r2_values, c='blue')
# plt.xscale('log')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
# Set font to Calibri and increase font sizes
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['lines.linewidth'] = 2.5
plt.tight_layout()
plt.show()