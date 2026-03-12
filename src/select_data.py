############
# This script parses the .json file created by fetchPolyInfo.py and ouputs the relevant features in a .csv file
############

import json
import pandas as pd
import numpy as np
import tkinter as tk

## This function looks through all the available parameters in a file and opens a GUI that allows the user to select what features they want to base their model on
def select_params(file_name):
    with open(file_name, "r") as file:
        data = json.load(file)

    # Find all available parameters across all polymers in the .json file
    prop_list = [] # list all of the available parameters
    for i in range(len(data["polymer_data"])):
        if "smiles" in data["polymer_data"][i]:
            prop_list.append("SMILES")
        if "formula_weight" in data["polymer_data"][i]:
            prop_list.append("formula_weight")
        for j in range(len(data["polymer_data"][i]["property_summaries"])):
            for k in range(len(data["polymer_data"][i]["property_summaries"][j]["properties"])):
                prop_list.append(data["polymer_data"][i]["property_summaries"][j]["properties"][k]["property_name"])
    # Count and index all parameters
    df = pd.DataFrame({"parameter" : prop_list})
    prop_count = df["parameter"].value_counts()


    ## GUI select what variables we want to include in our final csv and analysis
    fs = tk.Tk()
    fs.title("Label Picker")
    fs.geometry("300x400")
    width = fs.winfo_width()
    height = fs.winfo_height()
    screen_width = fs.winfo_screenwidth()
    screen_height = fs.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    fs.geometry(f"+{x}+{y}")
    label = tk.Label(fs, text="Select the desired label property:", wraplength=300)
    label.pack()
    label = tk.Label(fs, text="For our project, we selected glass transition temperature.", wraplength=300)
    label.pack()
    # Frame to hold list and scroll
    list_frame = tk.Frame(fs)
    list_frame.pack(fill=tk.BOTH, expand=True)
    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    paramlist = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE)
    for line in range(len(prop_count.index)):
        if prop_count.index[line] == "formula_weight":
            paramlist.insert(tk.END, "Formula weight (" + str(prop_count.values[line]) + ")")
        else:
            paramlist.insert(tk.END, prop_count.index[line] + " (" + str(prop_count.values[line]) + ")")
    paramlist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=paramlist.yview)
    # Define function for what happens when we click confirm button
    selected_label = []
    def confirm_selection():
        selected_index = paramlist.curselection()
        selected_label.append([prop_count.index[i] for i in selected_index])
        fs.destroy()
    # Confirmation button
    button = tk.Button(fs, text="Confirm", command=confirm_selection)
    button.pack(side=tk.BOTTOM, fill=tk.X)
    fs.lift()
    fs.update()
    fs.mainloop()

    fs = tk.Tk()
    fs.title("Feature Picker")
    fs.geometry("300x400")
    fs.geometry(f"+{x}+{y}")
    label = tk.Label(fs, text="Select the desired feature properties:", wraplength=300)
    label.pack()
    label = tk.Label(fs, text="For our project, we selected density, SMILES, formula weight, and melting temperature.",wraplength=300)
    label.pack()
    # Frame to hold list and scroll
    list_frame = tk.Frame(fs)
    list_frame.pack(fill=tk.BOTH, expand=True)
    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    paramlist = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, selectmode=tk.MULTIPLE)
    for line in range(len(prop_count.index)):
        if prop_count.index[line] == "formula_weight":
            paramlist.insert(tk.END, "Formula weight (" + str(prop_count.values[line]) + ")")
        else:
            paramlist.insert(tk.END, prop_count.index[line] + " (" + str(prop_count.values[line]) + ")")
    paramlist.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=paramlist.yview)
    # Define function for what happens when we click confirm button
    selected_items = []
    def confirm_selection():
        selected_index = paramlist.curselection()
        selected_items.append([prop_count.index[i] for i in selected_index])
        fs.destroy()
    # Confirmation button
    button = tk.Button(fs, text="Confirm", command=confirm_selection)
    button.pack(side=tk.BOTTOM, fill=tk.X)

    fs.lift()
    fs.update()
    fs.mainloop()

    # return selected items
    return selected_label[0], selected_items[0], data


