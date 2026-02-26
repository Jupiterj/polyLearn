# This is a machine learning model for predicting polymer materials properties
import pandas as pd
import numpy as np
import json
from pandas import json_normalize


def get_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        
    # Import properties data into a pandas dataframe, keeping the polymer name as metadata
    # Ignores polymers without name (polymer blends)
    df= json_normalize(data,
                        ['polymer_data','property_summaries','properties'],
                        [['polymer_data','polymer_name']],
                        errors='ignore',
                        ).dropna(subset=['polymer_data.polymer_name'])
    df = df[['polymer_data.polymer_name','property_name','property_value_median']]

    # Averages values from duplicates
    df = df.groupby(['polymer_data.polymer_name','property_name'],sort=False).mean()
    df = df.reset_index()

    # Converts each property into its own column using the median as the value
    df = df.pivot(index='polymer_data.polymer_name', columns='property_name', values='property_value_median')

    # Reset the index to turn 'polymer_data.polymer_name' back into a column
    df = df.reset_index()

    # Returns dataframe
    return df


# Main body
data_filename = 'poly_info_Radiation resistance.json'
df = get_data(data_filename)

# Determines the number of datapoints for each property
properties = np.array(df.columns.tolist())[1:]
d_points = np.count_nonzero(~np.isnan(df.to_numpy()[:,1:].astype(float)),axis = 0)

top_5_properties_arg = np.flip(np.argsort(d_points))[:5]

print('Top 5 properties:',properties[top_5_properties_arg])
print('Counts:',d_points[top_5_properties_arg])

# Converts to numpy array
array = df.to_numpy()[:,1:].astype(float)
properties = df.columns.to_numpy()[1:]
polymer_names = df['polymer_data.polymer_name'].to_numpy()

# Check dimensions of all outputs
print(array.shape)
print(properties.shape)
print(polymer_names.shape)