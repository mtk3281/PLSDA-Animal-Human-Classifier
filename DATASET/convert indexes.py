import pandas as pd
import os

# Path to the folder containing CSV files
folder_path = 'temp'

column_list = pd.read_csv('selected features.csv')

collist= column_list.columns.values
print(collist)

# Iterate through each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # Read the CSV file
        csv_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(csv_path)
        
        # Keep only the columns present in the column list
        # columns_to_keep = [col for col in df.columns if "WAVELENGTH"+str(col) in collist or col in ["SAMPLE","CLASS","NAME"]]
        columns_to_keep = [col for col in df.columns if col in collist]
     
        df = df[columns_to_keep]
        
        # Write the modified data back to the CSV file
        df.to_csv(csv_path, index=False)
