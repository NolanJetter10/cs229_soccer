import pandas as pd
import numpy as np

# Load data from the Excel file
excel_file = "Soccer kick Annotations.xlsx"
data = pd.read_excel(excel_file)  # Assuming no header row in the Excel file

# Convert the DataFrame to a NumPy array
data_array = data.to_numpy()
print(data_array.shape)

# Save the array as "Data_labels.npy"
np.save("Data_labels.npy", data_array)

print("Data successfully saved as 'Data_labels.npy'")