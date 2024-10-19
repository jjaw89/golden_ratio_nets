import os
import pandas as pd

# Set the directory containing the output files
# adjust this path if necessary
output_dir = 'c:/Users/Jaspar/Dropbox/PythonProjects/DEM/bench/outputs'

# Create an empty list to hold the data
data = []

# Loop through each file in the directory
for filename in os.listdir(output_dir):
    if filename.endswith('_stdout.txt'):
        # Extract a, b, num_digits, and algorithm from the filename
        base_name = filename.replace('_stdout.txt', '')
        parts = base_name.split('_')
        a = parts[3]
        b = parts[5]
        num_digits = parts[8]
        # sdiscr_dem = parts[10]

        # Read stdout, stderr, and time from the files
        with open(os.path.join(output_dir, base_name + '_stdout.txt'), 'r') as f:
            stdout = f.read().strip()
        with open(os.path.join(output_dir, base_name + '_time.txt'), 'r') as f:
            time_val = float(f.read().strip())

        # Append the data to our data list
        data.append([a, b, num_digits, stdout, time_val])

# Convert the data list to a pandas DataFrame
df = pd.DataFrame(
    data, columns=['a', 'b', 'num_digits', 'stdout',  'time'])

# Optional: save the dataframe to a CSV file
# df.to_csv('benchmark_results.csv', index=False)

# Print the dataframe
print(df)
df.to_csv('C:/Users/Jaspar/Dropbox/PythonProjects/Square root of 2/results.txt',
          sep=',', index=False)
