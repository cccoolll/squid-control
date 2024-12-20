import pandas as pd

# Load the CSV file
folder_path = "/media/reef/harddisk/20241112-hpa_2024-11-12_15-49-12.554140/0"
file_name = "coordinates.csv" 
df = pd.read_csv(f"{folder_path}/{file_name}")

# Function to calculate the cycle and allocate 'j'
def allocate_j(data):
    result = []
    for region, group in data.groupby('region'):
        i_values = group['i'].values
        # Determine the cycle length
        cycle_length = len(set(i_values))

        # Allocate 'j' based on the cycle
        j_values = []
        current_j = 0
        for idx, i in enumerate(i_values):
            if idx % cycle_length == 0 and idx != 0:
                current_j += 1
            j_values.append(current_j)

        # Append the results
        group['j'] = j_values
        result.append(group)

    return pd.concat(result)

# Apply the function to allocate 'j'
df_updated = allocate_j(df)

# Save the updated DataFrame to a new CSV file
output_file = "coordinates-processed.csv"
df_updated.to_csv(f"{folder_path}/{output_file}", index=False)

print(f"Updated data saved to {output_file}")