import pandas as pd

# Define file paths
input_path = "Dateset/OQMD_12767.csv"
output_path = "Dateset/OQMD_8110.csv"

# Read data
try:
    df = pd.read_csv(input_path)
    print(f"Data read successfully, original number of rows: {len(df)}")
except FileNotFoundError:
    print(f"Error: File does not exist - {input_path}")
    exit()

# Define deletion conditions (based on search results [1,4,6](@ref))
condition_delta = df['delta_e'] > 0        # Formation energy greater than 0
condition_stability = df['stability'] > 0.1  # Hull energy greater than 0.1
combined_condition = condition_delta | condition_stability  # Delete if either condition is met

# Delete rows meeting the conditions (two equivalent methods)
# Method 1: Boolean indexing filtering (recommended)
filtered_df = df[~combined_condition]

# Save results
filtered_df.to_csv(output_path, index=False)
print(f"Deleted {len(df) - len(filtered_df)} rows of data that do not meet the conditions")
print(f"Number of rows after processing: {len(filtered_df)}")
print(f"Cleaned dataset has been saved to: {output_path}")

# Verify deletion effect
print("\nDeletion condition statistics:")
print(f"Rows with formation energy > 0: {condition_delta.sum()}")
print(f"Rows with hull energy > 0.1: {condition_stability.sum()}")
print(f"Total deleted rows: {len(df) - len(filtered_df)}")

# Sample display comparing data before and after processing
if not filtered_df.empty:
    sample_size = min(3, len(filtered_df))
    sample = filtered_df.sample(sample_size, random_state=42)
    print("\nProcessed data example:")
    print(sample[['delta_e', 'stability']])
