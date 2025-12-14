import pandas as pd
import numpy as np
import os

# File path configuration
input_path = "Dateset/OQMD_8110.csv"
output_path = "Dateset/OQMD_7176.csv"

# Read data
df = pd.read_csv(input_path)

# 1. Mark duplicate rows (for debugging)
duplicate_mask = df.duplicated(subset=['name', 'spacegroup'], keep=False)
duplicates_df = df[duplicate_mask].sort_values(by=['name', 'spacegroup'])
print(f"Duplicate data found: {len(duplicates_df)} rows (total {duplicate_mask.sum()} duplicate groups)")


# 2. Group and select rows with band_gap closest to the median
def select_median_bg(group):
    """
    Select the row with band_gap value closest to the median within the group
    Logic:
    1. Calculate the median of band_gap within the group
    2. Calculate the absolute difference between each band_gap value and the median
    3. Select the row with the smallest difference (if multiple, take the first one)
    """
    if len(group) == 1:
        return group

    # Calculate the median of band_gap
    median_bg = group['band_gap'].median()

    # Calculate absolute difference from the median
    abs_diff = (group['band_gap'] - median_bg).abs()

    # Find the row closest to the median (if multiple, take the first one)
    min_diff_idx = abs_diff.idxmin()

    return group.loc[[min_diff_idx]]


# Group by key columns and apply selection logic
cleaned_df = df.groupby(['name', 'spacegroup'], group_keys=False).apply(select_median_bg).reset_index(drop=True)

# 3. Save results
cleaned_df.to_csv(output_path, index=False)
print(f"Data cleaning completed! Original data: {len(df)} rows -> After cleaning: {len(cleaned_df)} rows")
print(f"Saved to: {output_path}")

# 4. Verification example (optional)
if not cleaned_df.empty:
    sample_check = cleaned_df.sample(min(3, len(cleaned_df)), random_state=42)
    print("\nSample verification (random 3 groups):")
    print(sample_check[['name', 'spacegroup', 'band_gap']].merge(
        df[df.duplicated(subset=['name', 'spacegroup'], keep=False)],
        on=['name', 'spacegroup'],
        how='left',
        suffixes=('_cleaned', '_original')
    ))
