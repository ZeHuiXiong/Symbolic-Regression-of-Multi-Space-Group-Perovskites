import pandas as pd
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import warnings
import os
from tqdm import tqdm
import traceback
from io import StringIO


def parse_cif_string(cif_string):
    """Parse crystal structure from CIF string"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cif_io = StringIO(cif_string)
            parser = CifParser(cif_io)
            structures = parser.parse_structures(primitive=True)
            return structures[0] if structures else None
    except Exception as e:
        print(f"Failed to parse CIF string: {str(e)}")
        return None


def calculate_xrd_peaks(structure, wavelength='CuKa', min_angle=5, max_angle=90):
    """Calculate XRD characteristic peaks and return information for the top 6 strongest peaks"""
    if structure is None:
        return None

    try:
        calculator = XRDCalculator(wavelength=wavelength)
        pattern = calculator.get_pattern(structure, two_theta_range=(min_angle, max_angle))

        # If no peaks, return empty result
        if len(pattern.x) == 0:
            return {
                'num_peaks': 0,
                'peak1_intensity': np.nan, 'peak1_position': np.nan,
                'peak2_intensity': np.nan, 'peak2_position': np.nan,
                'peak3_intensity': np.nan, 'peak3_position': np.nan,
                'peak4_intensity': np.nan, 'peak4_position': np.nan,
                'peak5_intensity': np.nan, 'peak5_position': np.nan,
                'peak6_intensity': np.nan, 'peak6_position': np.nan
            }

        # Sort peaks by intensity in descending order
        sorted_indices = np.argsort(pattern.y)[::-1]
        sorted_angles = pattern.x[sorted_indices]
        sorted_intensities = pattern.y[sorted_indices]

        # Extract information for the top 6 peaks
        result = {'num_peaks': len(pattern.x)}

        # Fill information for the first 6 peaks
        for i in range(1, 7):
            if i <= len(sorted_intensities):
                result[f'peak{i}_intensity'] = sorted_intensities[i - 1]
                result[f'peak{i}_position'] = sorted_angles[i - 1]
            else:
                result[f'peak{i}_intensity'] = np.nan
                result[f'peak{i}_position'] = np.nan

        return result

    except Exception as e:
        print(f"Failed to calculate XRD: {str(e)}")
        return None


def process_cif_file(csv_path, output_dir=None):
    """
    Process all CIF strings in a CSV file and calculate XRD characteristic peak information
    Args:
        csv_path: Input CSV file path
        output_dir: Output directory (default: same as input file directory)
    """
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV file: {csv_path}")
        print(f"Dataset contains {len(df)} rows")
    except Exception as e:
        print(f"Failed to read CSV file: {str(e)}")
        return None

    # Check if 'cif' column exists
    if 'cif' not in df.columns:
        print("Error: CSV file does not have a 'cif' column")
        return None

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize result columns
    new_columns = [
        'num_peaks',
        'peak1_intensity', 'peak1_position',
        'peak2_intensity', 'peak2_position',
        'peak3_intensity', 'peak3_position',
        'peak4_intensity', 'peak4_position',
        'peak5_intensity', 'peak5_position',
        'peak6_intensity', 'peak6_position'
    ]

    for col in new_columns:
        df[col] = np.nan

    # Process each row
    print(f"\nStarting to process {len(df)} CIF strings...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing progress"):
        cif_content = row['cif']

        try:
            # Check if CIF content is empty
            if pd.isna(cif_content) or not cif_content:
                continue

            # Parse CIF string
            structure = parse_cif_string(str(cif_content))
            if structure is None:
                continue

            # Calculate XRD characteristic peaks
            peak_info = calculate_xrd_peaks(structure)

            if peak_info is None:
                continue

            # Update results
            for col in new_columns:
                if col in peak_info:
                    df.at[idx, col] = peak_info[col]

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            traceback.print_exc()

    # Generate output filename
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_csv = os.path.join(output_dir, f"{base_name}_Fxrd.csv")

    # Delete 'cif' column
    if 'cif' in df.columns:
        df = df.drop(columns=['cif'])
        print("Deleted 'cif' column from output file")

    # Delete 'struct_type' column if it exists
    if 'struct_type' in df.columns:
        df = df.drop(columns=['struct_type'])
        print("Deleted 'struct_type' column from output file")

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\nProcessing completed! Results saved to: {output_csv}")

    return df


if __name__ == "__main__":
    # Input file path
    input_csv = r'./Dateset_Feature/OQMD_7176_POSCAR_CIF_St_El.csv'

    # Check if file exists
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"File not found: {input_csv}")

    # Run processing
    print("===== XRD Characteristic Peak Extraction Program =====")
    print(f"Processing file: {input_csv}")
    result_df = process_cif_file(input_csv)

    # Print first few rows as preview
    if result_df is not None and not result_df.empty:
        print("\nResult preview:")
        preview_cols = ['num_peaks'] + [f'peak{i}_{metric}' for i in range(1, 7) for metric in
                                        ['intensity', 'position']]
        preview_df = result_df.head(3)[preview_cols]
        print(preview_df)

    print("\nProgram execution completed!")
