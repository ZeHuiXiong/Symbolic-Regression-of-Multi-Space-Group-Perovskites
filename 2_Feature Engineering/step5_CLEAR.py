import pandas as pd
import os


def merge_datasets_with_features():
    # Define file paths
    dataset_path = r"./Dateset_Feature/OQMD_7176_POSCAR_CIF_St_El_Fxrd_PEID.csv"
    features_path = r"./CLEAR/cg_features.csv"

    # Output file path (same directory as dataset)
    output_dir = os.path.dirname(dataset_path)
    output_path = os.path.join(output_dir, "OQMD_7176_with_features.csv")

    try:
        # Read dataset
        print("Reading dataset...")
        dataset_df = pd.read_csv(dataset_path)

        # Check if dataset contains structure_id column
        if "structure_id" not in dataset_df.columns:
            raise ValueError("Dataset file does not contain 'structure_id' column")

        # Read feature data
        print("Reading feature data...")
        features_df = pd.read_csv(features_path)

        # Check if feature data contains POSCAR_FileName column
        if "POSCAR_FileName" not in features_df.columns:
            raise ValueError("Feature file does not contain 'POSCAR_FileName' column")

        # Extract structure_id from POSCAR_FileName
        print("Matching structure_id...")
        # Use regex to extract numeric part from filename as structure_id
        features_df["structure_id"] = features_df["POSCAR_FileName"].str.extract(r'POSCAR_(\d+)').astype(int)

        # Remove original POSCAR_FileName column to avoid duplicate information
        features_df = features_df.drop(columns=["POSCAR_FileName"])

        # Merge two datasets using inner join based on structure_id
        # Inner join ensures only structure_ids present in both sides are kept
        print("Merging data...")
        merged_df = pd.merge(dataset_df, features_df, on="structure_id", how="inner")

        # Save merged dataset
        merged_df.to_csv(output_path, index=False)
        print(f"Merging completed! Total {len(merged_df)} records merged")
        print(f"Results saved to: {output_path}")

        # Statistics
        original_count = len(dataset_df)
        feature_count = len(features_df)
        merged_count = len(merged_df)

        print(f"\nStatistics:")
        print(f"Original dataset record count: {original_count}")
        print(f"Feature dataset record count: {feature_count}")
        print(f"Merged record count: {merged_count}")
        print(f"Successful match rate: {merged_count / original_count:.2%}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    merge_datasets_with_features()
