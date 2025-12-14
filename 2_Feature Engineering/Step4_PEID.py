import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import os

# Read element property tables
prop_df = pd.read_csv(r"./Element/Nd_EG.csv")
radius_df = pd.read_csv(r"./Element/CovalentRadius.csv")

# Create element-to-property mapping dictionaries
eg_map = dict(zip(prop_df['Symbol'], prop_df['EG']))
nd_map = dict(zip(prop_df['Symbol'], prop_df['Nd']))
radius_map = dict(zip(radius_df['Symbol'], radius_df['CovalentRadius']))


# Parse chemical formula function
def parse_formula(formula):
    elements = {}
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)
    for elem, count in matches:
        count = int(count) if count else 1
        elements[elem] = count
    return elements


# Calculate minimum image distance function
def min_image_distance(coord1, coord2, lattice):
    frac_diff = coord1 - coord2
    frac_diff -= np.round(frac_diff)
    cart_diff = np.dot(frac_diff, lattice)
    return np.linalg.norm(cart_diff)


# Main processing function
def process_dataset(input_path, output_path):
    # Read dataset
    df = pd.read_csv(input_path)

    # Prepare new columns
    new_columns = []
    # Non-structure related features
    for pos1 in ['A', 'B', 'X']:
        for pos2 in ['A', 'B', 'X']:
            for prop in ['EG', 'Nd']:
                for op in ['+', '-', '*', '/']:
                    if prop == 'Nd' and op in ['*', '/']:
                        continue
                    new_columns.append(f"{prop}_{pos1}{op}{pos2}")

    # Structure related features
    for pos in ['A', 'B', 'X']:
        for prop in ['EG', 'Nd']:
            for op in ['+', '-', '*', '/']:
                if prop == 'Nd' and op in ['*', '/']:
                    continue
                new_columns.append(f"{pos}_{prop}{op}")

    # Initialize new columns
    for col in new_columns:
        df[col] = np.nan

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ntypes = row['ntypes']
        formula = row['name']
        poscar_str = row['poscar']

        # Parse chemical formula
        elements_count = parse_formula(formula)

        # Determine A, B, X site elements
        if ntypes == 3:  # Single perovskite
            # Find X site (atom count = 3)
            X_elem = [elem for elem, count in elements_count.items() if count == 3][0]
            # Remaining two elements
            other_elems = [elem for elem in elements_count.keys() if elem != X_elem]
            # Determine A and B sites based on atomic radius
            radii = {elem: radius_map.get(elem, np.nan) for elem in other_elems}
            sorted_elems = sorted(other_elems, key=lambda x: radii[x], reverse=True)
            A_elem, B_elem = sorted_elems
            B_elements = [B_elem]
        elif ntypes == 4:  # Double perovskite
            # Find A site (atom count = 2) and X site (atom count = 6)
            A_elem = [elem for elem, count in elements_count.items() if count == 2][0]
            X_elem = [elem for elem, count in elements_count.items() if count == 6][0]
            # Remaining two elements are B site
            B_elements = [elem for elem, count in elements_count.items()
                          if elem not in [A_elem, X_elem] and count == 1]
        else:
            continue

        # Get property values for each element
        A_eg = eg_map.get(A_elem, np.nan)
        A_nd = nd_map.get(A_elem, np.nan)
        B_eg = np.mean([eg_map.get(b, np.nan) for b in B_elements])
        B_nd = np.mean([nd_map.get(b, np.nan) for b in B_elements])
        X_eg = eg_map.get(X_elem, np.nan)
        X_nd = nd_map.get(X_elem, np.nan)

        # Calculate total property values for each site (considering atom count)
        A_total_eg = elements_count[A_elem] * A_eg
        A_total_nd = elements_count[A_elem] * A_nd
        B_total_eg = sum(elements_count[b] * eg_map.get(b, np.nan) for b in B_elements)
        B_total_nd = sum(elements_count[b] * nd_map.get(b, np.nan) for b in B_elements)
        X_total_eg = elements_count[X_elem] * X_eg
        X_total_nd = elements_count[X_elem] * X_nd

        # Store position property values
        pos_values = {
            'A': {'EG': A_total_eg, 'Nd': A_total_nd},
            'B': {'EG': B_total_eg, 'Nd': B_total_nd},
            'X': {'EG': X_total_eg, 'Nd': X_total_nd}
        }

        # Calculate non-structure related features
        for pos1 in ['A', 'B', 'X']:
            for pos2 in ['A', 'B', 'X']:
                for prop in ['EG', 'Nd']:
                    val1 = pos_values[pos1][prop]
                    val2 = pos_values[pos2][prop]

                    for op in ['+', '-', '*', '/']:
                        if prop == 'Nd' and op in ['*', '/']:
                            continue

                        col_name = f"{prop}_{pos1}{op}{pos2}"
                        if op == '+':
                            df.at[idx, col_name] = val1 + val2
                        elif op == '-':
                            df.at[idx, col_name] = abs(val1 - val2)
                        elif op == '*':
                            df.at[idx, col_name] = val1 * val2
                        elif op == '/':
                            df.at[idx, col_name] = val1 / val2 if val2 != 0 else np.nan

        # Parse POSCAR
        lines = poscar_str.strip().split('\n')
        lattice = []
        for i in range(2, 5):
            lattice.append([float(x) for x in lines[i].split()])
        lattice = np.array(lattice)

        # Atom information
        atom_types = lines[5].split()
        atom_counts = [int(x) for x in lines[6].split()]
        coord_type = lines[7].strip()

        atoms = []
        index = 8
        for i, count in enumerate(atom_counts):
            for j in range(count):
                coords = [float(x) for x in lines[index].split()[:3]]
                index += 1
                element = atom_types[i]

                # Mark atom position
                if element == A_elem:
                    position = 'A'
                elif element in B_elements:
                    position = 'B'
                elif element == X_elem:
                    position = 'X'
                else:
                    position = 'Unknown'

                atoms.append({
                    'element': element,
                    'coords': np.array(coords),
                    'position': position
                })

        # Calculate structure related features
        for center_pos in ['A', 'B', 'X']:
            center_atoms = [a for a in atoms if a['position'] == center_pos]
            other_atoms = [a for a in atoms if a['position'] != center_pos]

            if not center_atoms or not other_atoms:
                continue

            for prop in ['EG', 'Nd']:
                prop_map = eg_map if prop == 'EG' else nd_map

                for op in ['+', '-', '*', '/']:
                    if prop == 'Nd' and op in ['*', '/']:
                        continue

                    col_name = f"{center_pos}_{prop}{op}"
                    total_coupling = 0

                    for center_atom in center_atoms:
                        center_val = prop_map.get(center_atom['element'], np.nan)
                        atom_coupling = 0

                        for other_atom in other_atoms:
                            other_val = prop_map.get(other_atom['element'], np.nan)
                            dist = min_image_distance(
                                center_atom['coords'],
                                other_atom['coords'],
                                lattice
                            )

                            if op == '+':
                                coupling_val = center_val + other_val
                            elif op == '-':
                                coupling_val = abs(center_val - other_val)
                            elif op == '*':
                                coupling_val = center_val * other_val
                            elif op == '/':
                                coupling_val = center_val / other_val if other_val != 0 else np.nan

                            atom_coupling += coupling_val * dist

                        total_coupling += atom_coupling

                    # Take average
                    df.at[idx, col_name] = total_coupling / len(center_atoms)

    # Save results
    df.to_csv(output_path, index=False)


# Input and output paths
input_path = r".\Dateset_Feature\OQMD_7176_POSCAR_CIF_St_El_Fxrd.csv"
output_path = os.path.splitext(input_path)[0] + "_PEID.csv"

# Execute processing
process_dataset(input_path, output_path)
print(f"Processing completed! Results saved to: {output_path}")
