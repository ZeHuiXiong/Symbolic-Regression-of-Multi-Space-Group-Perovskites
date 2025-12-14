import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
import os

input_file = ".\Dateset_Feature\OQMD_7176_POSCAR_CIF_St.csv"
base_name = os.path.splitext(input_file)[0]
output_file = f"{base_name}_El.csv"

if __name__ == "__main__":  #

    df = pd.read_csv(input_file)

    data = StrToComposition().featurize_dataframe(df, "name")

    ep_feat = ElementProperty.from_preset("magpie")
    data = ep_feat.featurize_dataframe(data, col_id="composition")

    data = data.drop(columns=["composition"])
    data.columns = data.columns.str.replace("MagpieData ", "")

    data.to_csv(output_file, index=False)