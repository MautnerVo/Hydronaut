"""
Segmentace signálů podle anotací
"""

import pandas as pd
import os
from annotation_loader import process_file

# excel_path = r"Y:\Datasets\Fyzio\2025-03-07\2\EMG_anotace_2.xlsx"
# df_raw = pd.read_excel(excel_path, sheet_name="EMG annotation", header=None)
#
# header_rows = df_raw.iloc[1:4].copy()
#
# header_rows.iloc[0] = header_rows.iloc[0].ffill(axis=0)
#
# header_rows.iloc[1] = header_rows.iloc[1].ffill(axis=0)
# header_rows.iloc[2] = header_rows.iloc[2].ffill(axis=0)
#
# multi_index = pd.MultiIndex.from_arrays(header_rows.values, names=["Block", "Exercise", "Type"])
#
# df_data = df_raw.iloc[4:].copy()
# df_data.columns = multi_index
# df_data.reset_index(drop=True, inplace=True)
#
# print(df_data.xs("BLOCK 2: Dynamic warm-up", level="Block", axis=1))


number = 20
folder = fr"Y:\Datasets\Fyzio\2025-03-28\{number}"
annotation = f"EMG_anotace_{number}.xlsx"
signals = "Emg_Imu.csv"
os.makedirs("exercises_signals", exist_ok=True)
output = process_file(os.path.join(folder,annotation))

df = pd.read_csv(os.path.join(folder,signals))

for exercise,row in output.items():
    sub_df = df[(df["Sample"] >= row['begin']) & (df["Sample"] <= row['end'])]
    sub_df.to_csv(rf"exercises_signals/{exercise}.csv", index=False)