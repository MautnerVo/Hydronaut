import pandas as pd

excel_path = r"C:\Users\vojtu\Downloads\EMG_anotace_16_test.xlsx"
df_raw = pd.read_excel(excel_path, sheet_name="EMG annotation", header=None)

header_rows = df_raw.iloc[1:4].copy()

header_rows.iloc[0] = header_rows.iloc[0].ffill(axis=0)

header_rows.iloc[1] = header_rows.iloc[1].ffill(axis=0)
header_rows.iloc[2] = header_rows.iloc[2].ffill(axis=0)

multi_index = pd.MultiIndex.from_arrays(header_rows.values, names=["Block", "Exercise", "Type"])

df_data = df_raw.iloc[4:].copy()
df_data.columns = multi_index
df_data.reset_index(drop=True, inplace=True)

print(df_data.xs("MVC", level="Block", axis=1))