import pandas as pd

csv_df = pd.read_csv("Student Programme Information (6).csv")
xlsx_df = pd.read_excel("Student Programme Information (8).xlsx")

merged = pd.concat([csv_df, xlsx_df], ignore_index=True)
merged.drop_duplicates(inplace=True)

merged.to_csv("Student Programme Information (merged).csv", index=False)
print(f"csv rows: {len(csv_df)}, xlsx rows: {len(xlsx_df)}, merged rows: {len(merged)}")
