import pandas as pd
from pathlib import Path

SHILLER_FILE = Path("framework/data/shiller_ie_data.xls")

df = pd.read_excel(SHILLER_FILE, sheet_name="Data", header=7)
df = df.iloc[:, :7].copy()
df.columns = ["date_raw", "price", "dividend", "earnings", "cpi", "col5", "cape"]
df = df[pd.to_numeric(df["date_raw"], errors="coerce").notna()].copy()
df["date_raw"] = df["date_raw"].astype(float)

year = df["date_raw"].astype(int)
month_frac = (df["date_raw"] - year).round(4)
month = (month_frac * 12).round(0).astype(int).clip(1, 12)
df["date"] = pd.to_datetime(dict(year=year, month=month, day=1))
df = df.set_index("date").sort_index()
df["cape"] = pd.to_numeric(df["cape"], errors="coerce")

print("Total rows:", len(df))
print("Duplicate dates:", df.index.duplicated().sum())
print()

# Check what .loc returns for a single date
test_date = df.index[500]
val = df.loc[test_date, "cape"]
print("Test date:", test_date)
print("Type returned by .loc:", type(val))
print("Value:", val)
print()

# If it is a Series, show why
if isinstance(val, pd.Series):
    print("Series has length:", len(val))
    print("This means that date appears", len(val), "times in the index")
    print("First few duplicate dates:")
    print(df[df.index.duplicated(keep=False)].head(10))