import sqlite3
import pandas as pd
import os

# ----------------------------
# Path to your database
# ----------------------------
db_path = os.path.join("instance", "predictions.db")

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in database:", tables)

# Set pandas display options
pd.set_option('display.max_rows', None)       # Show all rows
pd.set_option('display.max_columns', None)    # Show all columns
pd.set_option('display.width', 1000)          # Wide output
pd.set_option('display.max_colwidth', None)   # Allow truncation manually

# Check the exact table name (case-insensitive)
table_name = None
for t in tables:
    if t[0].lower() == 'prediction':
        table_name = t[0]  # Use exact case
        break

if table_name:
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # ----------------------------
    # Truncate image column to very small size
    # ----------------------------
    if "image" in df.columns:
        df["image"] = df["image"].str.slice(0, 20) + "..."  # small and readable

    # Print the dataframe (one row per record)
    print(df.to_string(index=False))

else:
    print("No Prediction table found.")

conn.close()
