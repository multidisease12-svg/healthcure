import sqlite3

# Path to your existing users.db
db_path = r"C:\Users\shrut\OneDrive\Desktop\7 disease\7 disease\instance\users.db"

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if 'users' table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
if cursor.fetchone():
    # Check if 'email' column already exists
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'email' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
        conn.commit()
        print("✅ 'email' column added successfully — existing data preserved.")
    else:
        print("ℹ️ 'email' column already exists — no change needed.")
else:
    print("⚠️ Table 'users' does not exist. Please check your database path.")

conn.close()
