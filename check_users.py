import sqlite3

# Path to your users database
db_path = "C:/Users/shrut/OneDrive/Desktop/7 disease/7 disease/instance/users.db"

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if 'users' table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
table_exists = cursor.fetchone()

if table_exists:
    # ✅ Ensure 'email' column exists
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'email' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")
        conn.commit()
        print("✅ Added missing 'email' column to users table.")
    else:
        print("ℹ️ 'email' column already exists.")

    # Fetch all users
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    print("Users in database:")
    for user in users:
        print(user)

conn.close()
