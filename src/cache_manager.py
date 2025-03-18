import sqlite3
from config import CACHE_DB_PATH

# Ensure the cache table exists
def initialize_cache_db():
    conn = sqlite3.connect(CACHE_DB_PATH)
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS cache (query TEXT PRIMARY KEY, response TEXT)")
    conn.commit()
    conn.close()  # ✅ Close connection immediately after use

# Ensure DB initialization on import
initialize_cache_db()

def get_db_connection():
    """Returns a new SQLite connection for each request."""
    return sqlite3.connect(CACHE_DB_PATH)

def get_cached_response(query):
    """Retrieve a cached response from the database."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT response FROM cache WHERE query = ?", (query,))
    row = c.fetchone()
    conn.close()  # ✅ Close connection after use
    return row[0] if row else None

def cache_response(query, response):
    """Insert a new cached response into the database."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO cache VALUES (?, ?)", (query, response))  # ✅ Prevent UNIQUE constraint errors
        conn.commit()
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        conn.close()  # ✅ Close connection to avoid issues
