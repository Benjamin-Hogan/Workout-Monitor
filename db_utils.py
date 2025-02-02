import sqlite3

DB_NAME = "workouts.db"


def create_connection():
    """Create or connect to a local SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    return conn


def init_db():
    """
    Create tables if they do not exist, including:
      - workouts
      - body_metrics
      - exercises (for workout presets)
    Insert default exercises as presets if they're not present.
    """
    conn = create_connection()
    c = conn.cursor()

    # Workouts table
    c.execute('''
    CREATE TABLE IF NOT EXISTS workouts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workout TEXT,
        weight REAL,
        sets INTEGER,
        reps INTEGER,
        date TEXT
    )
    ''')

    # Body metrics table (including optional columns)
    c.execute('''
    CREATE TABLE IF NOT EXISTS body_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_date TEXT,
        user_weight REAL,
        height REAL,
        age INTEGER,
        gender TEXT,
        body_fat REAL,
        chest REAL,
        waist REAL,
        hips REAL,
        arms REAL
    )
    ''')

    # Exercises table (for presets)
    c.execute('''
    CREATE TABLE IF NOT EXISTS exercises (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    )
    ''')

    # Insert some default preset exercises if not already inserted
    default_presets = ["Bench Press", "Squat",
                       "Deadlift", "Shoulder Press", "Pull-ups"]
    for preset in default_presets:
        try:
            c.execute("INSERT INTO exercises (name) VALUES (?)", (preset,))
        except sqlite3.IntegrityError:
            pass  # Ignore if it already exists

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
