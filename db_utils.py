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

    # Workouts table (Updated to include muscle_type and workout_type)
    c.execute('''
    CREATE TABLE IF NOT EXISTS workouts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workout TEXT,
        weight REAL,
        sets INTEGER,
        reps INTEGER,
        date TEXT,
        muscle_type TEXT,
        workout_type TEXT
    )
    ''')

    # Check and add missing columns if needed for workouts table
    c.execute("PRAGMA table_info(workouts)")
    existing_columns = [row[1] for row in c.fetchall()]

    if "muscle_type" not in existing_columns:
        c.execute("ALTER TABLE workouts ADD COLUMN muscle_type TEXT")

    if "workout_type" not in existing_columns:
        c.execute("ALTER TABLE workouts ADD COLUMN workout_type TEXT")

    # Body metrics table (unchanged)
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

    # Exercises table (Updated to include muscle_type and workout_type)
    c.execute('''
    CREATE TABLE IF NOT EXISTS exercises (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        muscle_type TEXT,
        workout_type TEXT
    )
    ''')

    # Check and add missing columns if needed for exercises table
    c.execute("PRAGMA table_info(exercises)")
    existing_columns = [row[1] for row in c.fetchall()]

    if "muscle_type" not in existing_columns:
        c.execute("ALTER TABLE exercises ADD COLUMN muscle_type TEXT")

    if "workout_type" not in existing_columns:
        c.execute("ALTER TABLE exercises ADD COLUMN workout_type TEXT")

    # Insert some default preset exercises if not already inserted
    default_presets = [
        ("Bench Press", "Chest", "Push"),
        ("Squat", "Legs", "Leg"),
        ("Deadlift", "Back", "Pull"),
        ("Shoulder Press", "Shoulders", "Push"),
        ("Pull-ups", "Back", "Pull")
    ]

    for preset, muscle, workout_type in default_presets:
        try:
            c.execute("INSERT INTO exercises (name, muscle_type, workout_type) VALUES (?, ?, ?)",
                      (preset, muscle, workout_type))
        except sqlite3.IntegrityError:
            pass  # Ignore if it already exists

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
