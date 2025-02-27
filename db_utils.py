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
      - progress_photos
      - body_metrics_goals
      - users
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

    # Body metrics table (updated to include all required columns)
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
        arms REAL,
        glutes REAL,
        thigh REAL,
        calf REAL,
        neck REAL
    )
    ''')

    # Check and add missing columns if needed for body_metrics table
    c.execute("PRAGMA table_info(body_metrics)")
    existing_columns = [row[1] for row in c.fetchall()]

    # Add missing columns to body_metrics if they don't exist
    for column, type_ in [
        ("glutes", "REAL"),
        ("thigh", "REAL"),
        ("calf", "REAL"),
        ("neck", "REAL")
    ]:
        if column not in existing_columns:
            try:
                c.execute(
                    f"ALTER TABLE body_metrics ADD COLUMN {column} {type_}")
            except sqlite3.OperationalError:
                # Column might already exist or table doesn't exist yet
                pass

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

    # Progress photos table
    c.execute('''
    CREATE TABLE IF NOT EXISTS progress_photos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_date DATE NOT NULL,
        photo_data BLOB NOT NULL,
        photo_type TEXT NOT NULL,
        notes TEXT
    )
    ''')

    # Body metrics goals table
    c.execute('''
    CREATE TABLE IF NOT EXISTS body_metrics_goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_name TEXT NOT NULL,
        target_value REAL NOT NULL,
        target_date DATE NOT NULL,
        created_date DATE DEFAULT CURRENT_DATE,
        achieved BOOLEAN DEFAULT 0
    )
    ''')

    # Users table (for authentication)
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        salt TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

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
