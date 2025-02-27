from setuptools import setup

APP = ['launch.py']
DATA_FILES = [
    ('', ['workouts.db']),
    ('pages', ['pages/View_Progress.py']),
]

OPTIONS = {
    'argv_emulation': True,
    'packages': [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'sklearn',
        'prophet',
        'scipy',
        'statsmodels',
    ],
    'plist': {
        'CFBundleName': 'Workout Tracker',
        'CFBundleDisplayName': 'Workout Tracker',
        'CFBundleIdentifier': 'com.workout.tracker',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
    },
}

setup(
    name='WorkoutTracker',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
