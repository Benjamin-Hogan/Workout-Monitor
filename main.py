import streamlit as st
from db_utils import init_db


def main():
    # Initialize the DB with necessary tables/columns
    init_db()


if __name__ == "__main__":
    main()
