"""Create the PostgreSQL tables.  Usage: python -m src.init_db"""
from src.db import init_db

if __name__ == "__main__":
    init_db()
    print("Database tables created.")
