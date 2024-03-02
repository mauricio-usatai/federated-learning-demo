import sqlite3
import pandas as pd


class SQLite:
    def __init__(self, database: str = "local.db") -> None:
        self.conn = sqlite3.connect(database)

    def get_local_training_set(self) -> pd.DataFrame:
        """Get all data from local db as a pandas DataFrame"""
        data = pd.read_sql_query(
            "SELECT * FROM data",
            self.conn,
        )

        return data

    def close(self) -> None:
        """Close db connection"""
        self.conn.close()
