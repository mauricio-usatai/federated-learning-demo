import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    CLIENT_ID: int = os.environ.get("CLIENT_ID")
    POOLING_TIME: int = 1  # Seconds

    # Local database
    LOCAL_DATABASE: str = f"./local_data/local_data_{CLIENT_ID}.db"

    # Training metadata
    TRAIN_HISTORY_DIR: str = f"./local_data/train_history/client_{CLIENT_ID}"

    # Global sefver options
    GLOBAL_SERVER_URL: str = "http://server"
    GLOBAL_SERVER_PORT: int = 8000
    SERVER_URL: str = f"{GLOBAL_SERVER_URL}:{GLOBAL_SERVER_PORT}"
    SERVER_TIMEOUT: int = 5

    RANDOM_STATE: int = 1
