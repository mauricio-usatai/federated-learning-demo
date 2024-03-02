import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    APP_NAME: str = "fl-demo-server"
    APP_VERSION: str = "0.0.1"

    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    K: int = os.environ.get("K", 4)
    C: float = os.environ.get("C", 1.0)
    E: int = os.environ.get("E", 100)

    MODEL_INPUT_SHAPE: int = 10

    SAVE_MODEL_PATH: str = "./local_data/global_models"
