import os

class Settings:
    APP_NAME: str = "SHL Recommender"
    DATA_PATH: str = os.getenv("DATA_PATH", "data/assessments.csv")

settings = Settings()
