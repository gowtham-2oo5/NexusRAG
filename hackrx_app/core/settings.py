import os
from dotenv import load_dotenv


class Settings:
    def __init__(self) -> None:
        load_dotenv()
        self.openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        os.environ["OPENAI_API_KEY"] = self.openai_api_key


def get_settings() -> Settings:
    return Settings()


