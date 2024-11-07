import yaml
from pathlib import Path
import logging


def load_config(config_path: str = "config/bot_config.yaml") -> dict:
    """Load bot configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BotSettings:
    def __init__(self):
        config = load_config()

        # Bot settings
        bot_config = config.get("bot", {})
        self.token = bot_config.get("token")
        self.api_url = bot_config.get("api_url")

        if not self.token:
            raise ValueError("Bot token not found in config")
        if not self.api_url:
            raise ValueError("API URL not found in config")

        # Logging settings
        log_config = config.get("logging", {})
        self.log_level = log_config.get("level", "INFO")
        self.log_format = log_config.get("format")
