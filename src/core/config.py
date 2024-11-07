import yaml
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    config_path = Path("config/api_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


class Settings:
    def __init__(self):
        config = load_config()

        self.server = config["server"]
        self.host = self.server["host"]
        self.port = self.server["port"]
        self.debug = self.server["debug"]
        self.reload = self.server["reload"]
        self.timeout = self.server["timeout"]
        self.workers = self.server["workers"]

        # API settings
        self.api = config["api"]
        self.api_title = self.api["title"]
        self.api_description = self.api["description"]
        self.prefix = self.api["prefix"]
        self.api_version = self.api["version"]

        # Models settings
        self.models = config["models"]
        self.max_active_models = self.models["max_active_models"]
