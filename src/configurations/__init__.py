import configparser
from typing import List, Dict


class ConfigParser:
    def __init__(self, components: List[str]):
        self.configs: Dict[str, configparser.ConfigParser] = {}
        for component in components:
            config_file = f"/configs/{component}.conf"
            parser = configparser.ConfigParser()
            parser.read(config_file)
            self.configs[component] = parser

    def get(self, component: str):
        return self.configs[component]
