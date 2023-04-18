import json
import os


class Config:
    def __init__(self, config_file):
        self.config_file = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), config_file)

        # Parse configuration file and load values
        self.data_dir = None
        self.output_dir = None
        self.model_path = None
        self.load_config()

    def load_config(self):
        # Load values from configuration file
        with open(self.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            self.data_dir = config["data_dir"]
            self.output_dir = config["output_dir"]
            self.model_path = config["model_path"]
        return self


# # Usage example
# config = Config("config.json")
# print(config.data_dir)  # /path/to/data
