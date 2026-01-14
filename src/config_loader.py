import yaml
from pathlib import Path


class ConfigLoader:
    def __init__(self, config_path=None):
        if config_path is None:
            # 默认配置文件路径
            config_path = Path(__file__).parent.parent / 'config' / 'milvus_config.yaml'

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_milvus_config(self):
        return self.config['milvus']

    def get_collection_config(self):
        return self.config['milvus']['collection']

    def get_embedding_config(self):
        return self.config['embedding']
