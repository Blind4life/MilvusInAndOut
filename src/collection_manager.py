from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility

from src.config_loader import ConfigLoader


class MilvusCollectionManager:
    def __init__(self, config_path=None):
        self.config_loader = ConfigLoader(config_path)
        self.milvus_config = self.config_loader.get_milvus_config()
        self.collection_config = self.config_loader.get_collection_config()

        # 连接Milvus服务器
        connections.connect(
            host=self.milvus_config['host'],
            port=self.milvus_config['port']
        )

        self.collection_name = self.collection_config['name']

    def create_collection(self):
        """创建Milvus集合"""
        if self.has_collection():
            print(f"Collection '{self.collection_name}' already exists.")
            return self.get_collection()

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.collection_config['dim']),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        # 创建集合模式
        schema = CollectionSchema(fields=fields, description="Document collection")

        # 创建集合
        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using='default',
            shards_num=2
        )

        # 创建索引
        '''
        todo 先不做索引的格式化要求
        '''
        index_params = {
            "index_type": self.collection_config['index_type'],
            "metric_type": self.collection_config['metric_type'],
            "params": self.collection_config['index_params']
        }

        collection.create_index(
            field_name="embedding"
            , index_params=index_params
        )

        print(f"Collection '{self.collection_name}' created successfully.")
        return collection

    def has_collection(self):
        """检查集合是否存在"""
        return utility.has_collection(self.collection_name)

    def get_collection(self):
        """获取集合"""
        if not self.has_collection():
            raise Exception(f"Collection '{self.collection_name}' does not exist.")

        collection = Collection(self.collection_name)
        return collection

    def delete_collection(self):
        """删除集合"""
        if self.has_collection():
            utility.drop_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' deleted successfully.")
        else:
            print(f"Collection '{self.collection_name}' does not exist.")
