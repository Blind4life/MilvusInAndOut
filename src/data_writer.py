from typing import List

from src.collection_manager import MilvusCollectionManager
from src.data_model import DocumentData


class MilvusDataWriter:
    def __init__(self, collection_manager: MilvusCollectionManager = None):
        if collection_manager is None:
            self.collection_manager = MilvusCollectionManager()
        else:
            self.collection_manager = collection_manager

        # 确保集合存在
        if not self.collection_manager.has_collection():
            self.collection = self.collection_manager.create_collection()
        else:
            self.collection = self.collection_manager.get_collection()

    def insert_data(self, documents: List[DocumentData]):
        """将文档插入到Milvus集合中"""
        # 准备数据
        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadata = [doc.metadata if doc.metadata else {} for doc in documents]

        # 构建插入的实体
        entities = [ids, texts, embeddings, metadata]

        # 插入数据
        insert_result = self.collection.insert(entities)

        # 加载集合以使更改生效
        self.collection.load()

        print(f"Inserted {len(documents)} documents. Inserted IDs: {insert_result.primary_keys}")
        return insert_result.primary_keys

    def insert_single_document(self, document: DocumentData):
        """插入单个文档"""
        return self.insert_data([document])

    def delete_by_ids(self, ids: List[int]):
        """通过ID删除文档"""
        expr = f"id in {ids}"
        self.collection.delete(expr)
        print(f"Deleted documents with IDs: {ids}")
