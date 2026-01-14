import numpy as np
from pymilvus import Collection
from typing import List, Dict, Any
from src.collection_manager import MilvusCollectionManager
from src.data_model import DocumentData
from src.config_loader import ConfigLoader

class MilvusDataReader:
    def __init__(self, collection_manager: MilvusCollectionManager = None):
        if collection_manager is None:
            self.collection_manager = MilvusCollectionManager()
        else:
            self.collection_manager = collection_manager

        self.config_loader = ConfigLoader()
        self.collection_config = self.config_loader.get_collection_config()

        # 确保集合存在
        if not self.collection_manager.has_collection():
            raise Exception("Collection does not exist. Create it first.")

        self.collection = self.collection_manager.get_collection()
        self.collection.load()  # 加载集合到内存

    def query_by_ids(self, ids: List[int], output_fields: List[str] = None):
        """通过ID查询文档"""
        if output_fields is None:
            output_fields = ["id", "text", "embedding", "metadata"]

        expr = f"id in {ids}"
        results = self.collection.query(
            expr=expr,
            output_fields=output_fields
        )

        # 将结果转换为DocumentData对象
        documents = []
        for result in results:
            doc = DocumentData(
                id=result["id"],
                text=result["text"],
                embedding=result["embedding"],
                metadata=result.get("metadata", {})
            )
            documents.append(doc)

        return documents

    def search_by_vector(self, query_vector: List[float], top_k: int = 10, filter_expr: str = None):
        """通过向量搜索最相似的文档"""
        search_params = self.collection_config.get('search_params', {"nprobe": 16})

        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["text", "metadata"]
        )

        # 处理搜索结果
        documents = []
        for hits in results:
            for hit in hits:
                doc = DocumentData(
                    id=hit.id,
                    text=hit.entity.get("text"),
                    embedding=[],  # 搜索结果通常不返回完整的向量
                    metadata=hit.entity.get("metadata", {})
                )
                doc.distance = hit.distance  # 添加距离信息
                documents.append(doc)

        return documents

    def count_documents(self, filter_expr: str = None):
        """计算集合中的文档数量"""
        return self.collection.num_entities
