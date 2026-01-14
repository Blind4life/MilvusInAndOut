from src.collection_manager import MilvusCollectionManager
from src.config_loader import ConfigLoader
from src.data_model import DocumentData
from src.data_reader import MilvusDataReader
from src.data_writer import MilvusDataWriter
from src.vector_encoder import DashScopeEncoder


def test():
    # 初始化集合管理器
    collection_manager = MilvusCollectionManager()

    # 如果需要，创建新集合（如果已存在则获取现有的）
    collection = collection_manager.create_collection()
    print(f"Collection schema: {collection.schema}")

    # 初始化数据写入器
    writer = MilvusDataWriter(collection_manager)

    # 获取DashScope API密钥，可以从环境变量中获取
    config_loader = ConfigLoader()
    dashscope_api_key = config_loader.get_embedding_config()["apiKey"]
    dashscope_model = config_loader.get_embedding_config()["model"]
    if not dashscope_api_key:
        print("警告：未设置DASHSCOPE_API_KEY环境变量，将使用随机向量而不是DashScope API")
        raise Exception("api_key 错误")

    # 创建一些示例文档
    docs = [
        DocumentData(
            id=1,
            text="这是第一个测试文档，关于人工智能技术",
            embedding=[],  # 暂时为空，稍后填充
            metadata={"source": "test", "category": "AI", "tags": ["machine learning", "neural networks"]}
        ),
        DocumentData(
            id=2,
            text="这是第二个测试文档，讨论云计算服务",
            embedding=[],  # 暂时为空，稍后填充
            metadata={"source": "test", "category": "Cloud Computing", "provider": "Alibaba Cloud"}
        ),
        DocumentData(
            id=3,
            text="这是第三个测试文档，探讨大数据分析方法",
            embedding=[],  # 暂时为空，稍后填充
            metadata={"source": "test", "category": "Big Data", "technologies": ["Hadoop", "Spark", "Flink"]}
        ),
    ]
    encoder = DashScopeEncoder(dashscope_api_key, dashscope_model)
    # 使用DashScope编码文本和元数据
    docs = encoder.encode_document_data(docs)

    # 插入文档
    writer.insert_data(docs)

    # 初始化数据读取器
    reader = MilvusDataReader(collection_manager)

    # 通过ID查询
    query_ids = [1, 3]
    print(f"\n查询ID为 {query_ids} 的文档:")
    query_results = reader.query_by_ids(query_ids)
    for doc in query_results:
        print(f"ID: {doc.id}, Text: {doc.text}")
        print(f"Metadata: {doc.metadata}")

    # 向量搜索示例

    print("\n使用DashScope生成查询向量并搜索相似文档:")
    # 使用DashScope生成查询向量
    query_text = "人工智能和机器学习技术"
    query_vector = encoder.encode_text(query_text)
    print(f"查询文本: '{query_text}'")

    search_results = reader.search_by_vector(query_vector, top_k=2)
    for i, doc in enumerate(search_results):
        print(f"结果 {i+1} - ID: {doc.id}, Text: {doc.text}, 距离: {doc.distance}")
        print(f"Meta{doc.metadata}")

    # 计数文档
    doc_count = reader.count_documents()
    print(f"\n集合中的文档总数: {doc_count}")

if __name__ == "__main__":
    main()
