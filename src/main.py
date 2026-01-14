from flask import Flask, request, jsonify

from src.collection_manager import MilvusCollectionManager
from src.config_loader import ConfigLoader
from src.data_model import DocumentData
from src.data_reader import MilvusDataReader
from src.data_writer import MilvusDataWriter
from src.vector_encoder import DashScopeEncoder

app = Flask(__name__)

# 全局变量存储组件实例
collection_managers = {}
writers = {}
readers = {}
encoder = None


def initialize_encoder():
    """初始化向量编码器"""
    global encoder
    config_loader = ConfigLoader()
    dashscope_api_key = config_loader.get_embedding_config()["apiKey"]
    dashscope_model = config_loader.get_embedding_config()["model"]

    if not dashscope_api_key:
        raise Exception("未配置DashScope API密钥")

    encoder = DashScopeEncoder(dashscope_api_key, dashscope_model)
    return encoder


@app.route('/init_database', methods=['POST'])
def init_database():
    """
    初始化数据库端点
    参数:
        - collection_name: 数据库名称
    """
    try:
        data = request.json
        collection_name = data.get('collection_name')

        if not collection_name:
            return jsonify({"error": "缺少collection_name参数"}), 400

        # 创建自定义配置路径（如果有）
        config_path = data.get('config_path')

        # 初始化集合管理器
        collection_manager = MilvusCollectionManager(config_path)

        # 修改集合名称
        collection_manager.collection_name = collection_name

        # 创建集合
        collection = collection_manager.create_collection()

        # 存储实例到全局变量
        collection_managers[collection_name] = collection_manager
        writers[collection_name] = MilvusDataWriter(collection_manager)
        readers[collection_name] = MilvusDataReader(collection_manager)

        # 确保编码器已初始化
        if encoder is None:
            initialize_encoder()

        return jsonify({
            "message": f"数据库 '{collection_name}' 初始化成功",
            "schema": str(collection.schema)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/store_data', methods=['POST'])
def store_data():
    """
    存储数据端点
    参数:
        - collection_name: 数据库名称
        - document: DocumentData 对象数据
    """
    try:
        data = request.json
        collection_name = data.get('collection_name')
        document_data = data.get('document')

        if not collection_name or not document_data:
            return jsonify({"error": "缺少必要参数"}), 400

        # 检查集合是否已初始化
        if collection_name not in writers:
            return jsonify({"error": f"数据库 '{collection_name}' 未初始化"}), 404

        writer = writers[collection_name]

        # 确保编码器已初始化
        if encoder is None:
            initialize_encoder()

        # 创建DocumentData对象
        doc = DocumentData(
            id=document_data.get('id'),
            text=document_data.get('text'),
            embedding=[],  # 暂时为空
            metadata=document_data.get('metadata', {})
        )

        # 使用编码器生成向量
        encoded_docs = encoder.encode_document_data([doc])

        # 存储文档
        inserted_ids = writer.insert_data(encoded_docs)

        return jsonify({
            "message": "数据存储成功",
            "inserted_ids": inserted_ids
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """
    向量搜索端点
    参数:
        - collection_name: 数据库名称
        - query_text: 搜索文本
        - top_k: 返回结果数量
    """
    try:
        data = request.json
        collection_name = data.get('collection_name')
        query_text = data.get('query_text')
        top_k = data.get('top_k', 10)

        if not collection_name or not query_text:
            return jsonify({"error": "缺少必要参数"}), 400

        # 检查集合是否已初始化
        if collection_name not in readers:
            return jsonify({"error": f"数据库 '{collection_name}' 未初始化"}), 404

        reader = readers[collection_name]

        # 确保编码器已初始化
        if encoder is None:
            initialize_encoder()

        # 生成查询向量
        query_vector = encoder.encode_text(query_text)

        # 搜索相似文档
        search_results = reader.search_by_vector(query_vector, top_k=top_k)

        # 准备响应
        results = []
        for i, doc in enumerate(search_results):
            results.append({
                "rank": i + 1,
                "id": doc.id,
                "text": doc.text,
                "similarity": 1.0 - doc.distance,  # 转换距离为相似度
                "distance": doc.distance,
                "metadata": doc.metadata
            })

        return jsonify({
            "query_text": query_text,
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "ok", "message": "服务正常运行"})


@app.route('/collections', methods=['GET'])
def list_collections():
    """列出所有已初始化的集合"""
    return jsonify({
        "collections": list(collection_managers.keys())
    })


@app.route('/collections/<collection_name>/count', methods=['GET'])
def count_documents(collection_name):
    """获取集合中的文档数量"""
    if collection_name not in readers:
        return jsonify({"error": f"数据库 '{collection_name}' 未初始化"}), 404

    reader = readers[collection_name]
    count = reader.count_documents()

    return jsonify({
        "collection_name": collection_name,
        "document_count": count
    })


if __name__ == '__main__':
    # 尝试初始化编码器
    try:
        initialize_encoder()
    except Exception as e:
        print(f"警告: 编码器初始化失败: {e}")
        print("服务仍将启动，但在使用相关功能前需要配置正确的API密钥")

    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True)
