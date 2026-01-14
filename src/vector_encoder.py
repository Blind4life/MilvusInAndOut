import json
from typing import Dict, Any, List, Optional

from dashscope import TextEmbedding


class DashScopeEncoder:
    """使用阿里云DashScope API进行文本向量编码的类"""

    def __init__(self, api_key: str, model_name: str = "text-embedding-v3"):
        """
        初始化DashScope编码器

        Args:
            api_key: 阿里云DashScope API密钥
            model_name: 要使用的模型名称，默认为text-embedding-v2
        """
        self.api_key = api_key
        self.model_name = model_name

    def _metadata_to_string(self, metadata: Dict[str, Any]) -> str:
        """
        将元数据转换为字符串表示

        Args:
            meta元数据字典

        Returns:
            元数据的字符串表示
        """
        if not metadata:
            return ""

        # 使用json格式化元数据，确保结构化展示
        try:
            formatted_json = json.dumps(metadata, ensure_ascii=False, indent=2)
            return f"Meta{formatted_json}"
        except Exception as e:
            print(f"Error converting metadata to string: {e}")
            # 回退到简单的字符串表示
            return f"Metadata: {str(metadata)}"

    def encode_text(self, text: str) -> List[float]:
        """
        编码单个文本字符串

        Args:
            text: 需要编码的文本

        Returns:
            浮点数向量列表
        """
        try:
            response = TextEmbedding.call(
                model=self.model_name,
                api_key=self.api_key,
                input=text,
            )

            if response.status_code == 200:
                # 提取嵌入向量
                embedding = response.output['embeddings'][0]['embedding']
                return embedding
            else:
                raise Exception(f"API返回错误: {response.status_code}, {response.message}")
        except Exception as e:
            print(f"编码文本时出错: {e}")
            raise

    def encode_text_with_metadata(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        编码文本及其元数据

        Args:
            text: 原始文本
            meta要编码的元数据字典

        Returns:
            浮点数向量列表
        """
        # 将元数据转换为字符串
        metadata_str = self._metadata_to_string(metadata)

        # 如果有元数据，将其添加到文本中
        if metadata_str:
            combined_text = f"{text}\n\n{metadata_str}"
        else:
            combined_text = text

        # 编码组合文本
        return self.encode_text(combined_text)

    def encode_document_data(self, document_data_list: List) -> List:
        """
        编码DocumentData对象列表

        Args:
            document_data_list: DocumentData对象列表

        Returns:
            更新后的DocumentData对象列表，包含编码后的embedding
        """
        for doc in document_data_list:
            # 使用文本和元数据生成向量
            embedding = self.encode_text_with_metadata(doc.text, doc.metadata)
            doc.embedding = embedding

        return document_data_list
