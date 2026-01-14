from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class DocumentData:
    """
    表示文档的数据类
    """
    id: int
    text: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata if self.metadata else {}
        }

    @classmethod
    def from_dict(cls, data_dict):
        return cls(
            id=data_dict["id"],
            text=data_dict["text"],
            embedding=data_dict["embedding"],
            metadata=data_dict.get("metadata", {})
        )
