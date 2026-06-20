"""
可训练的二分类模型

架构: Qwen3-VL-Embedding (冻结) → 全连接分类头 (可训练)

支持两种模式:
  1. online:  每次前向都实时调用 embedding API 提取特征
  2. offline: 预先提取好 embedding，只训练分类头（推荐，大幅加速）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class ClassifierHead(nn.Module):
    """可训练的二分类头 (embedding → 2 classes)"""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list = [512, 128],
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        """
        Args:
            input_dim: 输入 embedding 维度 (2B: 2048, 8B: 4096)
            hidden_dims: 隐藏层维度列表
            dropout: Dropout 比例
            num_classes: 分类数（默认2: HOCM阳/阴）
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        
        # 最后一层：输出 logits
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) 的 embedding
        
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.classifier(x)


class OnlineClassifier(nn.Module):
    """在线模式: embedding API (冻结) + 分类头 (可训练)
    
    每次前向传播都会调用 embedding API，适合数据量不大的场景。
    """
    
    def __init__(
        self,
        embedding_api,
        hidden_dims: list = [512, 128],
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        """
        Args:
            embedding_api: QwenVLEmbeddingAPI 实例（已冻结）
            hidden_dims: 分类头隐藏层
            dropout: Dropout 比例
            num_classes: 分类数
        """
        super().__init__()
        self.embedding_api = embedding_api
        self.embedding_dim = embedding_api.embedding_dim
        
        self.classifier_head = ClassifierHead(
            input_dim=self.embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=num_classes,
        )
    
    def forward(self, video_paths: list) -> torch.Tensor:
        """前向传播
        
        Args:
            video_paths: 视频路径列表, 每个元素为 str
            
        Returns:
            logits: (batch_size, num_classes)
        """
        # Step 1: 提取 embedding (逐个处理，因为 API 批量需统一类型)
        embeddings = []
        for vp in video_paths:
            emb = self.embedding_api.encode_video(vp)
            embeddings.append(emb)
        
        embeddings = torch.cat(embeddings, dim=0)  # (batch, dim)
        
        # Step 2: 分类头
        logits = self.classifier_head(embeddings)
        return logits
    
    def get_trainable_params(self):
        """返回可训练的参数（仅分类头）"""
        return self.classifier_head.parameters()
    
    def save_classifier(self, path: str):
        """仅保存分类头权重"""
        torch.save(self.classifier_head.state_dict(), path)
    
    def load_classifier(self, path: str):
        """加载分类头权重"""
        self.classifier_head.load_state_dict(torch.load(path, map_location="cpu"))


class OfflineClassifier(nn.Module):
    """离线模式: 仅分类头，embedding 预先提取好
    
    推荐使用此模式训练，因为 embedding 提取开销较大。
    先用 embedding_api 提取所有样本的 embedding 保存，再用此模型快速训练。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 128],
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.classifier_head = ClassifierHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_classes=num_classes,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) 预先提取的 embedding
            
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.classifier_head(x)
    
    def save(self, path: str):
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))


def test_classifier():
    """测试分类模型"""
    print("=" * 60)
    print("测试分类模型")
    print("=" * 60)
    
    # 测试 OfflineClassifier
    model = OfflineClassifier(input_dim=2048, hidden_dims=[512, 128], dropout=0.3)
    print(f"\n[OfflineClassifier] 结构:\n{model}")
    
    # 模拟输入
    dummy_input = torch.randn(4, 2048)  # batch=4
    logits = model(dummy_input)
    print(f"\n  输入: {dummy_input.shape}")
    print(f"  输出 logits: {logits.shape}")
    print(f"  预测类别: {logits.argmax(dim=-1)}")
    
    # 计算可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  可训练参数: {trainable:,}")
    print(f"  总参数: {total:,}")
    
    print("\n[Test] 通过!")


if __name__ == "__main__":
    test_classifier()
