"""
训练脚本

流程:
  1. 收集所有样本
  2. 预提取 embedding（或加载缓存）
  3. 训练二分类全连接层
  4. 评估 & 保存模型

使用方式:
  # 仅收集数据
  python train.py --mode collect

  # 预提取 embedding  
  python train.py --mode precompute

  # 训练
  python train.py --mode train

  # 一键全流程
  python train.py --mode all
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 导入自定义模块
from dataset import collect_samples, precompute_embeddings, create_dataloaders
from classifier_model import OfflineClassifier
from embedding_api import QwenVLEmbeddingAPI


# ============================================================
# 配置
# ============================================================

class Config:
    # 数据
    root_dir = os.path.dirname(__file__)
    label_map = {
        "HOCM玛法凯泰 影像": 1,
        "体检中心影像": 0,
    }
    cache_path = "embeddings_cache.pt"
    
    # DashScope API
    api_key = None  # 从环境变量 DASHSCOPE_API_KEY 或 .api_key 文件读取
    model_name = "qwen3-vl-embedding"  # 2560 维
    embedding_dim = 2560
    hidden_dims = [512, 128]
    dropout = 0.3
    num_classes = 2
    
    # 训练
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 50
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-4
    train_ratio = 0.8
    seed = 42
    patience = 10
    
    # embedding 提取参数 (API 调用)
    max_frames = 16
    fps = 2.0
    api_delay = 0.5

    # OSS 配置 (URL 上传模式)
    oss_bucket = "url11"
    oss_endpoint = "oss-cn-beijing.aliyuncs.com"
    oss_key_id = "LTAI5t798VpZRbj4dnLEA1hD"
    oss_key_secret = "f1TlQAmUUmU9jY5Th0VlU5eHE8emp6"
    
    # 保存
    model_save_path = "classifier_head.pt"


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 训练
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for embeddings, labels in tqdm(loader, desc="Training", leave=False):
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for embeddings, labels in tqdm(loader, desc="Evaluating", leave=False):
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        logits = model(embeddings)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # 正类概率
    
    avg_loss = total_loss / len(loader)
    
    # 计算指标
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "cm": confusion_matrix(all_labels, all_preds).tolist(),
    }
    
    # AUC (需要至少2个类别都有样本)
    if len(set(all_labels)) > 1:
        try:
            metrics["auc"] = roc_auc_score(all_labels, all_probs)
        except:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0
    
    return metrics


def train(config: Config):
    """主训练流程"""
    set_seed(config.seed)
    print(f"\n设备: {config.device}")
    print(f"Embedding 维度: {config.embedding_dim}")
    
    # 1. 加载数据
    if not os.path.exists(config.cache_path):
        print(f"\n[ERROR] 缓存文件不存在: {config.cache_path}")
        print("请先运行: python train.py --mode precompute")
        return
    
    data = torch.load(config.cache_path, map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]
    labels = data["labels"]
    print(f"\n加载数据: {embeddings.shape}, 标签: 0={sum(labels==0)}, 1={sum(labels==1)}")
    
    # 2. 创建 DataLoader
    train_loader, val_loader = create_dataloaders(
        embeddings, labels,
        train_ratio=config.train_ratio,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    
    # 3. 创建模型
    model = OfflineClassifier(
        input_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        num_classes=config.num_classes,
    ).to(config.device)
    
    print(f"\n模型结构:\n{model}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable:,}")
    
    # 4. 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 5. 训练循环
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print(f"开始训练 (epochs={config.epochs}, lr={config.learning_rate})")
    print(f"{'='*60}")
    
    for epoch in range(1, config.epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, config.device
        )
        
        # 验证
        val_metrics = eval_epoch(model, val_loader, criterion, config.device)
        
        # 学习率调整
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印
        print(f"\nEpoch {epoch:3d}/{config.epochs} | "
              f"LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f}")
        
        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            model.save(config.model_save_path)
            print(f"  >>> 保存最佳模型 (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # 6. 最终评估
    print(f"\n{'='*60}")
    print(f"训练完成! 最佳 epoch: {best_epoch}, val_loss: {best_val_loss:.4f}")
    print(f"模型已保存: {config.model_save_path}")
    
    # 加载最佳模型并最终评估
    best_model = OfflineClassifier(
        input_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        num_classes=config.num_classes,
    ).to(config.device)
    best_model.load(config.model_save_path)
    
    final_metrics = eval_epoch(best_model, val_loader, criterion, config.device)
    print(f"\n最终验证集指标:")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1:        {final_metrics['f1']:.4f}")
    print(f"  AUC:       {final_metrics['auc']:.4f}")
    print(f"  Confusion Matrix:\n{np.array(final_metrics['cm'])}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-Embedding + 二分类训练")
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["collect", "precompute", "train", "all"],
        help="运行模式: collect(收集数据), precompute(预提取embedding), train(训练), all(全流程)"
    )
    parser.add_argument("--device", type=str, default=None, help="设备 (cpu/cuda)")
    parser.add_argument("--api_key", type=str, default=None, help="DashScope API Key")
    args = parser.parse_args()
    
    config = Config()
    if args.device:
        config.device = args.device
    
    # 加载 API Key
    from embedding_api import get_api_key as _get_key
    config.api_key = _get_key(args.api_key)
    if not config.api_key and args.mode in ("precompute", "all"):
        print("[ERROR] 请提供 API Key: --api_key sk-xxx 或设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)
    
    root_dir = config.root_dir
    
    # === Mode: collect ===
    if args.mode in ("collect", "all"):
        print("=" * 60)
        print("Step 1: 收集数据")
        print("=" * 60)
        samples = collect_samples(root_dir, config.label_map)
    
    # === Mode: precompute ===
    if args.mode in ("precompute", "all"):
        print("\n" + "=" * 60)
        print("Step 2: 预提取 Embedding")
        print("=" * 60)
        
        # 如果还没收集过样本，先收集
        if args.mode == "precompute":
            samples = collect_samples(root_dir, config.label_map)
        
        api = QwenVLEmbeddingAPI(
            api_key=config.api_key,
            model_name=config.model_name,
            max_frames=config.max_frames,
            fps=config.fps,
            oss_bucket=config.oss_bucket,
            oss_endpoint=config.oss_endpoint,
            oss_key_id=config.oss_key_id,
            oss_key_secret=config.oss_key_secret,
        )
        
        embeddings, labels, samples = precompute_embeddings(
            samples, api, cache_path=config.cache_path
        )
    
    # === Mode: train ===
    if args.mode in ("train", "all"):
        print("\n" + "=" * 60)
        print("Step 3: 训练二分类模型")
        print("=" * 60)
        train(config)


if __name__ == "__main__":
    main()
