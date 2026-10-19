"""
静态图像：有无文字表述嵌入的 A/B 对比实验
============================================

设计 (严格对照实验):
  同一批静态图像 (LV-GLS / RV-GLS)
  ├── 实验组 A: embedding = encode_image_with_text(img, text=切面描述)   ← 有文字
  └── 对照组 B: embedding = encode_image(img)                            ← 无文字

  除"有无文字"外，其余完全一致:
    - 相同图片、相同标签
    - 相同数据划分 (固定 seed)
    - 相同分类头结构 / 超参 / 轮数
    - 相同早停策略

  对比指标: Accuracy / Precision / Recall / F1 / AUC

输出:
  ab_cmp_stats.pt          # 两组 embedding + 标签 + 划分
  ab_cmp_results.json      # 两组指标对比

使用:
  python ab_text_compare.py --mode collect     # 提取两组 embedding (需 API Key)
  python ab_text_compare.py --mode train       # 训练并对比
  python ab_text_compare.py --mode all         # 全流程
"""

import os
import re
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from embedding_api import QwenVLEmbeddingAPI, get_api_key
from classifier_model import OfflineClassifier
from dataset import load_view_descriptions, get_view_description, normalize_view

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# 配置
# ============================================================

class ABConfig:
    root_dir = os.path.dirname(os.path.abspath(__file__))
    label_map = {
        "HOCM玛法凯泰 影像": 1,
        "体检中心影像": 0,
    }
    # 只取静态图像
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    stats_path = "ab_cmp_stats.pt"
    result_path = "ab_cmp_results.json"

    # embedding (注意: 正确模型名为 qwen3-vl-embedding, 默认 2560 维;
    # 不存在 qwen3-vl-embedding-2b / -8b, 会返回 AccessDenied)
    model_name = "qwen3-vl-embedding"

    # 训练 (两组完全一致)
    hidden_dims = [256, 64]
    dropout = 0.3
    num_classes = 2
    epochs = 200
    batch_size = 8
    lr = 1e-3
    weight_decay = 1e-4
    train_ratio = 0.7
    seed = 42
    patience = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. 收集静态图像样本
# ============================================================

def collect_image_samples(cfg: ABConfig):
    """收集所有静态图像样本 (带切面文字描述)

    Returns:
        samples: [{path, label, view, desc, patient_id}, ...]
    """
    desc_map = load_view_descriptions()
    samples = []

    for folder_name, label in cfg.label_map.items():
        folder_path = os.path.join(cfg.root_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"[WARNING] 不存在: {folder_path}")
            continue

        for patient_dir in sorted(os.listdir(folder_path)):
            pp = os.path.join(folder_path, patient_dir)
            if not os.path.isdir(pp):
                continue

            m = re.match(r"(P\d+)_(V\d+)", patient_dir)
            patient_id = m.group(0) if m else patient_dir

            for fname in sorted(os.listdir(pp)):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in cfg.image_exts:
                    continue

                # 提取切面: 去掉 "P001_V1-" 前缀
                stem = os.path.splitext(fname)[0]
                view_raw = re.sub(r"^P\d+_V\d+-", "", stem)
                view = normalize_view(view_raw)
                desc = get_view_description(view_raw, desc_map)

                if not desc:
                    # 无对应切面描述，跳过 (保证 A 组的 text 有实际内容)
                    continue

                samples.append({
                    "path": os.path.join(pp, fname),
                    "label": label,
                    "view": view,
                    "desc": desc,
                    "patient_id": patient_id,
                })

    pos = sum(1 for s in samples if s["label"] == 1)
    neg = sum(1 for s in samples if s["label"] == 0)
    views = {}
    for s in samples:
        views[s["view"]] = views.get(s["view"], 0) + 1

    print(f"\n[收集] 静态图像样本: {len(samples)}")
    print(f"  阳性(HOCM): {pos}, 阴性(体检): {neg}")
    print(f"  切面分布: {views}")

    return samples


# ============================================================
# 2. 提取两组 embedding
# ============================================================

def extract_embeddings(cfg: ABConfig, samples, api: QwenVLEmbeddingAPI):
    """提取两组 embedding: A=有文字, B=无文字 (支持增量缓存/断点续跑)"""

    partial_path = cfg.stats_path + ".partial"
    done = {}   # path -> (embA, embB, label, sample)
    if os.path.exists(partial_path):
        try:
            prev = torch.load(partial_path, map_location="cpu", weights_only=False)
            for s, ea, eb in zip(prev["samples"], prev["A"], prev["B"]):
                done[s["path"]] = (ea, eb, s["label"], s)
            print(f"[续跑] 已加载 {len(done)} 个已完成样本 (来自 {partial_path})")
        except Exception as e:
            print(f"[续跑] 缓存读取失败, 重新开始: {e}")
            done = {}

    emb_A, emb_B, labels, valid = [], [], [], []

    print(f"\n[提取] 共 {len(samples)} 张图像 (每组需要 {len(samples)} 次 API 调用)...")
    import time
    t0 = time.time()

    for i, s in enumerate(samples):
        p = s["path"]

        if p in done:
            ea, eb, lab, _ = done[p]
            ea = torch.as_tensor(ea).float()
            eb = torch.as_tensor(eb).float()
        else:
            try:
                # A 组: 图片 + 文字 (融合)
                ea = torch.as_tensor(api.encode_image_with_text(p, text=s["desc"]).squeeze(0)).float()
                # B 组: 仅图片 (融合模式, 无文字)
                eb = torch.as_tensor(api.encode_image(p).squeeze(0)).float()
                lab = s["label"]
            except Exception as e:
                print(f"  [WARNING] {os.path.basename(p)} 失败: {e}")
                continue

        emb_A.append(ea)
        emb_B.append(eb)
        labels.append(lab)
        valid.append(s)

        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 0
            eta = (len(samples) - i - 1) / rate if rate > 0 else 0
            print(f"  进度 {i+1}/{len(samples)}  "
                  f"({el:.0f}s, {rate:.2f}/s, ETA {eta:.0f}s)")

        # 每 25 个存一次断点
        if (i + 1) % 25 == 0:
            torch.save({
                "A": torch.stack(emb_A), "B": torch.stack(emb_B),
                "samples": valid,
            }, partial_path)

    A = torch.stack(emb_A).float()
    B = torch.stack(emb_B).float()
    y = torch.tensor(labels, dtype=torch.long)

    # 清理断点文件
    if os.path.exists(partial_path):
        try:
            os.remove(partial_path)
        except Exception:
            pass

    print(f"\n[提取完成]")
    print(f"  A 组 (有文字): {tuple(A.shape)}")
    print(f"  B 组 (无文字): {tuple(B.shape)}")
    print(f"  标签: 0={int((y==0).sum())}, 1={int((y==1).sum())}")
    print(f"  API 统计: {api.stats}")

    torch.save({
        "A_with_text": A,
        "B_no_text": B,
        "labels": y,
        "samples": valid,
        "dims": {"A": A.shape[1], "B": B.shape[1]},
    }, cfg.stats_path)
    print(f"[保存] {cfg.stats_path}")

    return A, B, y, valid


# ============================================================
# 3. 一致性检查: 确认两组 embedding 确实不同
# ============================================================

def sanity_check(A, B):
    """检查 A/B 两组 embedding 是否真的不同 (证明文字生效)"""
    An = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    Bn = B / (B.norm(dim=1, keepdim=True) + 1e-8)
    cos = (An * Bn).sum(dim=1)

    print(f"\n{'='*60}")
    print("一致性检查: A(有文字) vs B(无文字) 的余弦相似度")
    print(f"{'='*60}")
    print(f"  均值: {cos.mean().item():.6f}")
    print(f"  最小: {cos.min().item():.6f}")
    print(f"  最大: {cos.max().item():.6f}")
    identical = int((cos > 0.99999).sum())
    print(f"  完全相同的样本数: {identical} / {len(cos)}")

    if cos.mean().item() > 0.9999:
        print("  [警告] 两组几乎相同 → 文字可能未生效!")
    else:
        print("  [OK] 两组存在差异 → 文字确实影响了 embedding")

    return float(cos.mean().item()), identical


# ============================================================
# 4. 训练 + 评估 (两组使用完全相同的流程)
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            all_logits.append(model(x).cpu())
            all_labels.append(y.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    preds = logits.argmax(dim=-1).numpy()

    if HAS_SKLEARN:
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc": auc,
        }
    else:
        acc = float((preds == labels).mean())
        return {"accuracy": acc, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}


def train_one_group(name, X, y, cfg: ABConfig, verbose=True):
    """训练一组 embedding，返回 (best_val_metrics, best_state)"""
    set_seed(cfg.seed)

    n = len(X)
    n_train = int(n * cfg.train_ratio)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed))
    tr_idx, va_idx = perm[:n_train], perm[n_train:]

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=cfg.batch_size, shuffle=False)

    model = OfflineClassifier(
        input_dim=X.shape[1],
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best = {"loss": float("inf")}
    best_state = None
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        tot, vlabels, vpreds = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                logits = model(xb)
                tot += criterion(logits, yb).item()
                vpreds.append(logits.argmax(-1).cpu())
                vlabels.append(yb.cpu())
        vloss = tot / len(val_loader)
        scheduler.step(vloss)

        if vloss < best["loss"]:
            best["loss"] = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    # 用最佳权重评估
    model.load_state_dict(best_state)
    metrics = evaluate(model, val_loader, cfg.device)
    metrics["best_val_loss"] = best["loss"]
    metrics["epochs_run"] = epoch
    metrics["n_train"] = n_train
    metrics["n_val"] = n - n_train

    if verbose:
        print(f"  [{name}] acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
              f"auc={metrics['auc']:.4f} (epochs={epoch})")

    return metrics, best_state


def run_comparison(A, B, y, cfg: ABConfig):
    """两组使用完全相同的流程训练并对比"""
    print(f"\n{'='*60}")
    print(f"训练对比 (device={cfg.device}, seed={cfg.seed})")
    print(f"{'='*60}")

    # 为保证公平: 数据划分基于相同的 seed，且两组样本顺序一致
    m_A, st_A = train_one_group("A-有文字", A, y, cfg)
    m_B, st_B = train_one_group("B-无文字", B, y, cfg)

    # 保存两个分类头
    torch.save(st_A, "ab_cmp_head_with_text.pt")
    torch.save(st_B, "ab_cmp_head_no_text.pt")

    def diff(a, b):
        return a - b

    print(f"\n{'='*60}")
    print("对比结果 (A=有文字, B=无文字)")
    print(f"{'='*60}")
    print(f"{'指标':<12}{'A 有文字':>12}{'B 无文字':>12}{'Δ(A-B)':>12}")
    print("-" * 48)
    for k in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"{k:<12}{m_A[k]:>12.4f}{m_B[k]:>12.4f}{diff(m_A[k], m_B[k]):>+12.4f}")
    print("-" * 48)
    print(f"{'样本数(A/B)':<12}{m_A['n_train']}/{m_A['n_val']:<12}\n")

    result = {
        "config": {
            "model_name": cfg.model_name,
            "hidden_dims": cfg.hidden_dims,
            "dropout": cfg.dropout,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "train_ratio": cfg.train_ratio,
            "seed": cfg.seed,
            "embedding_dim": int(A.shape[1]),
        },
        "A_with_text": m_A,
        "B_no_text": m_B,
        "delta_A_minus_B": {k: diff(m_A[k], m_B[k]) for k in ["accuracy", "precision", "recall", "f1", "auc"]},
    }

    with open(cfg.result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] 对比结果: {cfg.result_path}")

    return result


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="静态图像 有无文字嵌入 A/B 对比实验")
    parser.add_argument("--mode", choices=["collect", "train", "all"], default="all")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default=ABConfig.model_name)
    args = parser.parse_args()

    cfg = ABConfig()
    cfg.model_name = args.model

    if args.mode in ("collect", "all"):
        api_key = get_api_key(args.api_key)
        if not api_key:
            print("[ERROR] 需要 API Key (--api_key 或 DASHSCOPE_API_KEY 或 .api_key)")
            return

        samples = collect_image_samples(cfg)
        if not samples:
            print("[ERROR] 未收集到图像样本")
            return

        api = QwenVLEmbeddingAPI(api_key=api_key, model_name=cfg.model_name)
        A, B, y, valid = extract_embeddings(cfg, samples, api)
        sanity_check(A, B)

    if args.mode in ("train", "all"):
        if not os.path.exists(cfg.stats_path):
            print(f"[ERROR] 找不到 {cfg.stats_path}，请先运行 --mode collect")
            return
        d = torch.load(cfg.stats_path, map_location="cpu", weights_only=False)
        A, B, y = d["A_with_text"], d["B_no_text"], d["labels"]
        run_comparison(A, B, y, cfg)


if __name__ == "__main__":
    main()
