"""
数据加载器

数据结构:
  视频编码器/
  ├── HOCM玛法凯泰 影像/       ← 阳性 (label=1)
  │   ├── 患者A-日期-编号/
  │   │   ├── 二腔.mp4
  │   │   ├── 三腔.mp4
  │   │   ├── 四腔.mp4
  │   │   ├── LV-GLS.jpg
  │   │   └── RV-GLS.jpg
  │   └── ...
  └── 体检中心影像/             ← 阴性 (label=0)
      └── ...

支持:
  1. 遍历文件夹，收集所有视频/图片路径和标签
  2. 预提取 embedding 并缓存到 .pt 文件（加速训练）
  3. PyTorch DataLoader 集成
"""

import os
import re
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm


# ============================================================
# 0. 加载文字表述映射 (view → 描述文本)
# ============================================================

def load_view_descriptions(txt_path: str = None) -> Dict[str, str]:
    """从 文字表述.txt 加载切面名称 → 文字描述的映射
    
    文件格式:
      二腔：这是放大的左室心尖二腔心切面的动态图
      三腔：这是放大的左室心尖三腔心切面的动态图
      ...
    
    Returns:
        {"二腔": "这是放大的...", "三腔": "这是放大的...", ...}
    """
    if txt_path is None:
        txt_path = os.path.join(os.path.dirname(__file__), "文字表述.txt")
    
    mapping = {}
    if not os.path.exists(txt_path):
        print(f"[WARNING] 文字表述文件不存在: {txt_path}")
        return mapping
    
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: "二腔：这是放大的左室心尖二腔心切面的动态图"
            # 或:   "LV GLS：这是左室心肌纵向应变图"
            if "：" in line:
                key, value = line.split("：", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue
            key = key.strip()
            value = value.strip()
            mapping[key] = value
    
    print(f"[ViewDesc] 加载了 {len(mapping)} 条切面描述:")
    for k, v in mapping.items():
        print(f"  {k} → {v}")
    
    return mapping


# 切面名称归一化映射 (文件名中的 view → 文字表述中的 key)
VIEW_NORMALIZE = {
    "二腔": "二腔",
    "三腔": "三腔", 
    "四腔": "四腔",
    "LV-GLS": "LV GLS",
    "LV-GLS2": "LV GLS",
    "LV-GLS-1": "LV GLS",
    "LV GLS": "LV GLS",
    "LV-GLS 2": "LV GLS",
    "RV-GLS": "RV GLS",
    "RV GLS": "RV GLS",
}


def normalize_view(view: str) -> str:
    """将文件名提取的 view 归一化到文字表述中的 key"""
    # 先尝试精确匹配
    if view in VIEW_NORMALIZE:
        return VIEW_NORMALIZE[view]
    # 尝试模糊匹配
    for k, v in VIEW_NORMALIZE.items():
        if k in view or view in k:
            return v
    return view


def get_view_description(view: str, desc_map: Dict[str, str]) -> str:
    """根据切面名称获取文字描述"""
    normalized = normalize_view(view)
    return desc_map.get(normalized, "")


# ============================================================
# 1. 数据收集: 遍历文件夹，整理路径和标签
# ============================================================

def collect_samples(
    root_dir: str,
    label_map: Dict[str, int] = None,
    desc_txt_path: str = None,
) -> List[Dict]:
    """遍历根目录，收集所有样本
    
    Args:
        root_dir: 根目录路径
        label_map: 子文件夹名 → label 的映射，
                   默认: {"HOCM玛法凯泰 影像": 1, "体检中心影像": 0}
        desc_txt_path: 文字表述文件路径
    
    Returns:
        samples: 列表，每个元素:
            {
                "video_path": str,         # 视频路径
                "image_paths": [str],      # 图片路径列表
                "patient_name": str,       # 患者名
                "patient_id": str,         # 编号
                "label": int,              # 0 或 1
                "view": str,               # 切面名称 (二腔/三腔/四腔/LV-GLS/RV-GLS)
                "view_description": str,   # 对应的文字描述
                "media_type": str,         # "video" 或 "image"
            }
    """
    # 加载文字表述
    desc_map = load_view_descriptions(desc_txt_path)
    if label_map is None:
        label_map = {
            "HOCM玛法凯泰 影像": 1,
            "体检中心影像": 0,
        }
    
    samples = []
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for folder_name, label in label_map.items():
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"[WARNING] 文件夹不存在: {folder_path}")
            continue
        
        print(f"\n扫描: {folder_name} (label={label})")
        
        # 遍历患者文件夹
        for patient_dir in os.listdir(folder_path):
            patient_path = os.path.join(folder_path, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            
            # 解析患者名和编号: "陈桃花-20251212-60014895196"
            parts = patient_dir.rsplit('-', 2)
            patient_name = parts[0] if len(parts) >= 1 else patient_dir
            patient_id = parts[-1] if len(parts) >= 2 else ""
            
            # 收集该患者的所有文件
            video_files = []
            image_files = []
            
            for fname in os.listdir(patient_path):
                ext = os.path.splitext(fname)[1].lower()
                full_path = os.path.join(patient_path, fname)
                
                if ext in video_extensions:
                    # 提取切面名称: "陈桃花-20251212-60014895196-二腔.mp4" → "二腔"
                    view = _extract_view(fname, patient_dir)
                    video_files.append((full_path, view))
                
                elif ext in image_extensions:
                    view = _extract_view(fname, patient_dir)
                    image_files.append((full_path, view))
            
            # 为每个视频创建一个样本
            for vpath, view in video_files:
                view_desc = get_view_description(view, desc_map)
                samples.append({
                    "video_path": vpath,
                    "image_paths": [ip for ip, _ in image_files],
                    "patient_name": patient_name,
                    "patient_id": patient_id,
                    "label": label,
                    "view": view,
                    "view_description": view_desc,
                    "media_type": "video",
                })
            
            # 为每张图片也创建样本
            for ipath, view in image_files:
                view_desc = get_view_description(view, desc_map)
                samples.append({
                    "video_path": None,
                    "image_paths": [ipath],
                    "patient_name": patient_name,
                    "patient_id": patient_id,
                    "label": label,
                    "view": view,
                    "view_description": view_desc,
                    "media_type": "image",
                })
    
    print(f"\n总计收集 {len(samples)} 个样本")
    
    # 统计
    pos_count = sum(1 for s in samples if s["label"] == 1)
    neg_count = sum(1 for s in samples if s["label"] == 0)
    video_count = sum(1 for s in samples if s["media_type"] == "video")
    image_count = sum(1 for s in samples if s["media_type"] == "image")
    print(f"  阳性(HOCM): {pos_count}, 阴性(体检): {neg_count}")
    print(f"  视频: {video_count}, 图片: {image_count}")
    
    return samples


def _extract_view(filename: str, patient_dir: str) -> str:
    """从文件名提取切面名称
    
    示例: 
      "陈桃花-20251212-60014895196-二腔.mp4"           → "二腔"
      "陈桃花-20251212-60014895196-LV-GLS.jpg"         → "LV-GLS"
      "陈海涛-20260611-2506729-LV-GLS2.jpg"            → "LV-GLS2"
      "陈海涛-20260611-2506729-二腔.avi"               → "二腔"
    """
    import re
    name = os.path.splitext(filename)[0]
    # 尝试去掉患者名-日期-编号前缀
    # 模式: 中文名-8位日期-数字编号-
    # 也可能是: 中文名-7位日期-数字编号-
    name = re.sub(r'^[\u4e00-\u9fa5]+-\d{7,8}-\d+[-=]?', '', name)
    return name


# ============================================================
# 2. Embedding 缓存数据集 (推荐: 离线模式)
# ============================================================

class EmbeddingDataset(Dataset):
    """预先提取好的 embedding 数据集
    
    先用 embedding_api 提取所有样本的 embedding，缓存为 .pt 文件，
    然后用此 Dataset 快速加载训练。
    """
    
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: (N, embedding_dim) 的预提取 embedding
            labels: (N,) 的标签
        """
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def precompute_embeddings(
    samples: List[Dict],
    embedding_api,
    cache_path: str = "embeddings_cache.pt",
    batch_size: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """预计算所有样本的 embedding 并缓存
    
    Args:
        samples: collect_samples 返回的样本列表
        embedding_api: QwenVLEmbeddingAPI 实例 (DashScope 远程版)
        cache_path: 缓存文件路径
    
    Returns:
        embeddings: (N, dim) torch.Tensor
        labels: (N,) torch.Tensor  
        samples: 更新后的样本列表
    """
    if os.path.exists(cache_path):
        print(f"[Cache] 加载缓存: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return data["embeddings"], data["labels"], data.get("samples", samples)
    
    print(f"[Precompute] 开始提取 {len(samples)} 个样本的 embedding...")
    
    embeddings_list = []
    labels_list = []
    valid_samples = []
    
    for i, sample in enumerate(tqdm(samples, desc="提取 embedding")):
        try:
            # 使用文字描述与视觉内容同时输入
            view_desc = sample.get("view_description", "")
            
            if sample["media_type"] == "video" and sample["video_path"]:
                emb = embedding_api.encode_video_with_text(
                    video_path=sample["video_path"],
                    text=view_desc,
                )
            elif sample["media_type"] == "image" and sample["image_paths"]:
                emb = embedding_api.encode_image_with_text(
                    image_path=sample["image_paths"][0],
                    text=view_desc,
                )
            else:
                continue
            
            embeddings_list.append(emb.squeeze(0))  # (dim,) numpy array
            labels_list.append(sample["label"])
            valid_samples.append(sample)
            
        except Exception as e:
            print(f"\n[WARNING] 样本 {i} 提取失败: {e}")
            continue
    
    embeddings = torch.from_numpy(np.stack(embeddings_list))  # (N, dim)
    labels = torch.tensor(labels_list, dtype=torch.long)
    
    # 保存缓存
    torch.save({
        "embeddings": embeddings,
        "labels": labels,
        "samples": valid_samples,
    }, cache_path)
    
    print(f"[Cache] 已保存 {len(valid_samples)} 个 embedding 到 {cache_path}")
    print(f"  embedding shape: {embeddings.shape}")
    print(f"  标签分布: 0={sum(labels==0)}, 1={sum(labels==1)}")
    
    return embeddings, labels, valid_samples


# ============================================================
# 3. DataLoader 工厂
# ============================================================

def create_dataloaders(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_ratio: float = 0.8,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练/验证 DataLoader
    
    Args:
        embeddings: (N, dim)
        labels: (N,)
        train_ratio: 训练集比例
        batch_size: 批量大小
        seed: 随机种子
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import random_split
    
    dataset = EmbeddingDataset(embeddings, labels)
    
    # 划分
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n数据集划分:")
    print(f"  训练集: {train_size} 样本")
    print(f"  验证集: {val_size} 样本")
    
    return train_loader, val_loader


# ============================================================
# 4. 患者级聚合：5切面拼接
# ============================================================

def collect_patients(root_dir: str, compressed_dir: str = None) -> List[Dict]:
    """按患者收集，每个患者聚合 5 个切面
    
    切面顺序（固定）: 二腔(video) → 三腔(video) → 四腔(video) → LV-GLS(image) → RV-GLS(image)
    取每个切面的第一个匹配文件，忽略多余的（如 LV-GLS-1）
    
    Returns:
        patients: [{patient_id, label, views: [{type, path, view, desc}]}, ...]
    """
    if compressed_dir is None:
        compressed_dir = os.path.join(root_dir, "压缩文件夹")
    
    label_map = {"HOCM玛法凯泰 影像": 1, "体检中心影像": 0}
    desc_map = load_view_descriptions()
    
    # 固定切面顺序
    VIEW_ORDER = [
        {"view_key": "二腔", "type": "video", "ext": [".mp4"]},
        {"view_key": "三腔", "type": "video", "ext": [".mp4"]},
        {"view_key": "四腔", "type": "video", "ext": [".mp4"]},
        {"view_key": "LV-GLS", "type": "image", "ext": [".jpg", ".jpeg", ".png"]},
        {"view_key": "RV-GLS", "type": "image", "ext": [".jpg", ".jpeg", ".png"]},
    ]
    
    patients = []
    
    for folder_name, label in label_map.items():
        folder_path = os.path.join(compressed_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"[WARNING] 文件夹不存在: {folder_path}")
            continue
        
        for patient_dir in sorted(os.listdir(folder_path)):
            pp = os.path.join(folder_path, patient_dir)
            if not os.path.isdir(pp):
                continue
            
            # 提取匿名ID
            import re
            match = re.match(r'(P\d+)_(V\d+)', patient_dir)
            patient_id = match.group(0) if match else patient_dir
            
            views = []
            for vdef in VIEW_ORDER:
                found = None
                for ext in vdef["ext"]:
                    for f in sorted(os.listdir(pp)):
                        if f.endswith(ext) and vdef["view_key"] in f and '-1' not in f:
                            found = f
                            break
                    if found:
                        break
                if found:
                    view_name = _extract_view(found, patient_dir)
                    view_desc = get_view_description(view_name, desc_map)
                    views.append({
                        "type": vdef["type"],
                        "path": os.path.join(pp, found),
                        "view": view_name,
                        "desc": view_desc,
                    })
            
            if len(views) == 5:
                patients.append({
                    "patient_id": patient_id,
                    "label": label,
                    "views": views,
                })
    
    pos = sum(1 for p in patients if p["label"] == 1)
    neg = sum(1 for p in patients if p["label"] == 0)
    print(f"\n患者级收集: {len(patients)} 患者 (HOCM: {pos}, Normal: {neg})")
    return patients


def precompute_patient_embeddings(
    patients: List[Dict],
    embedding_api,
    cache_path: str = "patient_embeddings_cache.pt",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对每个患者的 5 个切面分别编码，拼接为 (5*dim) 维向量
    
    Returns:
        embeddings: (N, 5*dim) 拼接后的患者级 embedding
        labels: (N,)
    """
    if os.path.exists(cache_path):
        print(f"[Cache] 加载患者缓存: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return data["embeddings"], data["labels"]
    
    embeddings_list = []
    labels_list = []
    fail_count = 0
    
    for patient in tqdm(patients, desc="患者级编码"):
        try:
            view_embs = []
            for view in patient["views"]:
                if view["type"] == "video":
                    emb = embedding_api.encode_video_with_text(
                        view["path"], text=view["desc"]
                    )
                else:
                    emb = embedding_api.encode_image_with_text(
                        view["path"], text=view["desc"]
                    )
                view_embs.append(torch.from_numpy(emb.squeeze(0)))
            
            # 拼接 5 个切面: (5*dim,)
            concat_emb = torch.cat(view_embs, dim=0)
            embeddings_list.append(concat_emb)
            labels_list.append(patient["label"])
        except Exception as e:
            print(f"\n[WARNING] {patient['patient_id']} 失败: {e}")
            fail_count += 1
    
    embeddings = torch.stack(embeddings_list)
    labels = torch.tensor(labels_list, dtype=torch.long)
    
    torch.save({"embeddings": embeddings, "labels": labels}, cache_path)
    print(f"[Cache] 患者级: {len(embeddings_list)} 患者, dim={embeddings.shape[1]}, 失败{fail_count}")
    return embeddings, labels


if __name__ == "__main__":
    test_dataloader()
