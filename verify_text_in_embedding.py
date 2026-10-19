"""
验证: 文字描述是否真的被输入到了 embedding 向量中

方法: 对同一个视频/图片，分别做
  A) 只用视觉 (无文字)
  B) 视觉 + 文字描述
比较两者 embedding 的余弦相似度。
  - 如果相似度 == 1.0000  → 文字没有生效 (被忽略)
  - 如果相似度 < 1.0      → 文字确实参与了编码
"""

import os
import numpy as np
from embedding_api import QwenVLEmbeddingAPI, get_api_key

IMG = os.path.join(
    "HOCM玛法凯泰 影像", "P001_V1-20251217-60016610605", "P001_V1-LV-GLS.jpg"
)
VID = os.path.join(
    "HOCM玛法凯泰 影像", "P001_V1-20251217-60016610605", "P001_V1-二腔.mp4"
)

api = QwenVLEmbeddingAPI(api_key=get_api_key(), model_name="qwen3-vl-embedding")

def cos(a, b):
    a = a.ravel(); b = b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

print("\n" + "=" * 60)
print("测试 1: 图片  无文字 vs 有文字")
print("=" * 60)
e1 = api.encode_image(IMG)                                  # 无文字
e2 = api.encode_image_with_text(IMG, text="这是左室心肌纵向应变图")  # 有文字
print(f"  无文字 shape: {e1.shape}")
print(f"  有文字 shape: {e2.shape}")
print(f"  余弦相似度: {cos(e1, e2):.6f}")
print(f"  → 文字{'有' if cos(e1, e2) < 0.99999 else '没有'}生效")

print("\n" + "=" * 60)
print("测试 2: 同一图片 + 两种不同文字")
print("=" * 60)
eA = api.encode_image_with_text(IMG, text="这是左室心肌纵向应变图")
eB = api.encode_image_with_text(IMG, text="这是右室心肌纵向应变图")
print(f"  余弦相似度: {cos(eA, eB):.6f}")
print(f"  → 不同文字{'产生' if cos(eA, eB) < 0.99999 else '没有产生'}不同 embedding")

print("\n" + "=" * 60)
print("测试 3: 视频  无文字 vs 有文字")
print("=" * 60)
v1 = api.encode_video_with_text(VID, text="")
v2 = api.encode_video_with_text(VID, text="这是放大的左室心尖二腔心切面的动态图")
print(f"  无文字 shape: {v1.shape}")
print(f"  有文字 shape: {v2.shape}")
print(f"  余弦相似度: {cos(v1, v2):.6f}")
print(f"  → 文字{'有' if cos(v1, v2) < 0.99999 else '没有'}生效")

print("\n" + "=" * 60)
print("测试 4: 纯文字 与 纯文字")
print("=" * 60)
t1 = api.encode_text("这是放大的左室心尖二腔心切面的动态图")
t2 = api.encode_text("今天天气很好，适合出门散步")
print(f"  余弦相似度: {cos(t1, t2):.6f}")
print(f"  → 纯文本接口{'正常' if cos(t1, t2) < 0.99999 else '异常'}")

print("\n统计:", api.stats)
