"""
测试 qwen3-vl-embedding 输出维度
已验证: 从 modelscope 下载的 config + 源码分析

结论:
  - Qwen3-VL-Embedding-2B: embedding 维度 = 2048
  - Qwen3-VL-Embedding-8B: embedding 维度 = 4096

原理:
  模型使用 last_token_pooling:
  - 取 attention_mask 最后一个有效 token 位置的 hidden_state
  - 该 hidden_state 的维度 = text_config.hidden_size
  - 然后 L2 normalize

使用方法 (推荐用 Qwen3VLEmbedder):
  from qwen3_vl_embedding import Qwen3VLEmbedder
  embedder = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-8B")
  embeddings = embedder.process([
      {"video": "path/to/video.mp4", "instruction": "..."}
  ])
  # embeddings.shape = (batch_size, 4096)  for 8B
  # embeddings.shape = (batch_size, 2048)  for 2B
"""

import torch
import os

# ============================================================
# 验证: 从本地缓存读取 config 确认维度
# ============================================================
print("="*60)
print("从本地 modelscope 缓存确认维度")
print("="*60)

import json

for model_name in ["Qwen3-VL-Embedding-8B", "Qwen3-VL-Embedding-2B"]:
    config_path = os.path.expanduser(
        f"~/.cache/modelscope/hub/models/Qwen/{model_name}/config.json"
    )
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", {})
        vc = cfg.get("vision_config", {})
        print(f"\n{model_name}:")
        print(f"  text_config.hidden_size    = {tc.get('hidden_size')}  ← Embedding 输出维度")
        print(f"  text_config.num_layers     = {tc.get('num_hidden_layers')}")
        print(f"  vision_config.hidden_size  = {vc.get('hidden_size')}")
        print(f"  vision_config.out_hidden_size = {vc.get('out_hidden_size')}")
        print(f"  vision_config.patch_size   = {vc.get('patch_size')}")
        print(f"  vision_config.temporal_patch_size = {vc.get('temporal_patch_size')}")
    else:
        print(f"\n{model_name}: config 未缓存")

print("\n" + "="*60)
print("总结")
print("="*60)
print("""
┌────────────────────────────────┬──────────────┐
│ 模型                            │ Embedding维度 │
├────────────────────────────────┼──────────────┤
│ Qwen3-VL-Embedding-2B          │     2048     │
│ Qwen3-VL-Embedding-8B          │     4096     │
└────────────────────────────────┴──────────────┘

Pooling 方式: last_token_pooling (取最后一个有效token的hidden_state)
Normalize: L2 normalize (可选, 默认 True)
""")
