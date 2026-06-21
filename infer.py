"""
推理脚本

加载训练好的分类头，对新视频/图片进行预测。

使用方式:
  python infer.py --video "path/to/video.mp4"
  python infer.py --image "path/to/image.jpg"
  python infer.py --dir "path/to/patient_folder/"
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict

from embedding_api import QwenVLEmbeddingAPI
from classifier_model import OfflineClassifier
from dataset import collect_samples, load_view_descriptions, normalize_view, get_view_description


def load_model(
    api_key: str,
    classifier_path: str = "classifier_head.pt",
    model_name: str = "qwen3-vl-embedding",  # 2560维
    embedding_dim: int = 2560,
    hidden_dims: list = [512, 128],
    device: str = "cpu",
):
    """加载 embedding API + 分类头"""
    
    print(f"[Infer] 初始化 DashScope API ({model_name})...")
    api = QwenVLEmbeddingAPI(
        api_key=api_key,
        model_name=model_name,
        max_frames=16,
    )
    
    print(f"[Infer] 加载分类头 ({classifier_path})...")
    classifier = OfflineClassifier(
        input_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout=0.3,
        num_classes=2,
    )
    classifier.load_state_dict(
        torch.load(classifier_path, map_location=device, weights_only=True)
    )
    classifier.to(device)
    classifier.eval()
    
    print("[Infer] 模型加载完成!")
    return api, classifier


def predict_video(
    api, classifier, video_path: str, 
    view_text: str = "", 
    device: str = "cpu"
) -> Dict:
    """预测单个视频（可选附带文字描述）"""
    with torch.no_grad():
        if view_text:
            embedding = api.encode_video_with_text(video_path, text=view_text)
        else:
            embedding = api.encode_video(video_path)
        embedding = embedding.to(device)
        logits = classifier(embedding)
        probs = F.softmax(logits, dim=-1)
    
    pred = logits.argmax(dim=-1).item()
    prob = probs[0, pred].item()
    
    return {
        "prediction": pred,
        "label": "HOCM阳性" if pred == 1 else "体检阴性",
        "confidence": prob,
        "probabilities": probs[0].cpu().tolist(),
    }


def predict_image(
    api, classifier, image_path: str, 
    view_text: str = "", 
    device: str = "cpu"
) -> Dict:
    """预测单张图片（可选附带文字描述）"""
    with torch.no_grad():
        if view_text:
            embedding = api.encode_image_with_text(image_path, text=view_text)
        else:
            embedding = api.encode_image(image_path)
        embedding = embedding.to(device)
        logits = classifier(embedding)
        probs = F.softmax(logits, dim=-1)
    
    pred = logits.argmax(dim=-1).item()
    prob = probs[0, pred].item()
    
    return {
        "prediction": pred,
        "label": "HOCM阳性" if pred == 1 else "体检阴性",
        "confidence": prob,
        "probabilities": probs[0].cpu().tolist(),
    }


def predict_patient(
    api, classifier, patient_dir: str, 
    desc_map: Dict[str, str] = None,
    device: str = "cpu"
) -> Dict:
    """预测一个患者文件夹中的所有视频和图片，综合投票
    
    会自动从文件名提取 view 并匹配文字描述。
    """
    from dataset import _extract_view, get_view_description
    
    if desc_map is None:
        desc_map = load_view_descriptions()
    
    results = []
    
    for fname in sorted(os.listdir(patient_dir)):
        fpath = os.path.join(patient_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        
        # 提取切面并获取文字描述
        view = _extract_view(fname, os.path.basename(patient_dir))
        view_text = get_view_description(view, desc_map)
        
        if ext in {'.mp4', '.avi', '.mov'}:
            result = predict_video(api, classifier, fpath, view_text=view_text, device=device)
            results.append({"file": fname, "view": view, "desc": view_text, **result})
        elif ext in {'.jpg', '.jpeg', '.png'}:
            result = predict_image(api, classifier, fpath, view_text=view_text, device=device)
            results.append({"file": fname, "view": view, "desc": view_text, **result})
    
    if not results:
        return {"error": "No valid media files found"}
    
    # 投票
    preds = [r["prediction"] for r in results]
    votes = {0: preds.count(0), 1: preds.count(1)}
    final_pred = 1 if votes[1] > votes[0] else 0
    
    return {
        "patient_dir": patient_dir,
        "final_prediction": final_pred,
        "final_label": "HOCM阳性" if final_pred == 1 else "体检阴性",
        "votes": votes,
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-Embedding 二分类推理")
    parser.add_argument("--video", type=str, help="单个视频路径")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--dir", type=str, help="患者文件夹路径（含多个视频/图片）")
    parser.add_argument("--model", type=str, default="classifier_head.pt", help="分类头权重路径")
    parser.add_argument("--api_key", type=str, default=None, help="DashScope API Key")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--embedding_dim", type=int, default=2048, help="Embedding维度")
    args = parser.parse_args()
    
    # 加载 API Key
    from embedding_api import get_api_key as _get_key
    api_key = _get_key(args.api_key)
    if not api_key:
        print("[ERROR] 请提供 --api_key 或设置 DASHSCOPE_API_KEY 环境变量")
        return
    
    # 加载模型
    api, classifier = load_model(
        api_key=api_key,
        classifier_path=args.model,
        embedding_dim=args.embedding_dim,
        device=args.device,
    )
    
    device = args.device
    
    # 推理
    if args.video:
        print(f"\n[Infer] 视频: {args.video}")
        result = predict_video(api, classifier, args.video, device)
        print(f"  预测: {result['label']} (置信度: {result['confidence']:.4f})")
        print(f"  概率: 阴性={result['probabilities'][0]:.4f}, 阳性={result['probabilities'][1]:.4f}")
    
    elif args.image:
        print(f"\n[Infer] 图片: {args.image}")
        result = predict_image(api, classifier, args.image, device)
        print(f"  预测: {result['label']} (置信度: {result['confidence']:.4f})")
        print(f"  概率: 阴性={result['probabilities'][0]:.4f}, 阳性={result['probabilities'][1]:.4f}")
    
    elif args.dir:
        print(f"\n[Infer] 患者文件夹: {args.dir}")
        result = predict_patient(api, classifier, args.dir, device)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  综合预测: {result['final_label']}")
            print(f"  投票: 阴性={result['votes'][0]}, 阳性={result['votes'][1]}")
            print(f"\n  各文件详情:")
            for detail in result["details"]:
                print(f"    {detail['file']}: {detail['label']} ({detail['confidence']:.4f})")
    
    else:
        print("请指定 --video, --image 或 --dir")
        # 演示: 对工作空间中第一个视频做推理
        print("\n[Demo] 自动寻找测试视频...")
        root = os.path.dirname(__file__)
        for folder in ["HOCM玛法凯泰 影像", "体检中心影像"]:
            fp = os.path.join(root, folder)
            if os.path.exists(fp):
                for patient in os.listdir(fp):
                    pp = os.path.join(fp, patient)
                    if os.path.isdir(pp):
                        for f in os.listdir(pp):
                            if f.endswith('.mp4') or f.endswith('.avi'):
                                video_path = os.path.join(pp, f)
                                print(f"\n  找到: {video_path}")
                                result = predict_video(api, classifier, video_path, device)
                                print(f"  预测: {result['label']} (置信度: {result['confidence']:.4f})")
                                return


if __name__ == "__main__":
    main()
