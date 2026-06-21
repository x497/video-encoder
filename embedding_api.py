"""
Qwen3-VL-Embedding API 封装模块 (DashScope 远程 API 版)

通过阿里云 DashScope 的 HTTP API 调用 qwen3-vl-embedding 模型。
支持: 纯文本 / 图片+文本 / 视频+文本 同时输入。

输出维度:
  - qwen3-vl-embedding-2b: 2048
  - qwen3-vl-embedding-8b: 4096
"""

import os
import base64
import time
import json
import subprocess
import tempfile
from typing import List, Dict, Optional, Union

import requests
import numpy as np


# ============================================================
# DashScope 多模态 Embedding API
# ============================================================

class QwenVLEmbeddingAPI:
    """通过 DashScope API 调用 qwen3-vl-embedding"""

    # DashScope 多模态 embedding endpoint
    _ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

    # 支持的模型
    EMBEDDING_DIMS = {
        "qwen3-vl-embedding-2b": 2048,
        "qwen3-vl-embedding-8b": 4096,
        "qwen-vl-embedding2": 2560,
        "multimodal-embedding-v1": 1024,
    }

    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3-vl-embedding-2b",
        max_frames: int = 16,
        fps: float = 2.0,
        normalize: bool = True,
        endpoint: str = None,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_frames = max_frames
        self.fps = fps
        self.normalize = normalize
        self.endpoint = endpoint or self._ENDPOINT
        self.max_retries = max_retries
        self.embedding_dim = self.EMBEDDING_DIMS.get(model_name, 2048)

        print(f"[EmbeddingAPI] 初始化完成 (DashScope)")
        print(f"  Model: {model_name}, Dim: {self.embedding_dim}")
        print(f"  Endpoint: {self.endpoint}")

    # ============================================================
    # 核心 API 调用
    # ============================================================

    def _call_api(self, contents: List[Dict], extra_params: Dict = None) -> np.ndarray:
        """调用 DashScope 多模态 embedding API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "input": {"contents": contents},
            "parameters": {},
        }
        if extra_params:
            payload["parameters"].update(extra_params)

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=120
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return self._parse_response(data)

                elif resp.status_code == 429:
                    wait = min((attempt + 1) * 5, 30)
                    print(f"  [RateLimit] 等待 {wait}s...")
                    time.sleep(wait)
                    continue

                else:
                    print(f"  [ERROR] HTTP {resp.status_code}: {resp.text[:500]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(min((attempt + 1) * 2, 10))
                        continue
                    raise RuntimeError(f"API 失败: {resp.status_code}")

            except requests.exceptions.Timeout:
                print(f"  [Timeout] 重试 {attempt+1}/{self.max_retries}...")
                time.sleep(5)
                continue
            except requests.exceptions.ConnectionError as e:
                print(f"  [ConnectionError] {e}")
                time.sleep(3)
                continue

        raise RuntimeError(f"API 调用失败，已重试 {self.max_retries} 次")

    def _parse_response(self, data: Dict) -> np.ndarray:
        """解析 DashScope 返回的 embedding"""
        output = data.get("output", {})
        embeddings = output.get("embeddings", [])

        if not embeddings:
            # 尝试其他可能的返回格式
            if "data" in data:
                embeddings = data["data"]
            else:
                raise RuntimeError(f"未知返回格式: {json.dumps(data, ensure_ascii=False)[:500]}")

        emb = embeddings[0].get("embedding", [])
        emb_array = np.array(emb, dtype=np.float32).reshape(1, -1)

        if self.normalize:
            norm = np.linalg.norm(emb_array, axis=-1, keepdims=True)
            emb_array = emb_array / (norm + 1e-8)

        return emb_array

    # ============================================================
    # 编码方法
    # ============================================================

    def encode_text(self, text: str) -> np.ndarray:
        """纯文本编码"""
        contents = [{"text": text}]
        return self._call_api(contents)

    def encode_image(self, image_path: str, text: str = "") -> np.ndarray:
        """图片编码（可选附带文字）"""
        image_b64 = self._encode_image_to_base64(image_path)
        contents = [{"image": image_b64}]
        if text:
            contents.append({"text": text})
        return self._call_api(contents)

    def encode_image_with_text(
        self, image_path: str, text: str = "", instruction: str = ""
    ) -> np.ndarray:
        """图片 + 文字描述 同时编码"""
        image_b64 = self._encode_image_to_base64(image_path)
        contents = [{"image": image_b64}]
        if instruction:
            contents.append({"text": instruction})
        if text:
            contents.append({"text": text})
        return self._call_api(contents)

    def encode_video(self, video_path: str) -> np.ndarray:
        """纯视频编码（自动处理大视频：压缩或抽帧）"""
        return self._encode_video_smart(video_path)

    def encode_video_with_text(
        self, video_path: str, text: str = "", instruction: str = ""
    ) -> np.ndarray:
        """视频 + 文字描述 同时编码（自动处理大视频）"""
        return self._encode_video_smart(video_path, text=text, instruction=instruction)

    def _encode_video_smart(
        self, video_path: str, text: str = "", instruction: str = ""
    ) -> np.ndarray:
        """智能视频编码：≤15MB+≥2帧直传，否则压缩，失败回退单帧图片"""
        max_size = 15 * 1024 * 1024  # base64膨胀1.33x后 <= 20MB
        file_size = os.path.getsize(video_path)
        frame_count = self._get_video_frame_count(video_path)

        # 路径1：直传原始视频
        if file_size <= max_size and frame_count >= 2:
            try:
                video_b64 = self._encode_video_to_base64(video_path)
                contents = [{"video": video_b64}]
                if instruction:
                    contents.append({"text": instruction})
                if text:
                    contents.append({"text": text})
                return self._call_api(contents)
            except Exception as e:
                print(f"  [Video] 直传失败: {e}, 尝试压缩...")

        # 路径2：ffmpeg 压缩
        reason = f"太大 ({file_size/1024/1024:.1f}MB)" if file_size > max_size else f"帧数不足({frame_count})"
        if file_size <= max_size and frame_count >= 2:
            print(f"  [Video] {os.path.basename(video_path)} 直传失败，压缩中...")
        else:
            print(f"  [Video] {os.path.basename(video_path)} {reason}, 压缩中...")
            
        compressed_path = self._compress_video(video_path, max_size_mb=19)
        if compressed_path and os.path.exists(compressed_path) and os.path.getsize(compressed_path) > 0:
            new_size = os.path.getsize(compressed_path)
            print(f"    压缩后: {new_size/1024/1024:.1f}MB")
            try:
                video_b64 = self._encode_video_to_base64(compressed_path)
                contents = [{"video": video_b64}]
                if instruction:
                    contents.append({"text": instruction})
                if text:
                    contents.append({"text": text})
                result = self._call_api(contents)
                try:
                    os.remove(compressed_path)
                except:
                    pass
                return result
            except Exception as e:
                print(f"    压缩后仍失败: {e}, 回退单帧...")
                try:
                    os.remove(compressed_path)
                except:
                    pass

        # 路径3：终极回退 - 单帧图片
        print(f"    回退单帧图片...")
        frame_b64 = self._extract_single_frame(video_path)
        contents = [{"image": frame_b64}]
        if instruction:
            contents.append({"text": instruction})
        if text:
            contents.append({"text": text})
        return self._call_api(contents)

    def encode_video_with_frames(
        self, video_path: str, text: str = "", instruction: str = "", num_frames: int = 8
    ) -> np.ndarray:
        """视频（抽帧为图片序列）+ 文字 编码（大视频备选方案）"""
        frames = self._extract_frames(video_path, num_frames)
        contents = [{"image": f} for f in frames]
        if instruction:
            contents.append({"text": instruction})
        if text:
            contents.append({"text": text})
        return self._call_api(contents)

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _get_video_frame_count(video_path: str) -> int:
        """获取视频总帧数"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return count
        except:
            return 999  # 假设够用

    @staticmethod
    def _compress_video(video_path: str, max_size_mb: float = 9.0) -> str:
        """压缩视频到指定大小以内，返回压缩后路径"""
        import subprocess
        import tempfile

        out_path = os.path.join(
            tempfile.gettempdir(),
            f"compressed_{os.path.basename(video_path)}"
        )

        # 查找 ffmpeg 可执行文件
        ffmpeg_bin = "ffmpeg"
        # 尝试 conda 环境下的完整路径
        conda_ffmpeg = os.path.join(
            os.path.dirname(os.path.dirname(os.__file__)),
            "Library", "bin", "ffmpeg.exe"
        )
        if os.path.exists(conda_ffmpeg):
            ffmpeg_bin = conda_ffmpeg

        # ffmpeg 压缩：大幅降低分辨率和码率
        cmd = [
            ffmpeg_bin, "-y", "-i", video_path,
            "-vf", "scale=256:-1,fps=2",      # 极低分辨率 + 2fps
            "-b:v", "100k",                     # 极低码率
            "-maxrate", "150k",
            "-bufsize", "300k",
            "-an",                              # 去音频
            "-t", "30",                         # 最多30秒
            "-c:v", "libx264",
            "-preset", "ultrafast",
            out_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return out_path
            else:
                print(f"    ffmpeg stderr: {result.stderr.decode('utf-8', errors='ignore')[-200:]}")
        except FileNotFoundError:
            print("    [WARNING] ffmpeg 未找到")
        except Exception as e:
            print(f"    [WARNING] 压缩失败: {e}")
        return None

    @staticmethod
    def _extract_single_frame(video_path: str) -> str:
        """从视频中间抽取一帧，返回 base64"""
        try:
            import cv2
        except ImportError:
            raise ImportError("视频抽帧需要 opencv-python")

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"无法读取视频帧: {video_path}")

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _encode_image_to_base64(image_path: str) -> str:
        import mimetypes
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{data}"

    @staticmethod
    def _encode_video_to_base64(video_path: str) -> str:
        import mimetypes
        mime = mimetypes.guess_type(video_path)[0] or "video/mp4"
        with open(video_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{data}"

    @staticmethod
    def _extract_frames(video_path: str, num_frames: int = 8) -> List[str]:
        """从视频均匀抽帧，返回 base64 列表"""
        try:
            import cv2
        except ImportError:
            raise ImportError("视频抽帧需要 opencv-python: pip install opencv-python")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"无法读取视频: {video_path}")

        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            indices = [int(i * step) for i in range(num_frames)]

        frames_b64 = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                b64 = base64.b64encode(buffer).decode('utf-8')
                frames_b64.append(f"data:image/jpeg;base64,{b64}")

        cap.release()
        return frames_b64


# ============================================================
# API Key 管理
# ============================================================

def get_api_key(
    api_key: str = None,
    env_var: str = "DASHSCOPE_API_KEY",
    key_file: str = ".api_key",
) -> str:
    """从多个来源获取 API Key（优先级: 直接传入 > 环境变量 > 文件）"""
    if api_key:
        return api_key
    api_key = os.environ.get(env_var)
    if api_key:
        return api_key
    wd_key = os.path.join(os.path.dirname(__file__), key_file)
    if os.path.exists(wd_key):
        with open(wd_key, "r") as f:
            return f.read().strip()
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            return f.read().strip()
    return None


# ============================================================
# 测试
# ============================================================

def test_api(api_key: str = None):
    """测试 API 连通性"""
    api_key = get_api_key(api_key)
    if not api_key:
        print("[ERROR] 请设置 DASHSCOPE_API_KEY 环境变量或提供 API Key")
        return None

    print("=" * 60)
    print("测试 DashScope Qwen3-VL-Embedding API")
    print("=" * 60)

    api = QwenVLEmbeddingAPI(api_key=api_key, model_name="qwen3-vl-embedding-2b")

    # Test 1: 纯文本
    print("\n[Test 1] 纯文本编码...")
    try:
        emb = api.encode_text("这是一段心脏超声描述文本")
        print(f"  Shape: {emb.shape}, Dim: {emb.shape[-1]}")
        print(f"  Norm: {np.linalg.norm(emb):.6f}")
        print("  [OK] 文本编码成功!")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # Test 2: 图片
    print("\n[Test 2] 图片编码...")
    root = os.path.dirname(__file__)
    for folder in ["HOCM玛法凯泰 影像", "体检中心影像"]:
        fp = os.path.join(root, folder)
        if os.path.exists(fp):
            for patient in os.listdir(fp):
                pp = os.path.join(fp, patient)
                if os.path.isdir(pp):
                    jpgs = [f for f in os.listdir(pp) if f.endswith('.jpg')]
                    if jpgs:
                        img_path = os.path.join(pp, jpgs[0])
                        try:
                            emb = api.encode_image(img_path)
                            print(f"  {jpgs[0]}: Shape={emb.shape}")
                            print("  [OK]")
                        except Exception as e:
                            print(f"  [FAIL] {e}")
                        break
            break

    # Test 3: 图片+文字
    print("\n[Test 3] 图片+文字编码...")
    for folder in ["HOCM玛法凯泰 影像"]:
        fp = os.path.join(root, folder)
        if os.path.exists(fp):
            for patient in os.listdir(fp):
                pp = os.path.join(fp, patient)
                if os.path.isdir(pp):
                    jpgs = [f for f in os.listdir(pp) if 'LV-GLS' in f and f.endswith('.jpg')]
                    if jpgs:
                        img_path = os.path.join(pp, jpgs[0])
                        try:
                            emb = api.encode_image_with_text(
                                img_path,
                                text="这是左室心肌纵向应变图",
                            )
                            print(f"  {jpgs[0]}: Shape={emb.shape}")
                            print("  [OK]")
                        except Exception as e:
                            print(f"  [FAIL] {e}")
                        break
            break

    print("\n[Test] 完成!")
    return api


if __name__ == "__main__":
    test_api()
