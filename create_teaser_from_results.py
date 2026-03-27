#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

def create_teaser_video(result_dir, output_video="teaser_output.mp4", fps=12):
    """从推理结果创建teaser视频"""
    
    # 查找所有预测图片
    pred_dir = Path(result_dir)
    image_folders = [d for d in pred_dir.iterdir() if d.is_dir()]
    
    if not image_folders:
        print(f"错误: {result_dir} 中没有找到结果文件夹")
        return
    
    frames = []
    
    for img_folder in sorted(image_folders):
        pred_img_dir = img_folder / "pred_image"
        if not pred_img_dir.exists():
            continue
            
        pred_images = sorted(pred_img_dir.glob("*.png"))
        
        for img_path in pred_images:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                frames.append(frame)
    
    if not frames:
        print("错误: 没有找到可用的图片")
        return
    
    # 创建视频
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"✓ 视频已保存: {output_video} ({len(frames)} 帧)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True, help="推理结果目录")
    parser.add_argument("--output", type=str, default="teaser_output.mp4", help="输出视频路径")
    parser.add_argument("--fps", type=int, default=12, help="视频帧率")
    args = parser.parse_args()
    
    create_teaser_video(args.result_dir, args.output, args.fps)

