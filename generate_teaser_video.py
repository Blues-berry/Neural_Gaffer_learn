import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from pipeline_neural_gaffer import Neural_Gaffer_StableDiffusionPipeline
import cv2
import argparse

def load_hdr_as_ldr(hdr_path, height=256, width=512):
    import cv2
    hdr = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH)
    hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
    hdr = cv2.resize(hdr, (width, height))
    ldr = np.clip(hdr ** (1/2.2), 0, 1)
    return hdr, ldr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_video", type=str, default="teaser_output.mp4")
    parser.add_argument("--envmap_dir", type=str, default="demo/environment_map_sample")
    parser.add_argument("--num_rotations", type=int, default=36)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    pretrained_model = "stabilityai/stable-diffusion-2-1-base"
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model, subfolder="image_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet")
    
    # 修改unet输入通道
    conv_in_16 = torch.nn.Conv2d(16, unet.conv_in.out_channels, kernel_size=unet.conv_in.kernel_size, padding=unet.conv_in.padding)
    torch.nn.init.zeros_(conv_in_16.weight)
    conv_in_16.weight[:,:8,:,:].copy_(unet.conv_in.weight)
    conv_in_16.bias.copy_(unet.conv_in.bias)
    unet.conv_in = conv_in_16
    
    # 加载checkpoint
    state_dict = torch.load(os.path.join(args.checkpoint, "pytorch_model.bin"), map_location="cpu")
    unet.load_state_dict(state_dict, strict=False)
    
    vae.to(device).eval()
    image_encoder.to(device).eval()
    unet.to(device).eval()
    
    scheduler = DDIMScheduler.from_pretrained(pretrained_model, subfolder="scheduler")
    pipeline = Neural_Gaffer_StableDiffusionPipeline.from_pretrained(
        pretrained_model, vae=vae, image_encoder=image_encoder, feature_extractor=None,
        unet=unet, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16
    )
    pipeline = pipeline.to(device)
    
    # 加载输入图片
    image_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_img = Image.open(args.input_image).convert("RGB")
    input_tensor = image_transforms(input_img).unsqueeze(0).to(device, dtype=torch.float16)
    
    # 加载环境贴图
    envmap_files = sorted([f for f in os.listdir(args.envmap_dir) if f.endswith('.exr')])
    if not envmap_files:
        raise ValueError(f"No .exr files found in {args.envmap_dir}")
    
    frames = []
    
    for envmap_file in envmap_files:
        envmap_path = os.path.join(args.envmap_dir, envmap_file)
        hdr, ldr = load_hdr_as_ldr(envmap_path)
        
        # 旋转环境贴图生成多帧
        for rot_idx in range(args.num_rotations):
            angle = (rot_idx / args.num_rotations) * hdr.shape[1]
            hdr_rotated = np.roll(hdr, int(angle), axis=1)
            ldr_rotated = np.roll(ldr, int(angle), axis=1)
            
            hdr_tensor = torch.from_numpy(hdr_rotated).permute(2,0,1).unsqueeze(0).to(device, dtype=torch.float16) * 2 - 1
            ldr_tensor = torch.from_numpy(ldr_rotated).permute(2,0,1).unsqueeze(0).to(device, dtype=torch.float16) * 2 - 1
            
            with torch.no_grad(), torch.autocast("cuda"):
                output = pipeline(
                    input_imgs=input_tensor, prompt_imgs=input_tensor,
                    first_target_envir_map=hdr_tensor, second_target_envir_map=ldr_tensor,
                    poses=None, height=args.resolution, width=args.resolution,
                    guidance_scale=3.0, num_inference_steps=50
                ).images[0]
            
            frames.append(np.array(output))
    
    # 生成视频
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, args.fps, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"视频已保存到: {args.output_video}")

if __name__ == "__main__":
    main()

