import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin


def resolve_light_probe_maps(
    hdr_env_map: torch.Tensor,
    ldr_env_map: torch.Tensor,
    env_map_mode: str = "ldr",
    hdr_log_compress: bool = True,
) -> torch.Tensor:
    """
    组装 light-probe 编码器的输入。

    默认只使用 LDR 分支，原因是:
    - 保留原始 Neural Gaffer 的 HDR/LDR latent 条件闭环不动
    - 新增的 light-probe 分支尽量做轻量、稳定、低耦合增强
    - 避免 HDR 极大动态范围在新增支路里主导 token 幅值
    """
    mode = str(env_map_mode or "ldr").strip().lower()
    hdr_input = hdr_env_map.float()
    if hdr_log_compress:
        hdr_input = torch.sign(hdr_input) * torch.log1p(hdr_input.abs())

    if mode == "hdr":
        return hdr_input
    if mode == "ldr":
        return ldr_env_map.float()
    if mode == "both":
        return torch.cat([hdr_input, ldr_env_map.float()], dim=1)
    raise ValueError(f"Unsupported light probe env_map_mode={env_map_mode!r}. Choose from ['ldr', 'hdr', 'both'].")


class LightProbeEncoder(ModelMixin, ConfigMixin):
    """
    轻量环境光探针编码器。

    目标:
    - 从环境图提取一组全局 + 局部 light tokens
    - 这些 tokens 和原有 CLIP image prompt 一起送入 UNet cross-attention
    - 尽量不改原有 Neural Gaffer 的主数据流，只作为显式环境光增强支路
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 128,
        token_dim: int = 768,
        input_height: int = 32,
        input_width: int = 64,
        global_token_count: int = 2,
        local_grid_height: int = 2,
        local_grid_width: int = 4,
        use_sh_features: bool = True,
        env_map_mode: str = "ldr",
        hdr_log_compress: bool = True,
    ):
        super().__init__()
        self.register_to_config(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            token_dim=token_dim,
            input_height=input_height,
            input_width=input_width,
            global_token_count=global_token_count,
            local_grid_height=local_grid_height,
            local_grid_width=local_grid_width,
            use_sh_features=use_sh_features,
            env_map_mode=env_map_mode,
            hdr_log_compress=hdr_log_compress,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, global_token_count * token_dim),
        )
        self.local_proj = nn.Linear(hidden_dim, token_dim)
        self.sh_proj = nn.Sequential(
            nn.Linear(27, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    @staticmethod
    def _make_equirectangular_directions(height: int, width: int, device, dtype):
        theta = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height * math.pi
        phi = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width * (2.0 * math.pi) - math.pi
        theta, phi = torch.meshgrid(theta, phi, indexing="ij")
        sin_theta = torch.sin(theta)
        x = sin_theta * torch.cos(phi)
        y = torch.cos(theta)
        z = sin_theta * torch.sin(phi)
        return x, y, z, sin_theta

    def _compute_low_order_sh_features(self, env_map: torch.Tensor) -> torch.Tensor:
        height, width = env_map.shape[-2:]
        x, y, z, sin_theta = self._make_equirectangular_directions(
            height, width, env_map.device, env_map.dtype
        )
        basis = torch.stack(
            [
                torch.ones_like(x),
                x,
                y,
                z,
                x * y,
                y * z,
                3.0 * z * z - 1.0,
                x * z,
                x * x - y * y,
            ],
            dim=0,
        )
        weighted_basis = basis * sin_theta.unsqueeze(0)
        basis_flat = weighted_basis.view(9, -1)
        env_flat = env_map.view(env_map.shape[0], env_map.shape[1], -1)
        sh_rgb = torch.einsum("bcn,kn->bck", env_flat, basis_flat)
        norm = sin_theta.sum().clamp_min(1e-6)
        sh_rgb = sh_rgb / norm
        return sh_rgb.reshape(env_map.shape[0], -1)

    def forward(
        self,
        hdr_env_map: torch.Tensor,
        ldr_env_map: torch.Tensor,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if hdr_env_map.shape != ldr_env_map.shape:
            raise ValueError(
                f"Expected HDR/LDR environment maps to share the same shape, got {hdr_env_map.shape} vs {ldr_env_map.shape}."
            )

        dtype = output_dtype or hdr_env_map.dtype
        resize_hw = (self.config.input_height, self.config.input_width)
        probe_input = resolve_light_probe_maps(
            hdr_env_map=hdr_env_map,
            ldr_env_map=ldr_env_map,
            env_map_mode=self.config.env_map_mode,
            hdr_log_compress=self.config.hdr_log_compress,
        )
        probe_input = F.interpolate(probe_input, size=resize_hw, mode="bilinear", align_corners=False)
        features = self.stem(probe_input.float())

        global_feature = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        global_tokens = self.global_proj(global_feature).view(
            features.shape[0], self.config.global_token_count, self.config.token_dim
        )

        local_feature = F.adaptive_avg_pool2d(
            features,
            output_size=(self.config.local_grid_height, self.config.local_grid_width),
        )
        local_tokens = self.local_proj(local_feature.flatten(2).transpose(1, 2))

        tokens = [global_tokens]
        if self.config.use_sh_features:
            sh_features = self._compute_low_order_sh_features(probe_input)
            tokens.append(self.sh_proj(sh_features).unsqueeze(1))
        tokens.append(local_tokens)
        light_tokens = torch.cat(tokens, dim=1).to(dtype=dtype)

        if num_images_per_prompt > 1:
            light_tokens = light_tokens.repeat_interleave(num_images_per_prompt, dim=0)
        if do_classifier_free_guidance:
            light_tokens = torch.cat([torch.zeros_like(light_tokens), light_tokens], dim=0)
        return light_tokens
