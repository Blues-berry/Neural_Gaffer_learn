import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention import Attention
from diffusers.models.resnet import Downsample2D, Upsample2D
from typing import Optional, Dict, Any


class HighLightAwareAttention(Attention):
    """
    高光感知注意力机制，专门用于处理高光区域
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 高光特征提取层
        self.highlight_gate = nn.Conv2d(
            kwargs.get("query_dim", kwargs.get("dim")), 
            kwargs.get("query_dim", kwargs.get("dim")), 
            kernel_size=1
        )
        self.highlight_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # 原始注意力计算
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.to_q(hidden_states)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)
        
        # 计算高光特征
        if hidden_states.dim() == 3:
            # 将序列转换为空间特征图以进行高光检测
            hw = int((sequence_length) ** 0.5)
            spatial_hidden = hidden_states.view(batch_size, hw, hw, -1).permute(0, 3, 1, 2)
        else:
            spatial_hidden = hidden_states
            
        # 高光检测：通过门控机制识别高光区域
        highlight_features = self.highlight_gate(spatial_hidden)
        highlight_mask = torch.sigmoid(highlight_features)
        
        # 将高光信息整合到注意力中
        query = query + self.highlight_scale * highlight_mask.view(batch_size, sequence_length, -1)
        
        # 继续原始注意力计算
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)
        
        attention_probs = self.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        
        # 线性投影
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        
        return hidden_states


class SpecularAwareResBlock(nn.Module):
    """
    高光感知残差块，增强对高光反射的处理
    """
    def __init__(self, in_channels, out_channels, temb_channels=None):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        if temb_channels is not None:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 高光特征提取分支
        self.specular_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)
        )
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
            
    def forward(self, x, temb=None):
        h = x
        
        # 主分支
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 时间嵌入
        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # 高光分支
        specular_features = self.specular_branch(x)
        h = h + specular_features
        
        # 跳跃连接
        return h + self.skip_connection(x)


class EnhancedUNet2DConditionModel(UNet2DConditionModel):
    """
    增强的UNet模型，专门针对高光处理优化
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 替换注意力层为高光感知注意力
        self._replace_attention_layers()
        
        # 替换部分残差块为高光感知残差块
        self._enhance_res_blocks()
        
    def _replace_attention_layers(self):
        """替换标准注意力层为高光感知注意力层"""
        def replace_attention_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, Attention):
                    # 创建新的高光感知注意力层
                    new_attention = HighLightAwareAttention(
                        query_dim=child.to_q.in_features,
                        cross_attention_dim=child.to_k.in_features if hasattr(child, 'to_k') else None,
                        heads=child.heads,
                        dim_head=child.to_q.out_features // child.heads,
                        dropout=child.dropout
                    )
                    
                    # 复制权重
                    new_attention.to_q.load_state_dict(child.to_q.state_dict())
                    if hasattr(child, 'to_k'):
                        new_attention.to_k.load_state_dict(child.to_k.state_dict())
                        new_attention.to_v.load_state_dict(child.to_v.state_dict())
                    
                    setattr(module, name, new_attention)
                else:
                    replace_attention_recursive(child)
        
        replace_attention_recursive(self)
    
    def _enhance_res_blocks(self):
        """增强残差块以处理高光"""
        def enhance_blocks_recursive(module, depth=0):
            for name, child in module.named_children():
                # 在特定深度增强残差块（通常是下采样和上采样的关键层）
                if hasattr(child, 'conv1') and hasattr(child, 'norm1'):
                    # 检查是否是残差块
                    in_channels = child.conv1.in_channels
                    out_channels = child.conv1.out_channels
                    
                    # 创建新的高光感知残差块
                    new_block = SpecularAwareResBlock(in_channels, out_channels)
                    
                    # 复制权重（如果可能）
                    try:
                        new_block.conv1.load_state_dict(child.conv1.state_dict())
                        new_block.conv2.load_state_dict(child.conv2.state_dict())
                        new_block.norm1.load_state_dict(child.norm1.state_dict())
                        new_block.norm2.load_state_dict(child.norm2.state_dict())
                        if hasattr(child, 'skip_connection'):
                            new_block.skip_connection.load_state_dict(child.skip_connection.state_dict())
                    except:
                        pass  # 如果权重不匹配，使用随机初始化
                    
                    setattr(module, name, new_block)
                else:
                    enhance_blocks_recursive(child, depth + 1)
        
        enhance_blocks_recursive(self)
    
    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        """
        前向传播，增强高光处理
        """
        # 添加高光感知预处理
        if sample.dim() == 4:
            # 检测高光区域
            with torch.no_grad():
                # 简单的高光检测：亮度高于阈值的区域
                luminance = 0.299 * sample[:, 0] + 0.587 * sample[:, 1] + 0.114 * sample[:, 2]
                highlight_mask = (luminance > 0.8).float().unsqueeze(1)
                
                # 将高光信息传递给注意力层
                kwargs['highlight_mask'] = highlight_mask
        
        # 调用父类的前向传播
        return super().forward(sample, timestep, encoder_hidden_states, **kwargs)


def create_enhanced_unet_from_pretrained(pretrained_model_name_or_path, **kwargs):
    """
    从预训练模型创建增强的UNet
    """
    # 首先加载原始UNet
    original_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="unet", 
        **kwargs
    )
    
    # 创建增强UNet
    enhanced_unet = EnhancedUNet2DConditionModel(
        sample_size=original_unet.config.sample_size,
        in_channels=original_unet.config.in_channels,
        out_channels=original_unet.config.out_channels,
        layers_per_block=original_unet.config.layers_per_block,
        block_out_channels=original_unet.config.block_out_channels,
        down_block_types=original_unet.config.down_block_types,
        up_block_types=original_unet.config.up_block_types,
        cross_attention_dim=original_unet.config.cross_attention_dim,
    )
    
    # 复制权重
    enhanced_unet.load_state_dict(original_unet.state_dict(), strict=False)
    
    return enhanced_unet