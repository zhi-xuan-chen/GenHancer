import clip
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import SiglipProcessor, SiglipModel, SiglipImageProcessor, SiglipTokenizer
from einops import rearrange, repeat
from timm.models.vision_transformer import Attention, vit_base_patch16_224
import torch

class XRCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = vit_base_patch16_224(in_chans=1)
        self.model.head = nn.Identity()
        self.model.load_state_dict(torch.load('/jhcnas5/chenzhixuan/checkpoints/VIRAL/XR_clip.ckpt'), strict=True)
        self.model.to(torch.float32)
        self.config = config
        self.project_clip = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, config.clip_dim),
            nn.GELU(),
            nn.Linear(config.clip_dim, config.clip_dim),
        )
        self.project_t5 = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, config.t5_dim),
            nn.GELU(),
            nn.Linear(config.t5_dim, config.t5_dim),
        )
    def forward(self, images):  
        # Convert RGB (3 channels) to grayscale (1 channel) for XRCLIP
        # Since grayscale images converted to RGB have identical channels (R=G=B),
        # we can simply take the first channel
        if images.shape[1] == 3:
            images = images[:, 0:1, :, :]  # Take first channel and keep dim [B, 1, H, W]
        
        all_tokens = self.model.forward_features(images) # [B, 197, 768]
        class_token = all_tokens[:, 0, :] # [B, 768]
        class_token_expand = repeat(class_token, 'b d -> b l d', l=1)
        projection_clip = self.project_clip(class_token)
        projection_t5 = self.project_t5(class_token_expand)

        return class_token, projection_clip, projection_t5


class OpenAICLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        if config.clip_image_size == 336:
            model = CLIPModel.from_pretrained('laion/CLIP-ViT-L-14-laion2B-s32B-b82K')

        self.project_clip = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, config.clip_dim),
            nn.GELU(),
            nn.Linear(config.clip_dim, config.clip_dim),
        )
        self.project_t5 = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, config.t5_dim),
            nn.GELU(),
            nn.Linear(config.t5_dim, config.t5_dim),
        )
        self.model = model
        self.config = config

    def forward(self, images):

        class_token_pre = self.model.vision_model(images).pooler_output
        class_token = self.model.visual_projection(class_token_pre)
        class_token_expand = repeat(class_token, 'b d -> b l d', l=1)
        projection_clip = self.project_clip(class_token)
        projection_t5 = self.project_t5(class_token_expand)

        return class_token, projection_clip, projection_t5


class SigLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_image_size == 224:
            model = SiglipModel.from_pretrained('google/siglip-so400m-patch14-224')
        if config.clip_image_size == 384:
            model = SiglipModel.from_pretrained('google/siglip-so400m-patch14-384')

        self.project_clip = nn.Sequential(
            nn.LayerNorm(1152),
            nn.Linear(1152, config.clip_dim),
            nn.GELU(),
            nn.Linear(config.clip_dim, config.clip_dim),
        )
        self.project_t5 = nn.Sequential(
            nn.LayerNorm(1152),
            nn.Linear(1152, config.t5_dim),
            nn.GELU(),
            nn.Linear(config.t5_dim, config.t5_dim),
        )
        self.model = model
        self.config = config

    def forward(self, images):
        class_token = self.model.vision_model(images).pooler_output
        class_token_expand = repeat(class_token, 'b d -> b l d', l=1)
        projection_clip = self.project_clip(class_token)
        projection_t5 = self.project_t5(class_token_expand)

        return class_token, projection_clip, projection_t5


class MetaCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.clip_type == 'large':
            model = CLIPModel.from_pretrained('facebook/metaclip-l14-fullcc2.5b')

            self.project_clip = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, config.clip_dim),
                nn.GELU(),
                nn.Linear(config.clip_dim, config.clip_dim),
            )
            self.project_t5 = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, config.t5_dim),
                nn.GELU(),
                nn.Linear(config.t5_dim, config.t5_dim),
            )

        if config.clip_type == 'huge':
            model = CLIPModel.from_pretrained('facebook/metaclip-h14-fullcc2.5b')

            self.project_clip = nn.Sequential(
                nn.LayerNorm(1024),
                nn.Linear(1024, config.clip_dim),
                nn.GELU(),
                nn.Linear(config.clip_dim, config.clip_dim),
            )
            self.project_t5 = nn.Sequential(
                nn.LayerNorm(1024),
                nn.Linear(1024, config.t5_dim),
                nn.GELU(),
                nn.Linear(config.t5_dim, config.t5_dim),
            )

        self.model = model
        self.config = config

    def forward(self, images):
        class_token_pre = self.model.vision_model(images).pooler_output
        class_token = self.model.visual_projection(class_token_pre)
        class_token_expand = repeat(class_token, 'b d -> b l d', l=1)
        projection_clip = self.project_clip(class_token)
        projection_t5 = self.project_t5(class_token_expand)

        return class_token, projection_clip, projection_t5
