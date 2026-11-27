"""Improved DETR-Style Pre-LN ACT (State-ACT & Vision-ACT) for Push-T.

Key algorithmic improvements over the earlier compact ACT implementation:
1. True DETR Pre-LN Transformer Encoder & Decoder:
   - Pre-LayerNorm (norm_first) + final LayerNorm for stable high-LR optimization.
   - Layer-wise query_pos and encoder_pos injection into Q/K at EVERY decoder layer
     with zero-initialized decoder tgt (exact DETR / Tony Zhao ACT / LeRobot ACT).
2. Multi-token structured observation encoding (eliminates 1-token attention degeneracy):
   - State-ACT tokenizes per-frame pusher-centric geometry & object-goal geometry
     separately across observation history (no prev_action causal confusion).
   - Vision-ACT uses dual independent GroupNorm ResNet18 encoders with BOTH
     2D-sinusoidal spatial feature map tokens (4x4) AND multi-scale SpatialSoftmax
     sub-pixel keypoint tokens (layer3 8x8 + layer4 4x4) + camera/time embeddings.
3. Properly regularized CVAE posterior encoder with 1D sinusoidal positional encoding,
   supporting KL linear warmup and free-bits to prevent posterior collapse.
4. Front-weighted masked L1 chunk loss to sharpen immediately executed actions while
   preserving full action-chunk coherence.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


def sinusoidal_1d(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # [1, L, D]


def sinusoidal_2d(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """2D sinusoidal positional embedding (exact DETR / ACT formulation)."""
    assert dim % 4 == 0, "d_model must be divisible by 4 for 2D sinusoidal PE"
    half = dim // 2
    y_embed = torch.arange(1, h + 1, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, w)
    x_embed = torch.arange(1, w + 1, device=device, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
    y_embed = y_embed / (h + 1e-6) * 2.0 * math.pi
    x_embed = x_embed / (w + 1e-6) * 2.0 * math.pi

    dim_t = torch.arange(half, device=device, dtype=torch.float32)
    dim_t = 10000.0 ** (2 * (dim_t // 2) / half)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).reshape(1, h * w, dim)
    return pos  # [1, H*W, D]


class PreLNEncoderLayer(nn.Module):
    """DETR-style Pre-LN Transformer Encoder Layer with explicit Q/K pos injection."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(src)
        q = k = x if pos is None else x + pos
        attn_out = self.self_attn(q, k, value=x, key_padding_mask=src_key_padding_mask, need_weights=False)[0]
        src = src + self.dropout1(attn_out)
        x = self.norm2(src)
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        src = src + self.dropout2(ff)
        return src


class PreLNDecoderLayer(nn.Module):
    """DETR-style Pre-LN Transformer Decoder Layer with layer-wise query_pos & memory pos."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        memory_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Self-attention over action chunk queries
        x = self.norm1(tgt)
        q = k = x + query_pos
        sa = self.self_attn(q, k, value=x, need_weights=False)[0]
        tgt = tgt + self.dropout1(sa)

        # 2. Cross-attention to encoder observation memory
        x = self.norm2(tgt)
        q = x + query_pos
        k = memory if memory_pos is None else memory + memory_pos
        ca = self.cross_attn(q, k, value=memory, need_weights=False)[0]
        tgt = tgt + self.dropout2(ca)

        # 3. Feed-forward network
        x = self.norm3(tgt)
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        tgt = tgt + self.dropout3(ff)
        return tgt


class DETRTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [PreLNEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_layers = nn.ModuleList(
            [PreLNDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        src_pos: torch.Tensor,
        query_pos: torch.Tensor,
        return_memory: bool = False,
    ):
        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, pos=src_pos)
        memory = self.encoder_norm(memory)

        tgt = torch.zeros_like(query_pos)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, query_pos=query_pos, memory_pos=src_pos)
        out = self.decoder_norm(tgt)
        if return_memory:
            return out, memory
        return out


class CVAEEncoder(nn.Module):
    """ACT CVAE posterior encoder q(z | obs_summary, action_chunk)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        chunk_size: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.obs_proj = nn.Sequential(nn.Linear(obs_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.action_proj = nn.Linear(action_dim, d_model)
        self.layers = nn.ModuleList(
            [PreLNEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.latent_head = nn.Linear(d_model, latent_dim * 2)
        self.register_buffer("pos_embed", sinusoidal_1d(2 + chunk_size, d_model, torch.device("cpu")), persistent=False)

    def forward(
        self,
        obs_summary: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor,
        free_bits: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b = obs_summary.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        obs_tok = self.obs_proj(obs_summary).unsqueeze(1)
        act_tok = self.action_proj(actions)
        tokens = torch.cat([cls, obs_tok, act_tok], dim=1)
        pos = self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        padding = torch.cat([torch.zeros((b, 2), dtype=torch.bool, device=tokens.device), ~valid], dim=1)

        h = tokens
        for layer in self.layers:
            h = layer(h, pos=pos, src_key_padding_mask=padding)
        cls_out = self.norm(h)[:, 0]

        mu, logvar = self.latent_head(cls_out).float().chunk(2, dim=-1)
        logvar = logvar.clamp(-6.0, 2.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(mu)

        kl_per_dim = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)  # [B, latent_dim]
        raw_kl = kl_per_dim.sum(dim=-1).mean()
        if free_bits > 0.0:
            kl_loss = kl_per_dim.clamp(min=free_bits).sum(dim=-1).mean()
        else:
            kl_loss = raw_kl

        diag = {
            "raw_kl": raw_kl.detach(),
            "active_latents": (mu.var(dim=0, unbiased=False) > 0.01).sum().detach().float(),
            "mean_logvar": logvar.mean().detach(),
            "mu_norm": mu.norm(dim=-1).mean().detach(),
        }
        return z, kl_loss, diag


class StateACT(nn.Module):
    """Improved multi-token Pre-LN DETR State-ACT for precision Push-T.

    Expects normalized state sequences of shape [B, n_obs_steps, 16]:
      - indices 0..5  (6 dims): pusher-centric [ee_x, ee_y, ee_dx, ee_dy, ee_local_x, ee_local_y]
      - indices 6..15 (10 dims): object-goal   [T_x, T_y, cos_T, sin_T, g_dx, g_dy, cos_d, sin_d, g_local_x, g_local_y]
    """

    def __init__(
        self,
        n_obs_steps: int = 2,
        pusher_dim: int = 6,
        object_dim: int = 10,
        action_dim: int = 2,
        chunk_size: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_vae: bool = True,
        latent_dim: int = 32,
        vae_layers: int = 2,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.pusher_dim = pusher_dim
        self.object_dim = object_dim
        self.state_dim = pusher_dim + object_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.use_vae = use_vae
        self.latent_dim = latent_dim

        self.pusher_proj = nn.Sequential(nn.Linear(pusher_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.object_proj = nn.Sequential(nn.Linear(object_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.z_proj = nn.Linear(latent_dim, d_model)

        num_enc_tokens = 1 + 2 * n_obs_steps  # [z, pusher_0..T-1, object_0..T-1]
        self.encoder_pos = nn.Parameter(torch.randn(1, num_enc_tokens, d_model) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)

        self.transformer = DETRTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )
        if use_vae:
            self.cvae = CVAEEncoder(
                obs_dim=n_obs_steps * self.state_dim,
                action_dim=action_dim,
                chunk_size=chunk_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=vae_layers,
                dim_feedforward=dim_feedforward,
                latent_dim=latent_dim,
                dropout=dropout,
            )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.action_head[-1].bias)
        nn.init.normal_(self.action_head[-1].weight, std=0.01)

    def forward(
        self,
        obs_seq: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        free_bits: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # obs_seq: [B, n_obs_steps, state_dim]
        b = obs_seq.shape[0]
        kl = obs_seq.new_zeros(())
        diag: Dict[str, torch.Tensor] = {}

        if self.use_vae and actions is not None:
            assert valid is not None
            z, kl, diag = self.cvae(obs_seq.reshape(b, -1), actions, valid, free_bits=free_bits)
            z = z.to(dtype=obs_seq.dtype)
        else:
            z = obs_seq.new_zeros((b, self.latent_dim))

        pusher_tok = self.pusher_proj(obs_seq[..., : self.pusher_dim])  # [B, To, D]
        object_tok = self.object_proj(obs_seq[..., self.pusher_dim :])  # [B, To, D]
        z_tok = self.z_proj(z).unsqueeze(1)  # [B, 1, D]

        src = torch.cat([z_tok, pusher_tok, object_tok], dim=1)  # [B, 1+2*To, D]
        dec_out = self.transformer(
            src=src,
            src_pos=self.encoder_pos,
            query_pos=self.query_pos.expand(b, -1, -1),
        )
        pred_actions = self.action_head(dec_out)
        return pred_actions, kl, diag


class SpatialSoftmax(nn.Module):
    def __init__(self, in_channels: int, keypoints: int = 32):
        super().__init__()
        self.project = nn.Conv2d(in_channels, keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.project(x).float()  # [B, K, H, W]
        b, k, h, w = logits.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=logits.device),
            torch.linspace(-1.0, 1.0, w, device=logits.device),
            indexing="ij",
        )
        probs = logits.flatten(2).softmax(dim=-1)
        ex = (probs * xx.flatten()).sum(dim=-1)
        ey = (probs * yy.flatten()).sum(dim=-1)
        return torch.stack([ex, ey], dim=-1).reshape(b, k * 2)


class MultiScaleRGBEncoder(nn.Module):
    """From-scratch GroupNorm ResNet18 returning multi-scale SpatialSoftmax & optional 4x4 map."""

    def __init__(self, d_model: int = 256, keypoints_l3: int = 32, keypoints_l4: int = 32, token_mode: str = "hybrid"):
        super().__init__()
        self.token_mode = token_mode
        backbone = resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3  # [B, 256, 8, 8] on 116x116 input
        self.layer4 = backbone.layer4  # [B, 512, 4, 4] on 116x116 input

        self.spatial_l3 = SpatialSoftmax(256, keypoints_l3)
        self.spatial_l4 = SpatialSoftmax(512, keypoints_l4)
        if token_mode == "keypoints":
            self.kp_l3_proj = nn.Sequential(
                nn.Linear(keypoints_l3 * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.kp_l4_proj = nn.Sequential(
                nn.Linear(keypoints_l4 * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.map_proj = nn.Conv2d(512, d_model, kernel_size=1)
            kp_dim = (keypoints_l3 + keypoints_l4) * 2
            self.kp_proj = nn.Sequential(
                nn.Linear(kp_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

    def forward(self, x: torch.Tensor):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        f3 = self.layer3(h)
        f4 = self.layer4(f3)

        kp3 = self.spatial_l3(f3).to(dtype=f4.dtype)
        kp4 = self.spatial_l4(f4).to(dtype=f4.dtype)
        if self.token_mode == "keypoints":
            return self.kp_l3_proj(kp3), self.kp_l4_proj(kp4), (0, 0)

        kp = torch.cat([kp3, kp4], dim=-1)
        kp_tok = self.kp_proj(kp)  # [B, D]
        fmap = self.map_proj(f4)  # [B, D, H, W]
        _, _, fh, fw = fmap.shape
        map_tokens = fmap.flatten(2).transpose(1, 2)  # [B, H*W, D]
        return map_tokens, kp_tok, (fh, fw)


class VisionACT(nn.Module):
    """Pure-vision Pre-LN DETR Vision-ACT (dual cameras + robot_xy, zero privileged state)."""

    def __init__(
        self,
        n_obs_steps: int = 2,
        num_cameras: int = 2,
        robot_dim: int = 2,
        action_dim: int = 2,
        chunk_size: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_vae: bool = True,
        latent_dim: int = 32,
        vae_layers: int = 2,
        crop_size: int = 116,
        token_mode: str = "hybrid",
        keypoints_l3: int = 32,
        keypoints_l4: int = 32,
        aux_geom_dim: int = 0,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.num_cameras = num_cameras
        self.robot_dim = robot_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.use_vae = use_vae
        self.latent_dim = latent_dim
        self.crop_size = crop_size
        self.token_mode = token_mode
        self.keypoints_l3 = keypoints_l3
        self.keypoints_l4 = keypoints_l4
        self.aux_geom_dim = aux_geom_dim

        self.encoders = nn.ModuleList(
            [
                MultiScaleRGBEncoder(
                    d_model=d_model,
                    keypoints_l3=keypoints_l3,
                    keypoints_l4=keypoints_l4,
                    token_mode=token_mode,
                )
                for _ in range(num_cameras)
            ]
        )
        self.robot_proj = nn.Sequential(nn.Linear(robot_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.z_proj = nn.Linear(latent_dim, d_model)

        self.camera_embed = nn.Parameter(torch.randn(num_cameras, 1, d_model) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(n_obs_steps, 1, d_model) * 0.02)
        self.token_type_embed = nn.Parameter(torch.randn(4, 1, d_model) * 0.02)  # 0:z, 1:robot, 2:kp/l3, 3:map/l4
        self.query_pos = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)

        self.transformer = DETRTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )
        if aux_geom_dim > 0:
            # Training-only auxiliary geometric supervision heads (NEVER used at inference!)
            # 1) Pure-pixel per-frame head predicting crop-shift-invariant 8-d SE(2) geometry
            vis_in_dim = num_cameras * (2 if token_mode == "keypoints" else 1) * d_model
            self.vis_aux_head = nn.Sequential(
                nn.Linear(vis_in_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, 8),
            )
            # 2) Encoder-memory head predicting full 14-d SE(2) geometry (anchors robot_xy + pixels)
            self.enc_aux_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, aux_geom_dim),
            )
        if use_vae:
            obs_summary_dim = n_obs_steps * (robot_dim + num_cameras * d_model)
            self.cvae = CVAEEncoder(
                obs_dim=obs_summary_dim,
                action_dim=action_dim,
                chunk_size=chunk_size,
                d_model=d_model,
                nhead=nhead,
                num_layers=vae_layers,
                dim_feedforward=dim_feedforward,
                latent_dim=latent_dim,
                dropout=dropout,
            )
        self.register_buffer("pe_2d_4x4", sinusoidal_2d(4, 4, d_model, torch.device("cpu")), persistent=False)
        nn.init.zeros_(self.action_head[-1].bias)
        nn.init.normal_(self.action_head[-1].weight, std=0.01)

    def preprocess(self, images: torch.Tensor, augment: bool) -> torch.Tensor:
        # images: [B, T, C, H, W, 3] uint8
        b, t, c, h, w, ch = images.shape
        x = images.permute(0, 2, 1, 5, 3, 4).reshape(b * c, t, ch, h, w)  # [B*C, T, 3, H, W]
        cs = self.crop_size
        margin = h - cs
        if augment and margin > 0:
            offsets = torch.randint(0, margin + 1, (b * c, 2), device=x.device)
            yy = offsets[:, 0, None, None] + torch.arange(cs, device=x.device)[None, :, None]
            xx = offsets[:, 1, None, None] + torch.arange(cs, device=x.device)[None, None, :]
            x = x.permute(0, 3, 4, 1, 2)[torch.arange(b * c, device=x.device)[:, None, None], yy, xx].permute(0, 3, 4, 1, 2)
        else:
            half = margin // 2
            x = x[..., half : half + cs, half : half + cs]
        x = x.float() / 255.0
        if augment:
            scale = torch.empty(b * c, 1, 1, 1, 1, device=x.device).uniform_(0.96, 1.04)
            bias = torch.empty(b * c, 1, 1, 1, 1, device=x.device).uniform_(-0.02, 0.02)
            x = (x * scale + bias).clamp(0.0, 1.0)
        return x.reshape(b, c, t, ch, cs, cs)

    def forward(
        self,
        robot_xy: torch.Tensor,
        images: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        augment: bool = False,
        free_bits: float = 0.0,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, t, _ = robot_xy.shape
        x = self.preprocess(images, augment=augment)  # [B, C, T, 3, cs, cs]

        map_tok_list = []
        map_pos_list = []
        kp_tok_list = []
        kp_pos_list = []
        kp_for_vae = []
        per_frame_vis = [[] for _ in range(t)]

        for ci, enc in enumerate(self.encoders):
            cam_in = x[:, ci].reshape(b * t, 3, self.crop_size, self.crop_size)
            out1, out2, (fh, fw) = enc(cam_in)
            if self.token_mode == "keypoints":
                l3_tok = out1.reshape(b, t, self.d_model)
                l4_tok = out2.reshape(b, t, self.d_model)
                kp_for_vae.append(l4_tok)
                for ti in range(t):
                    per_frame_vis[ti].append(l3_tok[:, ti])
                    per_frame_vis[ti].append(l4_tok[:, ti])
                    cam_t_pos = self.camera_embed[ci] + self.time_embed[ti]
                    kp_tok_list.append(l3_tok[:, ti : ti + 1])
                    kp_pos_list.append((cam_t_pos.unsqueeze(0) + self.token_type_embed[2].unsqueeze(0)).expand(b, 1, -1))
                    kp_tok_list.append(l4_tok[:, ti : ti + 1])
                    kp_pos_list.append((cam_t_pos.unsqueeze(0) + self.token_type_embed[3].unsqueeze(0)).expand(b, 1, -1))
            else:
                m_tok = out1.reshape(b, t, fh * fw, self.d_model)
                k_tok = out2.reshape(b, t, self.d_model)
                kp_for_vae.append(k_tok)
                if (fh, fw) == (4, 4):
                    pe2d = self.pe_2d_4x4.to(device=m_tok.device, dtype=m_tok.dtype)
                else:
                    pe2d = sinusoidal_2d(fh, fw, self.d_model, m_tok.device).to(dtype=m_tok.dtype)
                for ti in range(t):
                    per_frame_vis[ti].append(k_tok[:, ti])
                    cam_t_pos = self.camera_embed[ci] + self.time_embed[ti]
                    map_tok_list.append(m_tok[:, ti])
                    map_pos_list.append((pe2d + cam_t_pos.unsqueeze(0) + self.token_type_embed[3].unsqueeze(0)).expand(b, -1, -1))
                    kp_tok_list.append(k_tok[:, ti : ti + 1])
                    kp_pos_list.append((cam_t_pos.unsqueeze(0) + self.token_type_embed[2].unsqueeze(0)).expand(b, 1, -1))

        kl = robot_xy.new_zeros(())
        diag: Dict[str, torch.Tensor] = {}
        if self.use_vae and actions is not None:
            assert valid is not None
            obs_summary = torch.cat([robot_xy.reshape(b, -1)] + [k.detach().reshape(b, -1) for k in kp_for_vae], dim=-1)
            z, kl, diag = self.cvae(obs_summary, actions, valid, free_bits=free_bits)
            z = z.to(dtype=robot_xy.dtype)
        else:
            z = robot_xy.new_zeros((b, self.latent_dim))

        z_tok = self.z_proj(z).unsqueeze(1)
        z_pos = self.token_type_embed[0].unsqueeze(0).expand(b, 1, -1)

        r_tok = self.robot_proj(robot_xy)
        r_pos = torch.cat(
            [(self.time_embed[ti].unsqueeze(0) + self.token_type_embed[1].unsqueeze(0)).expand(b, 1, -1) for ti in range(t)],
            dim=1,
        )

        src = torch.cat([z_tok, r_tok] + kp_tok_list + map_tok_list, dim=1)
        src_pos = torch.cat([z_pos, r_pos] + kp_pos_list + map_pos_list, dim=1)

        if return_aux and self.aux_geom_dim > 0:
            dec_out, memory = self.transformer(
                src=src,
                src_pos=src_pos,
                query_pos=self.query_pos.expand(b, -1, -1),
                return_memory=True,
            )
            vis_stack = torch.stack([torch.cat(per_frame_vis[ti], dim=-1) for ti in range(t)], dim=1)  # [B, T, vis_in]
            diag["aux_vis"] = self.vis_aux_head(vis_stack)  # [B, T, 8]
            # Pool encoder memory (mean + latest robot token) for full 14-d SE(2) geometry
            enc_feat = torch.cat([memory[:, 1:].mean(dim=1), memory[:, t]], dim=-1)  # [B, 2*D]
            diag["aux_enc"] = self.enc_aux_head(enc_feat)  # [B, aux_geom_dim]
        else:
            dec_out = self.transformer(
                src=src,
                src_pos=src_pos,
                query_pos=self.query_pos.expand(b, -1, -1),
            )
        pred_actions = self.action_head(dec_out)
        return pred_actions, kl, diag


def weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    time_decay: float = 0.08,
) -> torch.Tensor:
    """Front-weighted masked L1 loss across the predicted action chunk.

    pred, target: [B, K, A]; valid: [B, K] bool.
    """
    k = pred.shape[1]
    if time_decay > 0.0:
        w = torch.exp(-time_decay * torch.arange(k, device=pred.device, dtype=torch.float32))
        w = w / w.mean()
    else:
        w = torch.ones(k, device=pred.device, dtype=torch.float32)
    mask = valid.float() * w.unsqueeze(0)  # [B, K]
    l1 = (pred.float() - target.float()).abs().mean(dim=-1)  # [B, K]
    return (l1 * mask).sum() / mask.sum().clamp(min=1.0)
