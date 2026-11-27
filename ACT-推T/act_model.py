"""Compact deterministic action-chunking transformer with optional RGB tokens."""
import torch
from torch import nn
from pathlib import Path

class ACT(nn.Module):
    def __init__(self,vision=False,chunk=16,width=256,state_dim=10,dropout=.1):
        super().__init__()
        self.vision=vision;self.chunk=chunk
        self.state=nn.Linear(2 if vision else state_dim,width)
        self.queries=nn.Parameter(torch.randn(1,chunk,width)*.02)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,8,1024,dropout,activation='gelu',batch_first=True),3,enable_nested_tensor=False)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(width,8,1024,dropout,activation='gelu',batch_first=True),3)
        self.output=nn.Linear(width,2)
        if vision:
            from torchvision.models import resnet18
            from torchvision.ops.misc import FrozenBatchNorm2d
            backbone=resnet18(weights=None,norm_layer=FrozenBatchNorm2d)
            weights=Path(__file__).resolve().parent/'resnet18-f37072fd.pth'
            if not weights.exists():raise RuntimeError('Official ResNet18 weights are missing')
            backbone.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
            self.backbone=nn.Sequential(*list(backbone.children())[:-2])
            self.visual=nn.Conv2d(512,width,1)
            self.visual_position=nn.Parameter(torch.randn(1,16,width)*.02)
            self.register_buffer('image_mean',torch.tensor([.485,.456,.406])[None,:,None,None])
            self.register_buffer('image_std',torch.tensor([.229,.224,.225])[None,:,None,None])
    def forward(self,state,image=None):
        tokens=self.state(state[:,:2] if self.vision else state).unsqueeze(1)
        if self.vision:
            x=(image.float()/255-self.image_mean)/self.image_std
            features=self.visual(self.backbone(x)).flatten(2).transpose(1,2)
            tokens=torch.cat([tokens,features+self.visual_position],1)
        memory=self.encoder(tokens)
        return self.output(self.decoder(self.queries.expand(len(state),-1,-1),memory))
