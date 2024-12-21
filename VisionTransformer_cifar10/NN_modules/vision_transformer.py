import torch
import torch.nn as nn
import numpy as np

from .transformer_modules import ImagePatchEmbedding, ViTEncoder, ViTBlock
from .sinusoidal_positional_encoding import get_2d_sincons_pos_embed

class VisionTransformer(nn.Module):
    def __init__(self,image_size=224,patch_size=16,in_channel=3,dim=768,hidden_dim=768*4,
                num_heads=12,activation=nn.GELU(),num_blocks=8,qkv_bias=True,dropout=0.,
                quiet_attention=False,num_classes=1000):
        """
        Args:
            image_size (Union[int, tuple]): 画像の高さと幅
            patch_size (int): 1画像パッチのピクセル幅
            in_channel (int): 入力画像のチャネル数
            dim (int): 埋め込み次元数
            hidden_dim (int): FeedForward Networkの隠れ層次元数
            num_heads (int): MultiHeadAttentionのHead数
            activation (torch.nn.modules.activation): pytorchの活性化関数
            num_blocks (int): ブロックの数
            qkv_bias (bool): MultiHeadAttentionの埋め込み全結合層にbiasを付けるかどうか
            dropout (float): ドロップアウト確率
            quiet_attention (bool): Trueの場合、softmaxの分母に1を足す
            num_classes (int): 分類ヘッドの数
        """
        super().__init__()
        
        # input layer
        self.patch_embedding = ImagePatchEmbedding(image_size,patch_size,in_channel,dim)
        num_patches = self.patch_embedding.num_patches
        self.cls_token = nn.Parameter(torch.randn(size=(1,1,dim)))
        self.positional_embedding = nn.Parameter(torch.randn(size=(1,num_patches+1,dim)))
        
        # vit encoder 
        self.encoder = ViTEncoder(dim,hidden_dim,num_heads,activation,qkv_bias,dropout,num_blocks,quiet_attention)
        
        # mlp head
        self.ln = nn.LayerNorm(dim,eps=1e-10)
        self.head = nn.Linear(dim,num_classes)
    
    def forward(self,x):
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat((cls_token,x),dim=1) # (B,num_patches+1,embedding_dim)
        x = x + self.positional_embedding
        
        x = self.encoder(x)
        
        x = torch.index_select(x,1,torch.tensor(0,device=x.device))
        x = x.squeeze(1)
        x = self.ln(x)
        out = self.head(x)
        
        return out