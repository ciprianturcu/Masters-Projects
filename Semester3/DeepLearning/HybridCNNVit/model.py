import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, pool=True):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNBackbone(nn.Module):

    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 384]

        layers = []
        in_ch = 3
        for out_ch in channels:
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop_p = drop
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HybridCNNViT(nn.Module):

    def __init__(
        self,
        num_labels=14,
        cnn_channels=None,
        embed_dim=384,
        trans_depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        drop=0.0,
        image_size=224,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128, 256, embed_dim]
        cnn_channels = list(cnn_channels)
        cnn_channels[-1] = embed_dim

        self.num_labels = num_labels
        self.embed_dim = embed_dim

        self.cnn = CNNBackbone(channels=cnn_channels)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat = self.cnn(dummy)
            _, _, H, W = feat.shape
        self.num_patches = H * W

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + self.num_patches, embed_dim) * 0.02
        )

        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(trans_depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim, num_labels),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        feat = self.cnn(x)
        tokens = feat.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        tokens = tokens + self.pos_embed

        for block in self.transformer:
            tokens = block(tokens)

        cls_out = self.norm(tokens[:, 0])
        logits = self.head(cls_out)
        return logits


if __name__ == "__main__":
    model = HybridCNNViT(num_labels=14)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    print(f"Output shape: {logits.shape}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
