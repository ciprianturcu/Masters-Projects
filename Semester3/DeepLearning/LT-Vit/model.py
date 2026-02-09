import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LTViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop_p = drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.norm1_img = nn.LayerNorm(dim)
        self.norm1_lbl = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, img_tokens, lbl_tokens):
        B, N, D = img_tokens.shape
        C = lbl_tokens.shape[1]
        H = self.num_heads
        hd = self.head_dim
        drop_p = self.attn_drop_p if self.training else 0.0

        img_normed = self.norm1_img(img_tokens)
        lbl_normed = self.norm1_lbl(lbl_tokens)

        all_tokens = torch.cat([img_normed, lbl_normed], dim=1)
        qkv = self.qkv(all_tokens)
        qkv = qkv.reshape(B, N + C, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q_img = q[:, :, :N, :]
        q_lbl = q[:, :, N:, :]
        k_img = k[:, :, :N, :]
        v_img = v[:, :, :N, :]

        img_out = F.scaled_dot_product_attention(
            q_img, k_img, v_img, dropout_p=drop_p,
        ).transpose(1, 2).reshape(B, N, D)

        lbl_out = F.scaled_dot_product_attention(
            q_lbl, k, v, dropout_p=drop_p,
        ).transpose(1, 2).reshape(B, C, D)

        img_out = img_tokens + self.proj_drop(self.proj(img_out))
        lbl_out = lbl_tokens + self.proj_drop(self.proj(lbl_out))

        all_out = torch.cat([img_out, lbl_out], dim=1)
        all_out = all_out + self.ffn(self.norm2(all_out))

        img_tokens_out = all_out[:, :N, :]
        lbl_tokens_out = all_out[:, N:, :]

        return img_tokens_out, lbl_tokens_out


class LTViT(nn.Module):

    def __init__(self, num_labels=14, n2=4, pretrained=True, image_size=224, drop=0.0):
        super().__init__()
        self.num_labels = num_labels
        self.n2 = n2

        self.backbone = timm.create_model(
            "vit_small_patch16_224.dino",
            pretrained=pretrained,
            img_size=image_size,
            num_classes=0,
        )

        self.embed_dim = self.backbone.embed_dim
        total_depth = len(self.backbone.blocks)
        self.n1 = total_depth - n2

        self.vit_blocks = nn.ModuleList(self.backbone.blocks[:self.n1])
        num_heads = self.backbone.blocks[0].attn.num_heads

        self.lt_blocks = nn.ModuleList([
            LTViTBlock(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                drop=drop,
            )
            for _ in range(n2)
        ])

        self.label_tokens = nn.Parameter(
            torch.randn(1, num_labels, self.embed_dim) * 0.02
        )

        self.norm = self.backbone.norm

        self.classifier = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        B = x.shape[0]

        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        for block in self.vit_blocks:
            x = block(x)

        lbl = self.label_tokens.expand(B, -1, -1)
        for lt_block in self.lt_blocks:
            x, lbl = lt_block(x, lbl)

        lbl = self.norm(lbl)

        logits = self.classifier(lbl).squeeze(-1)
        return logits
