"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

scale_gene = 80 
scale_wsi = 80
scale_cross = 80 


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=768, norm_layer=None):
        super().__init__()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        return x


# yanrui
class EmbedReduction(nn.Module):
    """
    Pathway to GeneSet Embedding
    """

    def __init__(self, in_features, hidden_features=None, out_features=1280, act_layer=nn.GELU, drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.norm1 = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm2 = norm_layer(out_features)

    # def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
    #     super().__init__()
    #     img_size = (img_size, img_size)
    #     patch_size = (patch_size, patch_size)
    #     self.img_size = img_size
    #     self.patch_size = patch_size
    #     self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
    #     self.num_patches = self.grid_size[0] * self.grid_size[1]
    #
    #     self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
    #     self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        # import pdb;pdb.set_trace()
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        #x = self.drop(x)
        return x

class Attention_GENE(nn.Module):
    def __init__(self,
                 dim,  #
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 norm_layer=nn.LayerNorm):
        super(Attention_GENE, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = norm_layer(dim)#yanrui
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # import pdb;pdb.set_trace()
        x = self.norm(x)#yanrui
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #print("-------GENE-----------------------------query--------query--------query--------query--------query-------------------------")
        #print(q[0,0,0,:])
        #print("--------------GENE-----------------------key-------------key--------key--------key----------------------------------")
        #print(k[0,0,0,:])
        # print("------------------------------------value---------------------------------------")
        # print(v)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale * scale_gene
        #print("-------GENE-----------------------------attn---------------------------------------")
        #print("scale_gene-----",scale_gene)
        #print(attn[0,0,0,:])
        attn = attn.softmax(dim=-1)
        #print("--------GENE--------------------softmax--------attn---------------------------------------")
        #print(attn[0,0,0,:])
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_WSI(nn.Module):
    def __init__(self,
                 dim,  #
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 norm_layer=nn.LayerNorm):
        super(Attention_WSI, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = norm_layer(dim)#yanrui
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # import pdb;pdb.set_trace()
        x = self.norm(x)#yanrui
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #print("-------WSI-----------------------------query--------query--------query--------query--------query-----------------------------")
        #print(q)
        # print("------------------------------key-------------key--------key--------key----------------------------------")
        # print(k)
        # print("------------------------------------value---------------------------------------")
        # print(v)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale * scale_wsi
        #print("--------WSI----------------------------attn---------------------------------------")
        #print("scale_wsi-----",scale_wsi)
        #print(attn[0,0,0,:])
        attn = attn.softmax(dim=-1)
        #print("--------WSI--------------------softmax--------attn---------------------------------------")
        #print(attn[0,0,0,:])
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# zxy add
def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def exists(val):
    return val is not None


class Gene_Guided_Transformer_Fusion(nn.Module):
    def __init__(self,
                 num_patches=186,
                 dim=256,  # 输入token的dim
                 num_heads=16,
                 # qkv_bias=False,
                 q_bias=False,
                 kv_bias=False,
                 qk_scale=None,
                 sparse_topk=8,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Gene_Guided_Transformer_Fusion, self).__init__()
        self.num_heads = num_heads  # 多头的数目
        head_dim = dim // num_heads  # 多头
        self.scale = qk_scale or head_dim ** -0.5  # 根号d
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)   #全连接层实现qkv
        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)  # 拼接
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.sparse_topk = sparse_topk
        self.gap = nn.AdaptiveAvgPool2d((num_patches, 1))

    def forward(self, x1, x2):
        # import pdb;pdb.set_trace()
        # x1 is wsi: batch * 500 * 256; x2 is gene: batch*186*256
        # [batch_size, num_patches + 1, total_embed_dim]
        assert x1.shape[-1] == x2.shape[-1]
        B, M, C = x1.shape
        B, N, C = x2.shape
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # q: [batch_size, 186, 256] -> [batch, 16, 186, 16]
        q = self.q(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x1).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q.shape = v.shape: [batch, 500, 256] -> [batch, 16, 500, 16]
        k, v = kv[0], kv[1]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        # attn: [batch, 16, 186, 500]
        # import pdb;pdb.set_trace()
        attn_pre = (q @ k.transpose(-2, -1)) * self.scale * scale_cross
        #print("--------CROSS----------------------------attn---------------------------------------")
        #print("scale_cross-----",scale_cross)
        #print(attn_pre[0,0,0,:])

        attn = attn_pre.softmax(dim=-1)
        #print("--------CROSS-----------------softmax-----------attn---------------------------------------------------")
        #print(attn[0,0,0,:])
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        # x: []
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.gap(x)
        return x, attn


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block_GENE(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block_GENE, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_GENE(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block_WSI(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block_WSI, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_WSI(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, wsi_patches=500, gene_patches=186, embed_wsi_dim=384, embed_gene_dim=256, num_classes=1000,
                 depth_gene=3, depth_wsi=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(VisionTransformer, self).__init__()
        self.wsi_patches = wsi_patches
        self.gene_patches = gene_patches
        self.embed_wsi_dim = embed_wsi_dim
        self.embed_gene_dim = embed_gene_dim
        self.num_classes = num_classes
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.patch_embed = embed_layer(embed_dim=embed_wsi_dim)

        self.gene_embed = EmbedReduction(in_features=5245, hidden_features=256, out_features=self.embed_gene_dim,
                                         act_layer=act_layer, drop=drop_ratio)

        self.pos_wsi_embed = nn.Parameter(torch.zeros(1, self.wsi_patches, self.embed_wsi_dim))
        self.pos_gene_embed = nn.Parameter(torch.zeros(1, self.gene_patches, self.embed_gene_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_wsi)]  # stochastic depth decay rule
        self.blocks_wsi = nn.Sequential(*[
            Block_WSI(dim=self.embed_wsi_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_wsi)
        ])

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth_gene)]  # stochastic depth decay rule
        self.blocks_gene = nn.Sequential(*[
            Block_GENE(dim=self.embed_gene_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth_gene)
        ])

        self.norm_wsi = norm_layer(self.embed_wsi_dim)
        self.norm_gene = norm_layer(self.embed_gene_dim)

        self.wsi_embed_reduction = EmbedReduction(in_features=self.embed_wsi_dim, hidden_features=256,
                                                  out_features=self.embed_gene_dim, act_layer=act_layer,
                                                  drop=drop_ratio)
        self.gene_guided_wsi_fusion = Gene_Guided_Transformer_Fusion(num_patches=self.gene_patches,
                                                                     dim=self.embed_gene_dim)
        self.gap_gene = nn.AdaptiveAvgPool2d((self.gene_patches, 1))

        # Classifier head(s)
        self.head = nn.Linear(self.gene_patches * 2, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_gene_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_wsi_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_gene_embed, std=0.02)

        self.apply(_init_vit_weights)

    def forward_features_wsi(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 500, 1280]

        # print(x)
        # print(self.pos_wsi_embed)
        x = self.pos_drop(x + self.pos_wsi_embed)
        x = self.blocks_wsi(x)
        x = self.norm_wsi(x)
        return x

    def forward_features_gene(self, x):
        x = self.pos_drop(x + self.pos_gene_embed)
        x = self.blocks_gene(x)  # depth应该浅一点，基因数据量小，可能会过拟合
        x = self.norm_gene(x)
        return x

    def forward(self, x_wsi, x_gene):
        # import pdb;pdb.set_trace()
        # 500 * 1280 -> 500 * 1280
        #print("-----------------------------input to wsi branch-------x_wsi---------------------------------------")
        #print(x_wsi)
        wsi_features = self.forward_features_wsi(x_wsi)  # Transformer Encoder for wsi branch
        #print("--------------------output of wsi branch---------------------")
        #print(wsi_features)
        # 500 * 1280 -> 500 * 256
        wsi_features_reduction = self.wsi_embed_reduction(wsi_features)
        #print("--------------------wsi_features_reduction = self.wsi_embed_reduction(wsi_features)--------------")
        #print(wsi_features_reduction)
        # 186 * 5245 -> 186* 256
        gene_features_reduction = self.gene_embed(x_gene)
        # 186 * 256 -> 186* 256
        gene_features = self.forward_features_gene(gene_features_reduction)  # Transformer Encoder for gene branch

        # 186 * 500
        # wsi_features_reduction = wsi_features_reduction.permute(0, 2, 1)
        # gene: B * 186 * 256; wsi: B * 500 * 256
        # fine-grained contrastive loss: B * 186 * 500
        # logit_scale = self.logit_scale.exp()
        # gene2wsi_feature = logit_scale * gene_features_reduction @ wsi_features_reduction.transpose(-2, -1)
        ## 186 * 2
        # top2_values, _ = torch.topk(gene2wsi_feature, k=2, dim=2)
        # top2_values = F.log_softmax(top2_values, dim=-1)
        ## top2_values = top2_values.view(gene2wsi_feature.shape[0], -1)
        # gene2wsi_loss = -torch.mean(top2_values)

        # 186 * 256
        x, attn = self.gene_guided_wsi_fusion(wsi_features_reduction, gene_features)
        gene_features_gap = self.gap_gene(gene_features)
        fused_features = torch.cat([gene_features_gap, x], dim=1)
        fused_features = fused_features.squeeze(2)
        pred_head = self.head(fused_features)  # 全连接层,最后输出维度为1 (survival risk score)

        # return pred_head
        return pred_head, attn


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def my_model(num_classes: int = 21843, has_logits: bool = True, wsi_block=12, gene_block=3, dpr=0.1):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(wsi_patches=500,
                              gene_patches=186,
                              embed_wsi_dim=384,
                              embed_gene_dim=256,
                              depth_gene=gene_block,  # depth=12,
                              depth_wsi=wsi_block,
                              num_heads=16,  # 1280/20=64，多头可以整除
                              representation_size=2048 if has_logits else None,
                              drop_path_ratio=dpr,
                              drop_ratio=dpr,
                              attn_drop_ratio=dpr,
                              num_classes=num_classes)
    return model
