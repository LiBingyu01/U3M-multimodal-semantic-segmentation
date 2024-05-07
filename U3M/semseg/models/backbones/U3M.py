import torch
from torch import nn, Tensor

from torch.nn import functional as F
from semseg.models.layers import DropPath
import torch.nn.init as init
# -------------------- ChannelAttentionBlock
class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16):

        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
        # Initialize linear layers with Kaiming initialization
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x * y.expand_as(x)).flatten(2).transpose(1, 2)

# -------------------- CustomDWConv 
class CustomDWConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        # print(x.shape)
        x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

# -------------------- CustomDWConv_pooling 
class CustomDWConv_pooling(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        # print(x.shape)
        x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape)
        x = self.dwconv(x)
        return x

# -------------------- CustomPWConv
class CustomPWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        # Initialize pwconv layer with Kaiming initialization
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape)
        x = self.bn(self.pwconv(x))
        # print(x.shape)
        return x.flatten(2).transpose(1, 2)

class MultiScaleFusion_Conv(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFusion_Conv, self).__init__()
        self.conv1 = CustomDWConv(in_channels, 3)
        self.conv2 = CustomDWConv(in_channels, 5)
        self.conv3 = CustomDWConv(in_channels, 7)
        self.PWconv_conv_in = CustomPWConv(in_channels)
        self.PWconv_conv_out = CustomPWConv(in_channels)
        # self.FeedForwardNN = FeedForwardNN(in_channels, in_channels*2)
    def forward(self, x: Tensor, H, W) -> Tensor:
        
        B, _, C = x.shape
        # 卷积融合
        x = self.PWconv_conv_in(x,H,W)
        
        # Parallel convolution
        out1 = self.conv1(x,H,W)
        out2 = self.conv2(x,H,W)
        out3 = self.conv3(x,H,W)
        
        # res
        x_out = out1 + out2 + out3 + x
        x_out = self.PWconv_conv_out(x_out,H,W)
        return x_out

# [pool1,pool2,pool3,pool4]分别进行池化之后加和
# 使用1*1卷积进行特征融合
class MultiScaleFusion_Pooling(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(MultiScaleFusion_Pooling, self).__init__()

        self.pool_sizes = pool_sizes
        
        self.pool_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for size in pool_sizes:
            self.pool_layers.append(nn.AdaptiveAvgPool2d(output_size=(size,size)))
            self.conv_layers.append(CustomDWConv_pooling(dim=in_channels, kernel=1))
            
        self.PWconv_pool_in = CustomPWConv(in_channels)
        self.PWconv_pool_out = CustomPWConv(in_channels)

    def forward(self, x: Tensor, H, W) -> Tensor:
        
        B, _, C = x.shape
        x_res = x
        x = self.PWconv_pool_in(x,H,W)
        # Parallel pooling
        x_in = x.transpose(1, 2).view(B, C, H, W)
        x_add = 0
        for i in range(len(self.pool_layers)):
            pooled_feature = self.conv_layers[i](self.pool_layers[i](x_in).flatten(2).transpose(1, 2),self.pool_sizes[i],self.pool_sizes[i])
            x2 = F.interpolate(pooled_feature, size=(H,W), mode='bilinear')
            x_add += x2
        x_out = x_add.flatten(2).transpose(1, 2)

        # 输出
        x_out = self.PWconv_pool_out(x_out,H,W)
        x_out = x_res + x_out
        return x_out
# ------------------------------------------------ 
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)  
    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))

class FusionBlock(nn.Module):
    def __init__(self, channels, reduction=16, num_modals=2):
        super(FusionBlock, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.num_modals = num_modals

        self.liner_fusion_layers = nn.ModuleList([
            nn.Linear(self.channels[0]*self.num_modals, self.channels[0]),
            nn.Linear(self.channels[1]*self.num_modals, self.channels[1]),
            nn.Linear(self.channels[2]*self.num_modals, self.channels[2]),
            nn.Linear(self.channels[3]*self.num_modals, self.channels[3]),
        ])

        self.MultiScaleFusion_Conv = nn.ModuleList([
            MultiScaleFusion_Conv(self.channels[0]),
            MultiScaleFusion_Conv(self.channels[1]),
            MultiScaleFusion_Conv(self.channels[2]),
            MultiScaleFusion_Conv(self.channels[3]),
        ])

        self.MultiScaleFusion_Pooling = nn.ModuleList([
            MultiScaleFusion_Pooling(self.channels[0], (1, 2, 3, 6)),
            MultiScaleFusion_Pooling(self.channels[1], (1, 2, 3, 6)),
            MultiScaleFusion_Pooling(self.channels[2], (1, 2, 3, 6)),
            MultiScaleFusion_Pooling(self.channels[3], (1, 2, 3, 6)),
        ])


        self.ChannelAttentionBlock = nn.ModuleList([
            ChannelAttentionBlock(self.channels[0]),
            ChannelAttentionBlock(self.channels[1]),
            ChannelAttentionBlock(self.channels[2]),
            ChannelAttentionBlock(self.channels[3]),
        ])

        self.liner_fusion_layers_out = nn.ModuleList([
            nn.Linear(self.channels[0], self.channels[0]),
            nn.Linear(self.channels[1], self.channels[1]),
            nn.Linear(self.channels[2], self.channels[2]),
            nn.Linear(self.channels[3], self.channels[3]),
        ])

        # Initialize linear fusion layers with Kaiming initialization
        for linear_layer in self.liner_fusion_layers:
            init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, layer_idx):
        B, C, H, W = x[0].shape
        # print(x[0].shape) #[B,C,H,W]
        x = torch.cat(x, dim=1)#[B,2*C,H,W]

        x = x.flatten(2).transpose(1, 2) # [B,H*W,2C]

        # 多模态融合
        x_sum = self.liner_fusion_layers[layer_idx](x) # [B,H*W,C]
        
        # 特征提取模块
        x_pooling = self.MultiScaleFusion_Pooling[layer_idx](x_sum,H,W)
        x_conv = self.MultiScaleFusion_Conv[layer_idx](x_sum,H,W)
        # print(x_sum.shape)
        # print(x_pooling.shape)
        
        # 聚合
        x_fusion = x_pooling + x_conv
        x_fusion = F.gelu(self.liner_fusion_layers_out[layer_idx](x_fusion))
        # 通道注意力机制
        x_fusion = self.ChannelAttentionBlock[layer_idx](x_fusion,H,W)
    
        x_fusion = x_fusion.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_pooling = x_pooling.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x_conv = x_conv.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x_fusion, x_pooling, x_conv


class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)    # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., is_fan=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4)) if not is_fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim*4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1,-2)
        _, _, Nk, Ck  = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))
        
        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x 

mit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}

class MixTransformer(nn.Module):
    def __init__(self, model_name: str = 'B0', modality: str = 'depth'):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(cmnext_settings.keys())}"
        # self.model_name = 'B2'
        self.model_name = model_name
        # TODO: Must comment the following line later
        # self.model_name = 'B2' if modality == 'depth' else model_name
        embed_dims, depths = mit_settings[self.model_name] 
        self.modality = modality  
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7//2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3//2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3//2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3//2)
   
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        # Initialize with pretrained weights
        self.init_weights()

    def init_weights(self):
        print(f"Initializing weight for {self.modality}...")
        print("-----"*3)
        print(f'xxx/u3M/checkpoints/pretrained/segformers/mit_{self.model_name.lower()}.pth')
        checkpoint = torch.load(f'xxx/U3M/checkpoints/pretrained/segformers/mit_{self.model_name.lower()}.pth', map_location=torch.device('cpu'))
        
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        msg = self.load_state_dict(checkpoint, strict=False)

        del checkpoint
        print(f"Weight init complete with message: {msg}")
     
    def forward(self, x: Tensor) -> list:
        x_cam = x       
        
        B = x_cam.shape[0]
        outs = []

        # stage 1
        x_cam, H, W = self.patch_embed1(x_cam)
        # print(x_cam.shape)
        for blk in self.block1:
            x_cam = blk(x_cam, H, W)
        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x1_cam)

        # stage 2  

        x_cam, H, W = self.patch_embed2(x1_cam)
        for blk in self.block2:
            x_cam = blk(x_cam, H, W)
        x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)
        for blk in self.block3:
            x_cam = blk(x_cam, H, W)
        x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x_cam = blk(x_cam, H, W)
        x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x4_cam)

        return outs

class U3M(nn.Module):
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(cmnext_settings.keys())}"
        embed_dims, depths = mit_settings[model_name]
        self.modals = modals[1:] if len(modals)>1 else []  
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7//2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3//2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3//2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3//2)
   
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        
        # Have extra modality
        print("num_modals",self.num_modals)
        if self.num_modals > 0:
            # Backbones and Fusion Block for extra modalities
            self.extra_mit = nn.ModuleList([MixTransformer(model_name, self.modals[i]) for i in range(self.num_modals)])
            self.fusion_block = FusionBlock(self.channels, reduction=16, num_modals=self.num_modals+1)


    def forward(self, x: list) -> list:
        x_cam = x[0]        
        if self.num_modals > 0:
            x_ext = x[1:]
        B = x_cam.shape[0]

        if self.num_modals > 0: 
            outs = []
            outs_pooling = []
            outs_conv = []
            outs_se = []
        else:
            outs = []
            outs_pooling = []
            outs_conv = []
            outs_se = []

        # stage 1
        # print("x_cam.shape:",x_cam.shape)
        x_cam, H, W = self.patch_embed1(x_cam)
        # print("x_cam.shape:",x_cam.shape)
        for blk in self.block1:
            x_cam = blk(x_cam, H, W)
            
            # print("xw_cam.shape:",x_cam.shape)

        x1_cam = self.norm1(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # print("x1_cam.shape:",x1_cam.shape)
        # Extra Modalities
        if self.num_modals > 0:

            for i in range(self.num_modals):
                x_ext[i], _, _ = self.extra_mit[i].patch_embed1(x_ext[i])
                for blk in self.extra_mit[i].block1:
                    x_ext[i] = blk(x_ext[i], H, W)

                x_ext[i] = self.extra_mit[i].norm1(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)

            # print("x_ext.shape:",torch.cat([*x_ext], dim=1).shape)
            x_fused, x_pooling, x_conv = self.fusion_block([x1_cam, *x_ext], layer_idx=0)
            # print("x_fused.shape:",x_fused.shape)

            outs.append(x_fused)
            outs_pooling.append(x_pooling)
            outs_conv.append(x_conv)
        else:
            outs.append(x1_cam)

        # stage 2
        x_cam, H, W = self.patch_embed2(x1_cam)
        for blk in self.block2:
            x_cam = blk(x_cam, H, W)
       
        x2_cam = self.norm2(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.extra_mit[i].patch_embed2(x_ext[i])
                for blk in self.extra_mit[i].block2:
                    x_ext[i] = blk(x_ext[i], H, W)
                x_ext[i] = self.extra_mit[i].norm2(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused, x_pooling, x_conv = self.fusion_block([x2_cam, *x_ext], layer_idx=1)
            outs.append(x_fused)
            outs_pooling.append(x_pooling)
            outs_conv.append(x_conv)
        else:
            outs.append(x2_cam)

        # stage 3
        x_cam, H, W = self.patch_embed3(x2_cam)
        for blk in self.block3:
            x_cam = blk(x_cam, H, W)
        
        x3_cam = self.norm3(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.extra_mit[i].patch_embed3(x_ext[i])
                for blk in self.extra_mit[i].block3:
                    x_ext[i] = blk(x_ext[i], H, W)
                x_ext[i] = self.extra_mit[i].norm3(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused, x_pooling, x_conv = self.fusion_block([x3_cam, *x_ext], layer_idx=2)
            outs.append(x_fused)
            outs_pooling.append(x_pooling)
            outs_conv.append(x_conv)
        else:
            outs.append(x3_cam)

        # stage 4
        x_cam, H, W = self.patch_embed4(x3_cam)
        for blk in self.block4:
            x_cam = blk(x_cam, H, W)

        x4_cam = self.norm4(x_cam).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # Extra Modalities
        if self.num_modals > 0:
            for i in range(self.num_modals):
                x_ext[i], _, _ = self.extra_mit[i].patch_embed4(x_ext[i])
                for blk in self.extra_mit[i].block4:
                    x_ext[i] = blk(x_ext[i], H, W)
                x_ext[i] = self.extra_mit[i].norm4(x_ext[i]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_fused, x_pooling, x_conv = self.fusion_block([x4_cam, *x_ext], layer_idx=3)
            
            outs.append(x_fused)
            outs_pooling.append(x_pooling)
            outs_conv.append(x_conv)
        else:
            outs.append(x4_cam)

    # --------------------------- 不同尺度的信息融合 ---------------------------

        return outs, outs_pooling, outs_conv


if __name__ == '__main__':
    modals = ['img', 'aolp', 'dolp', 'nir']
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    model = U3M('B2', modals)
    outs = model(x)
    for y in outs:
        print(y.shape)

