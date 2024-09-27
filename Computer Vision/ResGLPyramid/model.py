import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from einops import rearrange
from global_local_attention_module_pytorch import GLAM  #https://github.com/LinkAnJarad/global_local_attention_module_pytorch

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, kernels_per_layer=1):
      super(depthwise_separable_conv, self).__init__()
      self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=1, groups=nin)
      self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
      out = self.depthwise(x)
      out = self.pointwise(out)
      return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x is local feature
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)') # global feat
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class MobileViTBlock(nn.Module):
    def __init__(self, dim, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = depthwise_separable_conv(channel,channel,kernel_size)

        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth= 2, heads=4, dim_head=8, mlp_dim=mlp_dim, dropout= dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = depthwise_separable_conv(channel+dim, channel, kernel_size)

    def forward(self, x):
        r = x.clone()

        # Local representations
        x = self.conv1(x)

        x = self.conv2(x)
        y = x.clone()
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        x +=r
        return x
    
class TCT_Module(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(TCT_Module, self).__init__()
        self.conv1 =depthwise_separable_conv(inplanes,planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*4, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*4)
        self.conv3 = nn.Conv2d(planes*4, planes, kernel_size=1, bias=False) # 4 --> 1
        self.bn3 = nn.BatchNorm2d(planes) # 4 -->1
        self.gelu = nn.GELU()
        self.Vitblock = MobileViTBlock(dim=planes , channel=planes, kernel_size=3, patch_size=(2, 2), mlp_dim=planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # Local Representation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)

        # MobileVit block -- Global Representation
        out = self.Vitblock(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.gelu(out)

        return out


class backbone(nn.Module):

    def __init__(self, block, layers,num_classes=2):  # 1000 --> 2
        self.inplanes = 64
        super(backbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.glam_attention3 = GLAM(in_channels=256, num_reduced_channels=128, feature_map_size=8, kernel_size=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 , num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.glam_attention3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return x


def ResGLPyramid(**kwargs):

    model = backbone(TCT_Module, [2, 2, 3], **kwargs) # basic block --> Bottleneck

    return model

def main():

    # Instantiate the model
    model = ResGLPyramid()
    # Generate random input data
    # Random tensor of shape (batch_size=20, channels=1, height=128, width=128)
    random_input = torch.rand(20, 1, 128, 128)

    # Pass random input through the model
    output = model(random_input)

    # Print the output shape
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
