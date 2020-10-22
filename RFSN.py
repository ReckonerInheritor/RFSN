from model import common

import torch
import torch.nn.functional as F
import torch.nn as nn


def make_model(args, parent=False):
    return RFSN(args)

class ResBlock(nn.Module):
    def __init__(self,n_feats):
        super(resblock, self).__init__()
        self.body=nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1),nn.ReLU(inplace=True),nn.Conv2d(n_feats, n_feats,3,1,1))
    def forward(self, x):
        out=self.body(x)+x
        return out

class LA(nn.Module):
    def __init__(self, channels,n_grou,reduction=4):
        M=n_grou
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels * M, 1)
        )

    def forward(self, groups):
        x=groups[0]
        feats = sum(groups)
        att = self.att(feats)
        att = att.view(x.size(0), len(groups), x.size(1))
        att = F.softmax(att, dim=1)
        att = att.view(x.size(0), -1, 1, 1)
        att = torch.split(att, x.size(1), dim=1)
        return sum([a * s for a, s in zip(att, groups)])



class LCA(nn.Module):
    def __init__(self,k_size=3):
        super(LCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class HRCAB(nn.Module):
    def __init__(self, n_feat):
        super(HRCAB, self).__init__()
        self.con3x3=nn.Conv2d(n_feat,n_feat,3,1,1)
        self.re0=nn.ReLU(inplace=True)
        self.ca=LCA()
        self.con1x1=nn.Sequential(nn.Conv2d(n_feat,n_feat,1,1,0),nn.ReLU(inplace=True))
    def forward(self, x):
        c3=self.con3x3(x)
        ca=self.ca(c3)
        ca=self.re0(ca)
        out=self.con1x1(ca)
        return x+out,c3

## Residual Group (RG)
class RFSG(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(RFSG, self).__init__()
        modules_body = [HRCAB(n_feat) for _ in range(n_resblocks)]
        self.con1x1=nn.Conv2d(n_feat*(n_resblocks+1),n_feat,1,1,0)
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        ori=x
        gr=[]
        for i,g in enumerate(self.body):
            x,c=g(x)
            gr.append(c)
        gr.append(x)
        out=torch.cat(gr,dim=1)
        out=self.con1x1(out)
        out=ori+out
        return out

## Residual Channel Attention Network (RCAN)
class RFSN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RFSN, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n=4
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        modules_body = [ RFSG ( n_feat=n_feats,n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        rs=[ResBlock(n_feats=n_feats) for i in range(n)]
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.LA=LA(channels=n_feats,n_grou=n_resgroups)
        self.con=conv(n_feats, n_feats, kernel_size)
        self.rs=nn.Sequential(*rs)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        ori = x
        groups = []
        for _, g in enumerate(self.body):
            x = g(x)
            groups.append(x)
        res = self.LA(groups)
        res=self.con(res)
        res += ori
        res=self.rs(res)
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
