import torch
import torch.nn as nn


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                        nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        attention_mask = self.attention(x)
        features = self.conv(x)
        return torch.mul(features, attention_mask)



class Distillation(nn.Module):
    def __init__(self, args):
        super(Distillation, self).__init__()

        channels = args.feat_dim
        self.attention = {}
        for supervision in args.supervision:
            name = supervision['name']
            self.attention[name] = SABlock(channels, channels)
        self.attention = nn.ModuleDict(self.attention)


    def forward(self, feature_map_dict):
        adapters = []
        for k, v in feature_map_dict.items():
            if k == 'classify':
                adapters.append(v)
                continue
            adapters.append(self.attention[k](v))
        out = torch.sum(torch.stack(adapters), dim=0)
        return out
