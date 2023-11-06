import torch
import torch.nn as nn

from SpatioTemporalCrossAttention import SpatioTemporalCrossAttention

from rich import print

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

class MLP(nn.Module):
  
    def __init__(
        self, 
        in_features, out_features
    ):
        super().__init__()

        self.fc_1 = nn.Linear(in_features, out_features)
        self.act_1 = nn.GELU()
        self.fc_2 = nn.Linear(out_features, out_features)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
            
            
    def forward(self, x):
        x = self.fc_1(x)

        x = self.act_1(x)

        x = self.fc_2(x)

        return(x)
    
    




