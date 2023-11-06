import torch.nn as nn 

def conv_init(conv):
  nn.init.kaiming_normal_(conv.weight, mode='fan_out')
  # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
  nn.init.constant_(bn.weight, scale)
  nn.init.constant_(bn.bias, 0)

def ln_init(ln, scale):
  bn_init(ln, scale)

def fc_init(fc):
  nn.init.xavier_normal_(fc.weight)
  nn.init.constant_(fc.bias, 0)

def init_model_modules(modules):
  
  for m in modules:
    if isinstance(m, nn.Conv2d):
      conv_init(m)
    
    elif isinstance(m, nn.BatchNorm2d):
      bn_init(m, 1)
    
    elif isinstance(m, nn.Linear):
      fc_init(m)

  return modules