import torch
import torch.nn as nn
from SpatioTemporalCrossAttention import SpatioTemporalCrossAttention

def conv_init(conv):
    """
    Initializes the weights of a convolutional layer using the Kaiming normal initialization method.

    Parameters:
        conv (nn.Module): The convolutional layer to be initialized.

    Returns:
        None
    """
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    """
    Initializes a batch normalization layer.

    Args:
        bn (torch.nn.BatchNorm2d): The batch normalization layer to initialize.
        scale (float): The scale factor for the weights.

    Returns:
        None
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    """
    Initialize the weight and bias parameters of a fully connected layer.

    Parameters:
        fc (nn.Module): The fully connected layer to initialize.

    Returns:
        None
    """
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

class MLP(nn.Module):
  
    def __init__(
        self, 
        in_features, out_features
    ):
        """
        Initializes the neural network layers and parameters.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.

        Returns:
            None
        """
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
        """
        Performs a forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor to the neural network.

        Returns:
            torch.Tensor: The output tensor of the neural network.
        """
        x = self.fc_1(x)

        x = self.act_1(x)

        x = self.fc_2(x)

        return(x)
    
    




