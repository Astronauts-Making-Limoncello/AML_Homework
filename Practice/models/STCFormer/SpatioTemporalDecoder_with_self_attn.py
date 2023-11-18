import torch
import torch.nn as nn
from SpatioTemporalCrossAttention import SpatioTemporalCrossAttention

from rich import print

def conv_init(conv):
    """
    Initializes the weights of a convolutional layer using the Kaiming Normal initialization method.

    Parameters:
        conv (torch.nn.Module): The convolutional layer to initialize.

    Returns:
        None
    """
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')

def bn_init(bn, scale):
    """
    Initializes the batch normalization layer with the given scale.

    Parameters:
        bn (torch.nn.BatchNorm2d): The batch normalization layer to initialize.
        scale (float): The scale to initialize the batch normalization layer with.

    Returns:
        None
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    """
    Initialize the weights and biases of a fully connected layer.

    Parameters:
        fc (nn.Linear): The fully connected layer to initialize.

    Returns:
        None
    """
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

class SpatioTemporalDecoder(nn.Module):
  
    def __init__(
        self, 
        in_features, hidden_features, out_features,
        spatio_temporal_self_attention_config,
        spatio_temporal_cross_attention_config
    ):
        """
        Initializes the SpatioTemporalTransformer model.

        Args:
            in_features (int): The number of input features.
            hidden_features (int): The number of hidden features.
            out_features (int): The number of output features.
            spatio_temporal_self_attention_config (List[Tuple[int, int, int, int, int]]): 
                A list of tuples specifying the configuration for each self-attention layer. 
                Each tuple contains the input features, output features, number of frames in the input, 
                number of frames in the output, and number of attention heads for a self-attention layer.
            spatio_temporal_cross_attention_config (List[Tuple[int, int, int, int, int]]): 
                A list of tuples specifying the configuration for each cross-attention layer. 
                Each tuple contains the input features, output features, number of frames in the input, 
                number of frames in the output, and number of attention heads for a cross-attention layer.

        Returns:
            None
        """
        super().__init__()

        assert len(spatio_temporal_self_attention_config) == len(spatio_temporal_cross_attention_config), "spatio-temporal self and cross attention configs must have the same number of configurations"

        self.x_fc_in              = nn.Linear(in_features, hidden_features)
        self.encoder_output_fc_in = nn.Linear(in_features, hidden_features)

        self.x_fc_out = nn.Linear(spatio_temporal_cross_attention_config[-1][1], out_features)

        self.blocks = nn.ModuleList()
        layer_id = 0
        
        for (
            self_attn_in_features, self_attn_out_feat, self_attn_num_frames, self_attn_num_frames_out, self_attn_num_heads
        ), (
            cross_attn_in_features, cross_attn_out_feat, cross_attn_num_frames, cross_attn_num_frames_out, cross_attn_num_heads
        ) in zip(
            spatio_temporal_self_attention_config, spatio_temporal_cross_attention_config
        ):
            
            self_attn_layer = SpatioTemporalCrossAttention(
                in_features=self_attn_in_features, out_features=self_attn_out_feat,
                # self-attention does NOT need temporal dimension explosion, because it 
                # uses decoder input (x) for q, k and v.
                # So, setting temporal explosion to explode to same temporal dimensionality
                num_frames=self_attn_num_frames_out, num_frames_out=self_attn_num_frames_out,
            )
            self.blocks.add_module(f"self_attn_{layer_id}", self_attn_layer)
            
            cross_attn_layer = SpatioTemporalCrossAttention(
                in_features=cross_attn_in_features, out_features=cross_attn_out_feat,
                num_frames=cross_attn_num_frames, num_frames_out=cross_attn_num_frames_out,
            )
            self.blocks.add_module(f"cross_attn_{layer_id}", cross_attn_layer)

            layer_id += 1
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
            
            
    def forward(self, x, encoder_output):
        """
        The `forward` function takes in two parameters: `x` and `encoder_output`. It performs a series of operations on these inputs to generate an output `x`.

        Args:
            x (torch.Tensor): The input tensor.
            encoder_output (torch.Tensor): The output of the encoder.

        Returns:
            torch.Tensor: The output tensor.
        """

        x = self.x_fc_in(x)
        encoder_output = self.encoder_output_fc_in(encoder_output)

        for layer_name, layer in self.blocks._modules.items():
            
            if layer_name.startswith("self_attn_"):               
                # if self-attention, then q, k and v come from the decoder input (x)
                x = layer(x, x)
             
            
            elif layer_name.startswith("cross_attn_"):
                # if cross-attention, q comes from decoder input (x), k and v come from encoder_output
                x = layer(x, encoder_output)

        x = self.x_fc_out(x)

        return x

def main():
    """
    A function that represents the main entry point of the program.

    This function performs the following steps:
    1. Checks if a CUDA-enabled GPU is available and sets the device accordingly.
    2. Prints the type of device being used.
    3. Initializes the variables for the number of joints, frames, and output frames.
    4. Initializes the variables for the input, hidden, and output features.
    5. Defines the configuration for the spatio-temporal self-attention and cross-attention.
    6. Initializes the spatio-temporal transformer using the defined configuration.
    7. Sets the batch size.
    8. Initializes the encoder output and decoder input tensors.
    9. Calls the forward method of the spatio-temporal transformer to process the decoder input with the encoder output.
    10. Prints the shape of the processed tensor.

    Parameters:
        None

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))
    
    num_joints = 22
    num_frames = 10
    num_frames_out = 25

    in_features = 3
    hidden_features = 128
    out_features = 3

    # spatial_features_in, spatial_features_out, temporal_features_in, temporal_features_out, num_heads
    spatio_temporal_self_attention_config = [
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
    ]
    
    spatio_temporal_cross_attention_config = [
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
    ]

    spatio_temporal_transformer = SpatioTemporalDecoder(
        in_features, hidden_features, out_features,
        spatio_temporal_self_attention_config=spatio_temporal_self_attention_config,
        spatio_temporal_cross_attention_config=spatio_temporal_cross_attention_config
    )

    batch_size = 256
    # encoder output must come already mapped to hidden features, because it is
    # supposed to come out of the encoder
    encoder_output = torch.rand((batch_size, num_frames    , num_joints, in_features))
    # decoder input must come with in_features, because it comes directly from the
    # dataset!
    x              = torch.rand((batch_size, num_frames_out, num_joints, in_features)) 

    x = spatio_temporal_transformer.forward(x, encoder_output)

    print(f"x.shape: {x.shape}")


if __name__ == "__main__":
    main()
    
    




