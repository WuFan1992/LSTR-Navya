# Copyright (c) . and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 128
        self.temperature = temperature  # 10000
        self.normalize = normalize   # True

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 2pi

    def forward(self, x, mask):
    
    """
       The following function is used to calculate:
       
       PE(pos, 2i) = sin(pos / (10000^(2i / d) ))
       
       where pos is the word position, d is the vector dimension and i is the index in each dimension
    
        example for the following function
        mask = [[false, false, true, false], [true , false, false, true]
        ~mask = [[true, true, false, true], [false, true, true, false]]
        
        x_embed=[1,2,2,3], [0,1,2,2]
        y_embed=[[1,1,0,1], [1,2,1,1]
        
    
    """
        assert mask is not None
        not_mask = ~mask  # set false to true and true to false for the mask


        y_embed = not_mask.cumsum(1, dtype=torch.float32) # true =1 and false=0 , accumulate in row; dim = 16x12x20 
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # dim = 16x12x20

     
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale


        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)

        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # dim_t = 1x16

        pos_x = x_embed[:, :, :, None] / dim_t   #dim = 16x12x20x16
        pos_y = y_embed[:, :, :, None] / dim_t   #dim = 16x12x20x16 



        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 0::2 means that from the index=0, take the element with index = 2,4,6,8...
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 1::2 means that from the index=1, take the element with index = 3,5,7,9...

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


def build_position_encoding(hidden_dim, type):
"""
   Default position embedding: sine 
"""
    N_steps = hidden_dim // 2
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    return position_embedding
