"""
Reference:
- https://github.com/OpenGVLab/PonderV2/blob/main/ponder/models/sparse_unet/spconv_unet_v1m3_pdnorm.py
- https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/pointnet.py
"""

from functools import partial

import spconv.pytorch as spconv
import torch
import torch.nn as nn

def offset2batch(offset):
    return (
        torch.cat(
            [
                (
                    torch.tensor([i] * (o - offset[i - 1]))
                    if i > 0
                    else torch.tensor([i] * o)
                )
                for i, o in enumerate(offset)
            ],
            dim=0,
        )
        .long()
        .to("cuda")
    )



class PointNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=0,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.embedding_table = None

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, 64, kernel_size=1, bias=False),
            norm_fn(64),
            nn.ReLU(),
        )
        self.conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(64, 64, kernel_size=1, bias=False),
            norm_fn(64),
            nn.ReLU(),
        )
        self.conv3 = spconv.SparseSequential(
            spconv.SubMConv3d(64, 64, kernel_size=1, bias=False),
            norm_fn(64),
            nn.ReLU(),
        )
        self.conv4 = spconv.SparseSequential(
            spconv.SubMConv3d(64, 128, kernel_size=1, bias=False),
            norm_fn(128),
            nn.ReLU(),
        )
        self.conv5 = spconv.SparseSequential(
            spconv.SubMConv3d(128, 512, kernel_size=1, bias=False),
            norm_fn(512),
            nn.ReLU(),
        )

        self.final = (
            spconv.SubMConv3d(512, num_classes, kernel_size=1, padding=1, bias=True)
            if num_classes > 0
            else spconv.Identity()
        )
        self.num_channels = num_classes if num_classes > 0 else 512

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]

        batch_size, num_coords, num_points = grid_coord.shape
        _, num_features, _ = feat.shape

        # Create batch indices: repeat each batch index num_points times
        batch_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, num_points).reshape(-1).cuda()  # shape: [batch * num_points]

        # Flatten grid coordinates correctly without permuting
        grid_coord_flat = grid_coord.permute(0, 2, 1).reshape(-1, 3)  # Correctly flatten to [batch * num_points, 3]
        
        # Concatenate batch indices with grid coordinates
        indices = torch.cat([batch_indices.unsqueeze(1), grid_coord_flat], dim=1)  # shape: [batch * num_points, 4]

        # Flatten features correctly without permuting
        feat_flat = feat.permute(0, 2, 1).reshape(-1, num_features)  # Correctly flatten to [batch * num_points, num_features]

        # Calculate spatial shape as integer values
        sparse_shape = torch.add(torch.max(grid_coord_flat, dim=0).values, 96).int().tolist()

        # print(f"indices: {indices.shape}")  # should be [batch * num_points, 4]
        # print(f"feat: {feat_flat.shape}")  # should be [batch * num_points, num_features]
        # print(f"sparse_shape: {sparse_shape}")

        # Create the SparseConvTensor
        x = spconv.SparseConvTensor(
            features=feat_flat,
            indices=indices.int(),
            spatial_shape=sparse_shape,
            batch_size=batch_size,
        )
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.final(x)

        # print(f"features: {x.features.shape}") # should be [batch * num_points, num_classes]
        # print("END OF POINTNET")

        return x.features