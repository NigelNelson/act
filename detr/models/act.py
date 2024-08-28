"""
Reference:
- https://github.com/tonyzhaozh/act
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_softmax
except:
    print("[Warning] torch_scatter not installed")

import torchvision.transforms as T
from einops import pack, rearrange, reduce, repeat, unpack

from .pointnet import offset2batch
from .rotation_conversions import matrix_to_quaternion, rotation_6d_to_matrix

from .util import get_sinusoid_encoding_table, reparametrize, KLDivergence



def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


class ToTensorIfNot(T.ToTensor):
    def __call__(self, pic):
        if not torch.is_tensor(pic):
            return super().__call__(pic)
        return pic


class ACT(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        encoder,
        hidden_dim,
        num_queries,
        num_cameras,
        action_dim=8,
        qpos_dim=9,
        env_state_dim=0,
        latent_dim=32,
        action_loss=None,
        klloss=None,
        kl_weight=20.0,
        goal_cond_dim=0,
        obs_feature_pos_embedding=None,
        freeze_backbone=False,
        ignore_vae=False,
        pretrained_weight=None,
        feature_mode="cls",
    ):
        super().__init__()

        self.backbone = backbone
        self.transformer = transformer
        self.encoder = encoder

        self.num_queries = num_queries
        self.num_cameras = num_cameras
        self.action_dim = action_dim
        self.qpos_dim = qpos_dim
        self.env_state_dim = env_state_dim
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim  # final size of latent z
        self.goal_cond_dim = goal_cond_dim
        self.obs_feature_pos_embedding = obs_feature_pos_embedding
        self.freeze_backbone = freeze_backbone
        self.ignore_vae = ignore_vae
        self.feature_mode = feature_mode

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if not action_loss:
            self.action_loss = nn.MSELoss(reduction="none")
        else:
            self.action_loss = action_loss
        if not klloss:
            self.klloss = KLDivergence()
        else:
            self.klloss = klloss

        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        if self.backbone is not None:
            self.input_proj = nn.Conv2d(
                self.backbone.num_channels, self.hidden_dim, kernel_size=1
            )
            self.input_proj_robot_state = nn.Linear(self.qpos_dim, self.hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(self.qpos_dim, self.hidden_dim)
            self.input_proj_env_state = nn.Linear(self.env_state_dim, self.hidden_dim)
            self.pos = nn.Embedding(2, self.hidden_dim)

        self.cls_embed = nn.Embedding(1, self.hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            self.action_dim, self.hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(
            self.qpos_dim, self.hidden_dim
        )  # project qpos to embedding

        self.latent_proj = nn.Linear(
            self.hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var

        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + self.num_queries, self.hidden_dim),
        )  # [CLS], obs_actions, a_seq

        if self.goal_cond_dim > 0:
            self.proj_goal_cond_emb = nn.Linear(self.goal_cond_dim, self.hidden_dim)

    def build_decoder(self):
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(self.hidden_dim, 1)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, self.hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2 + int(self.goal_cond_dim > 0), self.hidden_dim
        )  # learned position embedding for goal cond (optional), proprio and latent

    def forward_encoder(self, data_dict):
        qpos = data_dict["qpos"]
        actions = data_dict.get("actions", None)
        is_pad = data_dict.get("is_pad", None)

        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        data_dict["is_training"] = is_training

        # print("FORWARD ENCODER")
        # print(f"qpos: {qpos.shape}")
        # print(f"actions: {actions.shape}")
        # print(f"is_pad: {is_pad.shape}")

        if is_training and not self.ignore_vae:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            # print(f"action_embed: {action_embed.shape}")
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            # print(f"qpos_embed: {qpos_embed.shape}")
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            # print(f"cls_embed: {cls_embed.shape}")
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            # print(f"encoder_input: {encoder_input.shape}")
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            # print(f"perm encoder_input: {encoder_input.shape}")
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            # print(f"cls_joint_is_pad: {cls_joint_is_pad.shape}")
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            # print(f"is_pad: {is_pad.shape}")
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # print(f"pos_embed: {pos_embed.shape}")
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            # print(f"encoder_output: {encoder_output.shape}")
            latent_info = self.latent_proj(encoder_output)
            # print(f"latent_info: {latent_info.shape}")
            mu = latent_info[:, : self.latent_dim]
            # print(f"mu: {mu.shape}")
            logvar = latent_info[:, self.latent_dim :]
            # print(f"logvar: {logvar.shape}")
            latent_sample = reparametrize(mu, logvar)
            # print(f"latent_sample: {latent_sample.shape}")
            latent_input = self.latent_out_proj(latent_sample)
            # print(f"latent_input: {latent_input.shape}")
        else:  # test, no tgt actions
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        data_dict["mu"] = mu
        data_dict["logvar"] = logvar
        data_dict["latent_input"] = latent_input

        return data_dict

    def forward_obs_embed(self, data_dict):
        qpos = data_dict["qpos"]
        image = data_dict["image"]
        env_state = data_dict.get("env_state", None)
        actions = data_dict.get("actions", None)
        latent_input = data_dict["latent_input"]

        if self.goal_cond_dim > 0:
            if data_dict["goal_cond"].dim() > 2:
                data_dict["goal_cond"] = data_dict["goal_cond"].reshape(
                    data_dict["goal_cond"].shape[0], -1
                )
            goal_cond = self.proj_goal_cond_emb(data_dict["goal_cond"])

        if self.backbone is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id in range(self.num_cameras):
                if hasattr(self.backbone, "forward_feature_extractor"):
                    features = self.backbone.forward_feature_extractor(
                        self.image_transform(image[:, cam_id]),
                        mode=self.feature_mode,
                    )
                else:
                    features = self.backbone(
                        image[:, cam_id]
                    )  # (b, c, h, w) for resnet or (b, c) for vit
                # print(features.shape)
                if features.dim() == 2:  # vit
                    features = features.unsqueeze(-1).unsqueeze(-1)
                pos = self.obs_feature_pos_embedding(features).to(features.dtype)
                # print(self.input_proj(features))
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            latent_input = latent_input.unsqueeze(0)
            if self.goal_cond_dim > 0:
                goal_cond = goal_cond.unsqueeze(0)

            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos).unsqueeze(0)
            if self.goal_cond_dim > 0:
                proprio_input = torch.cat([proprio_input, goal_cond], axis=0)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            if self.goal_cond_dim <= 0:
                src = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            else:
                src = torch.cat([qpos, env_state, goal_cond], axis=1)
            pos = self.pos.weight
            latent_input = None
            proprio_input = None

        data_dict["src"] = src
        data_dict["pos"] = pos
        data_dict["latent_input"] = latent_input
        data_dict["proprio_input"] = proprio_input

        return data_dict

    def forward_decoder(self, data_dict):
        src = data_dict["src"]
        pos = data_dict["pos"]
        latent_input = data_dict["latent_input"]
        proprio_input = data_dict["proprio_input"]

        # print("FORWARD DECODER CALL")
        # print(f"src: {src.shape}")
        # print(f"pos: {pos.shape}")
        # print(f"latent_input: {latent_input.shape}")
        # print(f"proprio_input: {proprio_input.shape}")

        # (bs, num_queries, hidden_dim)
        hs = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight if latent_input is not None else None,
        )[0]

        a_hat = self.action_head(hs)  # (bs, num_queries, action_dim)

        is_pad_hat = self.is_pad_head(hs)  # (bs, num_queries, 1)

        data_dict["a_hat"] = a_hat
        data_dict["is_pad_hat"] = is_pad_hat

        return data_dict

    def forward_loss(self, data_dict):
        total_kld = self.klloss(data_dict["mu"], data_dict["logvar"])

        action_loss = self.action_loss(data_dict["a_hat"], data_dict["actions"])
        action_loss = (action_loss * ~data_dict["is_pad"].unsqueeze(-1)).mean()

        data_dict["action_loss"] = action_loss
        data_dict["kl_loss"] = total_kld
        data_dict["loss"] = action_loss + total_kld * self.kl_weight

        return data_dict

    def forward(self, data_dict):
        # obtain latent z from action sequence
        # print("START ENCODER")
        data_dict = self.forward_encoder(data_dict)
        # print("COMPLETE ENCODER")
        # print()
        # print("START OBS EMBED")
        # obtain proprioception and image features
        data_dict = self.forward_obs_embed(data_dict)
        # print("COMPLETE OBS EMBED")
        # print()
        # print("START DECODER")
        # decode action sequence from proprioception and image features
        data_dict = self.forward_decoder(data_dict)
        # print("COMPLETE DECODER")

        if not data_dict["is_training"]:
            return data_dict

        # compute loss
        data_dict = self.forward_loss(data_dict)

        return data_dict


class ACTPCD(ACT):
    def __init__(
        self,
        backbone,
        transformer,
        encoder,
        hidden_dim,
        num_queries,
        num_cameras,
        action_dim=8,
        qpos_dim=9,
        env_state_dim=0,
        latent_dim=32,
        action_loss=None,
        klloss=None,
        kl_weight=20.0,
        goal_cond_dim=0,
        obs_feature_pos_embedding=None,
        freeze_backbone=False,
        pcd_nsample=16,
        pcd_npoints=1024,
        sampling="fps",
        heatmap_th=0.1,
        ignore_vae=False,
        use_mask=False,
        bg_ratio=0.0,
        pre_sample=False,
        in_channels=6,
    ):
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            encoder=encoder,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_cameras=0,
            action_dim=action_dim,
            qpos_dim=qpos_dim,
            env_state_dim=env_state_dim,
            latent_dim=latent_dim,
            action_loss=action_loss,
            klloss=klloss,
            kl_weight=kl_weight,
            goal_cond_dim=goal_cond_dim,
            obs_feature_pos_embedding=None,
            freeze_backbone=freeze_backbone,
            ignore_vae=ignore_vae,
        )

        self.input_proj = None

        # build fps sampler
        self.pcd_nsample = pcd_nsample
        self.pcd_npoints = pcd_npoints
        self.pre_sample = pre_sample

        self.pre_sample = False
        self.use_mask = False

        if not pre_sample:
            self.linear = nn.Linear(
                3 + self.backbone.num_channels, hidden_dim, bias=False
            )
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.linear = nn.Linear(
                3 + self.backbone.in_channels, self.backbone.in_channels, bias=False
            )
            self.bn = nn.BatchNorm1d(self.backbone.in_channels)
        self.pool = nn.MaxPool1d(pcd_nsample)

        self.relu = nn.ReLU(inplace=True)
        self.sampling = sampling
        self.use_mask = use_mask
        self.bg_ratio = bg_ratio

    def coord_embedding_sine(self, coord, temperature=10000, normalize=False, scale=None):
        # coord shape: [B, 3, N] where B is batch size, N is number of points
        B, _, N = coord.shape
        num_pos_feats = self.hidden_dim // 3
        num_pad_feats = self.hidden_dim - num_pos_feats * 3

        if normalize:
            coord = coord / (coord.max(dim=2, keepdim=True)[0] + 1e-6) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=coord.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = coord[:, 0, :, None] / dim_t
        pos_y = coord[:, 1, :, None] / dim_t
        pos_z = coord[:, 2, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=3).flatten(2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_x, pos_y, pos_z), dim=2)
        pos = torch.cat((pos, torch.zeros(B, N, num_pad_feats, device=pos.device)), dim=2)
        
        return pos  # Shape: [B, N, hidden_dim]

    def forward_pcd_embed(self, pcd_dict):
        # print("FORWARD PCD EMBED")
        features = pcd_dict["feat"]  # Assuming features are already in the required shape [BS, 1024, C]
        coord = pcd_dict["grid_coord"]
        offset = pcd_dict["offset"]
        # print(f"features: {features.shape}")
        # print(f"coord: {coord.shape}")
        # print(f"offset: {offset.shape}")

        # Pass through backbone to get features
        features = self.backbone(pcd_dict)
        # print(f"bb features: {features.shape}")



        # Compute position embeddings for the point cloud coordinates
        pcd_pos = self.coord_embedding_sine(coord)
        # print(f"pcd_pos: {pcd_pos.shape}")
        
        features = rearrange(
            features,
            "(b n) c -> b c 1 n",
            n=self.pcd_npoints,
        )
        # print(f"rearranged features: {features.shape}")
        pcd_pos = rearrange(pcd_pos, "b n c -> b c 1 n")
        # print(f"repeated pcd_pos: {pcd_pos.shape}")
        return features, pcd_pos

    def forward_obs_embed(self, data_dict):
        qpos = data_dict["qpos"]
        pcd_dict = data_dict["pcds"]
        env_state = data_dict.get("env_state", None)
        actions = data_dict.get("actions", None)
        latent_input = data_dict["latent_input"]

        # Process the point cloud data to get tokens and position embeddings
        pcd_tokens, pcd_pos = self.forward_pcd_embed(pcd_dict)

        latent_input = latent_input.unsqueeze(0)
        if self.goal_cond_dim > 0:
            goal_cond = goal_cond.unsqueeze(0)

        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos).unsqueeze(0)
        if self.goal_cond_dim > 0:
            proprio_input = torch.cat([proprio_input, goal_cond], axis=0)

        src = pcd_tokens
        pos = pcd_pos

        data_dict["src"] = src
        data_dict["pos"] = pos
        data_dict["latent_input"] = latent_input
        data_dict["proprio_input"] = proprio_input

        return data_dict