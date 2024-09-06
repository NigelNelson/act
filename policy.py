import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from detr.models import PointNet, ACTPCD
from detr.models.transformer import Transformer, TransformerEncoder, TransformerEncoderLayer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class ACT3DPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        self.model = self._build_model(args_override)
        self.optimizer = self._build_optimizer(args_override)
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def _build_model(self, args_override):
        # Hard-coded configuration based on ACTPCD_train_config
        qpos_dim = 16 if args_override['dual_arm'] else 8
        action_dim = 14 if args_override['dual_arm'] else 7
        config = {
            "hidden_dim": 512,
            "num_queries": args_override['num_queries'],
            "num_cameras": 0,
            "action_dim": action_dim,
            "qpos_dim": qpos_dim,
            "env_state_dim": 0,
            "latent_dim": 32,
            "kl_weight": args_override['kl_weight'],
            "goal_cond_dim": 0,
            "pcd_nsample": 16,
            "pcd_npoints": 1024,  # Changed to match your input size
            "pre_sample": False,  # No need to sample as we have fixed size
            "in_channels": 3,  # XYZ coordinates
        }

        # Initialize backbone, transformer, and encoder (simplified for brevity)
        backbone = self._build_backbone()
        transformer = self._build_transformer(config)
        encoder = self._build_encoder(config)

        return ACTPCD(
            backbone=backbone,
            transformer=transformer,
            encoder=encoder,
            **config
        )

    def _build_backbone(self):
        # Implement a simple backbone for point cloud processing
        return PointNet(
            in_channels=3,
            num_classes=0
        )

    def _build_transformer(self, config):
        return Transformer(
            d_model=config['hidden_dim'],
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=7,
            dim_feedforward=32,
            dropout=0.1,
            return_intermediate_dec=True
        )

    def _build_encoder(self, config):
        return TransformerEncoder(
            d_model=config['hidden_dim'],
            nhead=8,
            dim_feedforward=32,
            dropout=0.1,
            normalize_before=False,
            activation='relu'
            )

    def _build_optimizer(self, args_override):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=args_override['lr'],
            weight_decay=0.05
        )


    def __call__(self, qpos, pcds, actions=None, is_pad=None):
        # Prepare input data
        data_dict = {
            "qpos": qpos,
            "pcds": {
                "grid_coord": pcds.transpose(1, 2),  # Change to (B, 3, 1024)
                "feat": pcds.transpose(1, 2),   # Use coordinates as features
                "offset": torch.cumsum(torch.tensor([pcds.shape[0]] * pcds.shape[0]), dim=0),
            },
            "actions": actions,
            "is_pad": is_pad
        }

        # Forward pass
        output = self.model(data_dict)

        if actions is not None:  # Training mode
            return {
                "loss": output["loss"],
                "l1": output["action_loss"],
                "kl": output["kl_loss"]
            }
        else:  # Inference mode
            return output["a_hat"]

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld