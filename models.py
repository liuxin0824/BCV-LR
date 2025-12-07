import config
import data_loader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from funcy import partition
from tensordict import TensorDict
from torch.distributions import Categorical
import utils

import sys

import math

ObsShapeType = tuple[int, int, int]  # (channels, height, width)


def merge_TC_dims(x: torch.Tensor):
    """x.shape == (B, T, C, H, W) -> (B, T*C, H, W)"""
    return x.reshape(x.shape[0], -1, *x.shape[3:])


class ResidualLayer(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_out_dim, hidden_dim, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_out_dim, kernel_size, stride, padding),
        )

    def forward(self, x):
        return x + self.res_block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.conv = nn.Conv2d(in_depth, out_depth, 3, 1, padding=1)
        self.norm = nn.BatchNorm2d(out_depth)
        self.res = ResidualLayer(out_depth, out_depth // 2, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.pool(self.res(self.norm(self.conv(x)))))


class UpsampleBlock(nn.Module):
    def __init__(self, in_depth, out_depth):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_depth, out_depth, 2, 2)
        # maybe put bn before conv since that's where unet connections are catted
        self.norm = nn.BatchNorm2d(out_depth)
        self.res = ResidualLayer(out_depth, out_depth // 2, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.res(self.norm(self.conv(x))))




class WorldModel(nn.Module):
    """UNet-based world model"""

    def __init__(self, action_dim, in_depth, out_depth, base_size=16):
        super().__init__()
        b = base_size

        self.conv = ConvSequence_forencoder((9,84,84)) #impala_scale = 4   equal to that in IDM
        obs_shape = self.conv.get_output_shape() 

        in_depth = obs_shape[0]*(1+config.ADD_TIME_HORIZON)
        out_depth = obs_shape[0]

        # downscaling   
        down_sizes = (in_depth + action_dim, b, 2 * b, 4 * b, 8 * b, 16 * b)
        self.down = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, down_sizes)):
            if i < len(down_sizes) - 2:
                self.down.append(DownsampleBlock(in_size, out_size))
            else:
                self.down.append(nn.Conv2d(in_size, out_size, 2, 1))

        # upscaling
        up_sizes = ( 16 * b, 8 * b, 4 * b, 2 * b, b, b)
        self.up = nn.ModuleList()
        for i, (in_size, out_size) in enumerate(partition(2, 1, up_sizes)):
            incoming = action_dim if i == 0 else down_sizes[-i - 1]
            self.up.append(UpsampleBlock(in_size + incoming, out_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(up_sizes[-1] + in_depth, b, kernel_size=3, stride=1, padding=1),
            ResidualLayer(b, b // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, out_depth, 1, 1),    #out_depth 3
        )



        

    def forward(self, state_seq, action):
        """
        state_seq.shape = (B, L, C, H, W)
        action.shape = (B, L)
        """

        state = merge_TC_dims(state_seq)   # 128, 6 , 64, 64

        _, _, h, w = state.shape
        action = action[:, :, None, None]

        # we inject the latent action at two points: at the very first layer, and in the middle of the UNet.
        # this seems to work well in practice, but can probably be simplified

        
        

        # repeat action (batch, dim) across w x h dimensions
        x = torch.cat([state, action.repeat(1, 1, h, w)], dim=1)

        '''
        print(state.shape)
        print(action.repeat(1, 1, h, w).shape)
        print(x.shape)
        sys.exit()
        torch.Size([128, 6, 64, 64])  前两帧
        torch.Size([128, 128, 64, 64])   128维度动作直接展开成64x64
        torch.Size([128, 134, 64, 64])
        '''

        xs = []
        for layer in self.down:
            x = layer(x)
            xs.append(x)

        xs[-1] = action

        for i, layer in enumerate(self.up):
            x = layer(torch.cat([x, xs[-i - 1]], dim=1))

        out = self.final_conv(torch.cat([x, state], dim=1))
        return F.tanh(out) / 2

    def label(self, batch: TensorDict) -> torch.Tensor:
        wm_in_seq = batch["obs"][:, :-1]
        #print(wm_in_seq.shape) #torch.Size([128, 2, 3, 64, 64])
        wm_targ = batch["obs"][:, -1]
        #print(wm_targ.shape) #torch.Size([128, 3, 64, 64])

        with torch.no_grad():
            batch_size = wm_in_seq.shape[0]
            x_batchout = merge_BT_dims(wm_in_seq)
            conv_feature_mergebt = self.conv(x_batchout)
            conv_feature = divide_BT_dims(conv_feature_mergebt,batch_size)
            #wm_in_seq = merge_TC_dims(conv_feature)  # 128,128,32,32
            wm_in_seq = conv_feature #128,2,64,32,32 forward contains the merge TC

            wm_targ = self.conv(wm_targ)



        #sys.exit()
        la = batch["la_q"]#la = batch["la_q"]  # TODO: also allow using la(noq)
        batch["wm_pred"] = self(wm_in_seq, la)
        return F.mse_loss(batch["wm_pred"], wm_targ)


def layer_init(layer, std=None, bias_const=0.0):
    if std is not None:
        std = np.sqrt(2)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# based on https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ImpalaResidualBlock(nn.Module):   #残差连接，不改变channel，h，w
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.res_block0 = ImpalaResidualBlock(self._out_channels)
        self.res_block1 = ImpalaResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self) -> tuple[int, int, int]:
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class ConvSequence_forencoder(nn.Module):
    def __init__(self, input_shape, out_channels = 32):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Sequential(nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
            stride=2
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        ), nn.ReLU(),
        nn.Conv2d(
            in_channels=self._out_channels,
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        ), nn.ReLU()
        
        
        )

        self.apply(utils.weight_init)
        #self.res_block0 = ImpalaResidualBlock(self._out_channels)
        #self.res_block1 = ImpalaResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        #x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #x = self.res_block0(x)
        #x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    #def get_output_shape(self) -> tuple[int, int, int]:   #使用max_pool2d
        #_c, h, w = self._input_shape
        #return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

    def get_output_shape(self) -> tuple[int, int, int]:   #不使用max_pool2d
        _c, h, w = self._input_shape
        return (32, 32, 32 )


def get_impala(
    shape: ObsShapeType,
    impala_cnn_scale: int,
    out_channels: tuple[int, ...],
    out_features: int,
) -> tuple[nn.Sequential, nn.Linear]:
    conv_stack = []
    for out_ch in out_channels:
        conv_seq = ConvSequence(shape, impala_cnn_scale * out_ch)
        shape = conv_seq.get_output_shape()
        conv_stack.append(conv_seq)
    conv_stack = nn.Sequential(*conv_stack, nn.Flatten(), nn.ReLU())
    fc = nn.Linear(in_features=np.prod(shape), out_features=out_features)
    return conv_stack, fc

    


class Policy(nn.Module):
    """IMPALA CNN-based policy"""

    def __init__(
        self,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()

        self.conv = ConvSequence_forencoder((9,84,84))
        obs_shape = self.conv.get_output_shape() 

        #self.conv_stack, self.fc = get_impala(
            #obs_shape, impala_scale, impala_channels, impala_features
        #)

        repr_dim = np.prod(obs_shape)
        feature_dim = 50
        hidden_dim = 1024

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))
        #self.policy_head = nn.Linear(impala_features, action_dim)
        #self.value_head = layer_init(nn.Linear(impala_features, 1), std=1)

    def forward(self, x):
        with torch.no_grad():   #share encoder with idm
            x = self.conv(x)    #share encoder with idm
            x = x.view(x.shape[0], -1)
        return self.policy_head(self.trunk(x))






class VQEmbeddingEMA(nn.Module):
    def __init__(
        self,
        cfg: config.VQConfig,
        epsilon=1e-5,
    ):
        super(VQEmbeddingEMA, self).__init__()
        self.epsilon = epsilon
        self.cfg = cfg

        embedding = torch.zeros(cfg.num_codebooks, cfg.num_embs, cfg.emb_dim)
        embedding.uniform_(-1 / cfg.num_embs * 5, 1 / cfg.num_embs * 5)

        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(cfg.num_codebooks, cfg.num_embs))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward_2d(self, x):
        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D

        x = x.view(B, N, D, H, W).permute(1, 0, 3, 4, 2)
        x_flat = x.detach().reshape(N, -1, D)

        distances = torch.baddbmm(
            torch.sum(self.embedding**2, dim=2).unsqueeze(1)
            + torch.sum(x_flat**2, dim=2, keepdim=True),
            x_flat,
            self.embedding.transpose(1, 2),
            alpha=-2.0,
            beta=1.0,
        )
        indices = torch.argmin(distances, dim=-1)

        encodings = F.one_hot(indices, M).float()
        quantized = torch.gather(
            self.embedding, 1, indices.unsqueeze(-1).expand(-1, -1, D)
        )
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.cfg.decay * self.ema_count + (
                1 - self.cfg.decay
            ) * torch.sum(encodings, dim=1)

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (
                (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            )
            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = (
                self.cfg.decay * self.ema_weight + (1 - self.cfg.decay) * dw
            )

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.cfg.commitment_cost * e_latent_loss

        quantized = quantized.detach() + (x - x.detach())

        avg_probs = torch.mean(encodings, dim=1)
        perplexity = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)
        )

        return (
            quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W),
            loss,
            perplexity.sum(),
            indices.view(N, B, H, W).permute(1, 0, 2, 3),
        )

    def forward(self, x):
        bs = len(x)
        x = x.view(
            bs,
            self.cfg.num_codebooks * self.cfg.emb_dim,
            self.cfg.num_discrete_latents,
            1,
        )

        z_q, loss, perplexity, indices = self.forward_2d(x)

        return (
            z_q.view(
                bs,
                self.cfg.num_codebooks
                * self.cfg.num_discrete_latents
                * self.cfg.emb_dim,
            ),
            loss,
            perplexity,
            indices,
        )

    def inds_to_z_q(self, indices):
        """look up quantization inds in embedding"""
        assert not self.training
        N, M, D = self.embedding.size()
        B, N_, H, W = indices.shape
        assert N == N_

        # N ... num_codebooks
        # M ... num_embs
        # D ... emb_dim
        # H ... num_discrete_latents (kinda)

        inds_flat = indices.permute(1, 0, 2, 3).reshape(N, B * H * W)
        quantized = torch.gather(
            self.embedding, 1, inds_flat.unsqueeze(-1).expand(-1, -1, D)
        )
        return (
            quantized.view(N, B, H, W, D).permute(1, 0, 4, 2, 3).reshape(B, N * D, H, W)
        )  # shape is (B, num_codebooks * emb_dim, num_discrete_latents, 1)


def merge_BT_dims(x: torch.Tensor):
    """x.shape == (B, T, C, H, W) -> (B*T,C, H, W)"""
    return x.reshape(-1,*x.shape[2:])

def divide_BT_dims(x: torch.Tensor, batch_size):
    """x.shape == (B*T, E) -> (B,T,E)"""
    return x.reshape(batch_size, -1, *x.shape[1:])

def merge_TE_dims(x: torch.Tensor):
    """x.shape == (B, T, E) -> (B, E*T)"""
    return x.reshape(x.shape[0],-1)



class IDM(nn.Module):   # add an encoder before previous architecture
    """Quantized inverse dynamics model"""

    def __init__(
        self,
        vq_config: config.VQConfig,
        obs_shape: ObsShapeType,
        action_dim: int,
        impala_scale: int,
        impala_channels: tuple[int, ...] = (16, 32, 32),
        impala_features=256,
    ):
        super().__init__()

        self.conv = ConvSequence_forencoder((9,64,64))
        obs_shape = self.conv.get_output_shape() 
        


        self.conv_target = ConvSequence_forencoder((9,64,64))
        self.conv_target.load_state_dict(self.conv.state_dict())

        feature_dim = 128  

        
        self.projector = nn.Sequential(nn.Flatten(),nn.Linear(math.prod(obs_shape), feature_dim))      
        self.projector_target = nn.Sequential(nn.Flatten(),nn.Linear(math.prod(obs_shape), feature_dim))
        


        self.projector.apply(utils.weight_init)
        self.projector_target.load_state_dict(self.projector.state_dict())

        self.proto = Proto_cluster(feature_dim, 512)

        self.proto_optimizer = torch.optim.Adam(utils.chain(
            self.conv.parameters(), self.projector.parameters(), self.proto.parameters()),
                                                lr=1e-4)
        print('ok')

        self.aug = utils.RandomShiftsAug_nobinliear(pad=4)

        
        obs_shape = (obs_shape[0]*2,)+obs_shape[1:]

        self.conv_stack, self.fc = get_impala(
            obs_shape, impala_scale, impala_channels, impala_features
        )
        self.policy_head =  nn.Linear(impala_features, action_dim)

        # initialize quantizer
        self.vq = VQEmbeddingEMA(vq_config)

        self.convout_feature_dim = impala_features

        self.update_count = 0

    def update_repre(self, x):
        current = self.aug(x[:,0,:,:,:])
        #next = self.aug(x[:,0,:,:,:])   # curl
        next = self.aug(x[:,1,:,:,:])   # atc, proto
        #assert current.shape == torch.Size([128, 9, 64, 64])
        
        z = self.projector(self.conv(current))
        with torch.no_grad():
            next_z = self.projector_target(self.conv_target(next))

        loss = self.proto(z, next_z)
        self.proto_optimizer.zero_grad()
        loss.backward()
        self.proto_optimizer.step()

        self.update_count += 1
        #if self.update_count %2 == 0:
        utils.soft_update_params(self.conv, self.conv_target,
                                        0.05)
        utils.soft_update_params(self.projector, self.projector_target,
                                        0.05)

        return loss


    def forward(self, x):
        """
        x.shape = (B, T, C, H, W)
        the IDM predicts the action between the last and second to last frames (T dim).
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            x_batchout = merge_BT_dims(x)
            conv_feature_mergebt = self.conv(x_batchout)
            conv_feature = divide_BT_dims(conv_feature_mergebt,batch_size)
            x = merge_TC_dims(conv_feature)

        la = self.policy_head(F.relu(self.fc(self.conv_stack(x))))
        la_q, vq_loss, vq_perp, la_qinds = self.vq(la)

        action_dict = TensorDict(
            dict(
                la=la,
                la_q=la_q,
                la_qinds=la_qinds,
            ),
            batch_size=len(la),
        )

        return action_dict, vq_loss, vq_perp

    def label(self, batch: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        #self.update_repre(batch["obs"])
        action_td, vq_loss, vq_perp = self(batch["obs"])
        batch.update(action_td)
        return vq_loss, vq_perp

    @torch.no_grad()
    def label_chunked(
        self,
        data: TensorDict,
        chunksize: int = 128,
    ) -> TensorDict:

        def _label(batch: TensorDict):
            return self(data_loader.normalize_obs(batch["obs"].to(config.DEVICE)))[0].to(
                batch.device
            )

        action_dicts = torch.cat(list(map(_label, data.split(chunksize))))
        assert len(action_dicts) == len(data)
        data.update(action_dicts)
        return data


    


class Proto_cluster(nn.Module):
    def __init__(self, proj_dim, pred_dim ):
        super().__init__()

        self.predictor = nn.Sequential(nn.Linear(proj_dim,
                                                 pred_dim), nn.ReLU(),
                                       nn.Linear(pred_dim, proj_dim))

        self.num_iters = 3
        self.T = 0.1
        self.num_protos = 512

        self.protos = nn.Linear(proj_dim, self.num_protos, bias=False)
        

        
        self.apply(utils.weight_init)

        



    def forward(self, s, t):
        # normalize prototypes
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

        s = self.predictor(s)

        s = F.normalize(s, dim=1, p=2)
        t = F.normalize(t, dim=1, p=2)

        scores_s = self.protos(s)
        log_p_s = F.log_softmax(scores_s / self.T, dim=1)

        with torch.no_grad():
            scores_t = self.protos(t)
            q_t = self.sinkhorn(scores_t)

        loss = -(q_t * log_p_s).sum(dim=1).mean()
        return loss


    def sinkhorn(self, scores):
        def remove_infs(x):
            m = x[torch.isfinite(x)].max().item()
            x[torch.isinf(x)] = m
            return x

        Q = scores / self.T
        Q -= Q.max()

        Q = torch.exp(Q).T
        Q = remove_infs(Q)
        Q /= Q.sum()

        r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
        c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
        for it in range(self.num_iters):
            u = Q.sum(dim=1)
            u = remove_infs(r / u)
            Q *= u.unsqueeze(dim=1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
        Q = Q / Q.sum(dim=0, keepdim=True)
        return Q.T
        





