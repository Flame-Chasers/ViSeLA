from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel
import torch.distributed as dist
from transformers.configuration_utils import PretrainedConfig

def compute_itc(sim_matrix, label_distribution):
    loss = -torch.sum(F.log_softmax(sim_matrix, dim=1) * label_distribution, dim=1).mean()
    return loss

def compute_sdm(sim_matrix, label_distribution, epsilon=1e-6):
    pred = F.softmax(sim_matrix, dim=1)
    loss = pred * (F.log_softmax(sim_matrix, dim=1) - torch.log(label_distribution + epsilon))
    loss = torch.mean(torch.sum(loss, dim=1))
    return loss

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )

class RefinerConfig(PretrainedConfig):
    model_type = 'refiner'

    def __init__(
            self,
            enable: bool = True,
            enable_sdm: bool = False,
            use_contrastive_loss: bool = True,  # whether to compute the refiner contrastive loss (ablation switch)
            num_query: int = 32,
            num_hidden_layers: int = 2,
            sim_type: str = 'avg',
            lambda_feat_div: float = 1e-2,
            lambda_attn_div: float = 1e-3,
            diversity_mode: str = 'hard',       # 'hard' or 'soft'
            diversity_threshold: float = 0.5,   # threshold for soft mode
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable = enable
        self.enable_sdm = enable_sdm
        self.use_contrastive_loss = use_contrastive_loss
        self.num_query = num_query
        self.num_hidden_layers = num_hidden_layers
        self.sim_type = sim_type
        self.lambda_feat_div = lambda_feat_div
        self.lambda_attn_div = lambda_attn_div
        self.diversity_mode = diversity_mode
        self.diversity_threshold = diversity_threshold

class Refiner(PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"query_token"]
    _keys_to_ignore_on_load_missing = [r"attr_query_token"]

    def __init__(self, config):
        super(Refiner, self).__init__(config)

        self.attr_query_token = nn.Parameter(torch.empty(1, config.num_query, config.hidden_size))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.hidden_size // 64,
            batch_first=True
        )
        self.sim_type = config.sim_type

        self.enable_sdm = config.enable_sdm

        self.post_init()

    def post_init(self):
        scale = 768 ** -0.5
        attn_std = scale
        proj_std = scale * ((2 * 2) ** -0.5)

        # init query_token
        self.attr_query_token.data.normal_(std=0.02)

        # init cross_attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

    def forward_train(self, x, attention_mask=None, return_attn=False):
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        x, attn = self.cross_attn(
            query=self.attr_query_token.repeat(x.shape[0], 1, 1),
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        if return_attn:
            return x, attn
        return x
    
    def compute_sim(self, attrs_embeds_1: torch.Tensor, attrs_embeds_2: torch.Tensor):
        """
        Compute cosine similarity for each attribute.

        Args:
            attrs_embeds_1: (N, K, C)
            attrs_embeds_2: (M, K, C)
            K is the number of attributes.

        Returns:
            sim: (N, M)
        """
        attrs_embeds_1 = attrs_embeds_1.unsqueeze(1)  # N 1 K C
        attrs_embeds_2 = attrs_embeds_2.unsqueeze(0)  # 1 M K C

        sim = F.cosine_similarity(attrs_embeds_1, attrs_embeds_2, dim=-1)

        if self.sim_type == 'avg':
            sim = sim.mean(dim=-1)
        elif self.sim_type == 'max':
            sim = sim.max(dim=-1)[0]
        elif self.sim_type == 'min':
            sim = sim.min(dim=-1)[0]
        elif self.sim_type == 'lsp_max':
            sim = torch.logsumexp(sim, dim=-1)
        elif self.sim_type == 'lsp_min':
            sim = -torch.logsumexp(-sim, dim=-1)
        else:
            raise ValueError(f"Unknown sim_type: {self.sim_type}")

        return sim
    
    def get_attn(self, embeds):
        _, attn = self.forward_train(embeds, attention_mask=None, return_attn=True)
        return attn

    def forward(self, image_embeds, text_embeds, attention_mask, idx, temp, return_aux=False):
        image_attrs, image_attn = self.forward_train(image_embeds, attention_mask=None, return_attn=True)  # N K C
        text_attrs, text_attn = self.forward_train(text_embeds, attention_mask=attention_mask, return_attn=True)  # N K C
        image_attrs_all = AllGather.apply(image_attrs.contiguous(), torch.distributed.get_rank(), torch.distributed.get_world_size())  # NG K C
        text_attrs_all = AllGather.apply(text_attrs.contiguous(), torch.distributed.get_rank(), torch.distributed.get_world_size())  # NG k C

        sim_i2t = self.compute_sim(image_attrs_all, text_attrs_all)
        sim_t2i = sim_i2t.t()

        sim_i2t = sim_i2t / temp
        sim_t2i = sim_t2i / temp

        idx = idx.view(-1, 1)
        assert idx.size(0) == image_attrs.size(0)
        idx_all = AllGather.apply(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
        pos_idx = torch.eq(idx_all, idx_all.t()).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        loss_t2i = compute_itc(sim_t2i, sim_targets)
        loss_i2t = compute_itc(sim_i2t, sim_targets)
        loss = (loss_t2i + loss_i2t) / 2

        if self.enable_sdm:
            loss_t2i = compute_sdm(sim_t2i, sim_targets)
            loss_i2t = compute_sdm(sim_i2t, sim_targets)
            loss_sdm = (loss_t2i + loss_i2t) / 2
            loss = loss + loss_sdm

        if return_aux:
            aux = {
                "image_attrs": image_attrs,
                "text_attrs": text_attrs,
                "image_attn": image_attn,
                "text_attn": text_attn,
            }
            return loss, aux

        return loss


def query_feature_diversity_loss(q_feat: torch.Tensor, mode='hard', threshold=0.5):
    """
    Encourage different queries within a sample to be orthogonal.

    Args:
        q_feat: [batch_size, num_queries, hidden_dim] - query features
        mode: 'hard' for full orthogonality, 'soft' for threshold-based penalty
        threshold: only penalize similarities above this value (for soft mode)

    Returns:
        diversity_loss: scalar tensor
    """
    q_norm = F.normalize(q_feat, dim=-1)
    gram = torch.matmul(q_norm, q_norm.transpose(1, 2))
    off_diag = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))

    if mode == 'soft':
        # Soft constraint: only penalize similarity above the threshold,
        # allowing some overlap without over-constraining the queries
        excess = F.relu(off_diag - threshold)
        return (excess ** 2).mean()
    else:
        # Hard constraint: penalize every off-diagonal similarity
        return (off_diag ** 2).mean()


def attention_diversity_loss(attn: torch.Tensor, attention_mask=None, mode='hard', threshold=0.5):
    """
    Encourage queries to attend to different regions.

    Args:
        attn: [batch_size, num_queries, seq_len] - attention weights from MultiheadAttention
        attention_mask: [batch_size, seq_len] - 1 for valid tokens, 0 for padding
        mode: 'hard' for full diversity, 'soft' for threshold-based penalty
        threshold: only penalize overlaps above this value (for soft mode)

    Returns:
        diversity_loss: scalar tensor
    """
    # Mask out padding positions when an attention_mask is provided
    if attention_mask is not None:
        # attention_mask: [B, L] -> [B, 1, L]
        mask = attention_mask.unsqueeze(1).float()  # [B, 1, L]
        # Zero the attention on padding tokens
        attn = attn * mask  # [B, num_queries, L]

    # Renormalize over the valid positions (epsilon avoids division by zero)
    attn_sum = attn.sum(dim=-1, keepdim=True)
    attn = attn / (attn_sum + 1e-8)

    # Gram matrix: G[i,j] = similarity between query i and query j
    gram = torch.matmul(attn, attn.transpose(1, 2))  # [B, num_queries, num_queries]

    # Off-diagonal entries (similarity between different queries)
    off_diag = gram - torch.diag_embed(torch.diagonal(gram, dim1=1, dim2=2))

    if mode == 'soft':
        # Soft constraint: only penalize overlap above the threshold,
        # allowing queries to share salient regions to some extent
        excess = F.relu(off_diag - threshold)
        return (excess ** 2).mean()
    else:
        # Hard constraint: penalize every attention overlap
        return (off_diag ** 2).mean()


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)