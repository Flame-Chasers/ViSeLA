# -*- coding: utf-8 -*-
# PyTorch Native DDP Accelerator (no Apex dependency)

import os
import random
from typing import Tuple
import numpy as np

import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

Net = torch.nn.Module

from .accelerator import Accelerator


class TorchDDPAccelerator(Accelerator):
    """
    PyTorch Native DDP Accelerator with AMP support (no Apex required)
    """

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.accelerator_rng_seed = self.cfg.RNG_SEED
        self.accelerator_syncbn = self.cfg.SYNCBN
        self.accelerator_fp16_opt_level = self.cfg.FP16_OPT_LEVEL

        self.use_amp = self.accelerator_fp16_opt_level != 'O0'
        self.scaler = GradScaler(enabled=self.use_amp)

    def set_up(self, model: Net, optimizer: Optimizer, lr_scheduler: LambdaLR,
               local_rank: int, world_size: int, rank: int) -> Tuple[DDP, Optimizer, LambdaLR]:
        seed = int(self.accelerator_rng_seed) + int(rank)
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        master_address = os.environ.get('MASTER_ADDR', "127.0.0.1")
        master_port = int(os.environ.get('MASTER_PORT', 34171))

        torch.cuda.set_device(local_rank)
        model = model.cuda()

        if not torch.distributed.is_initialized():
            distributed.init_process_group(
                backend='nccl',
                init_method='tcp://{}:{}'.format(master_address, master_port),
                world_size=world_size,
                rank=rank)
            print(f'TorchDDPAccelerator distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')

        self.broadcast(model)

        if self.accelerator_syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        self.ddp_model = ddp_model

        return ddp_model, optimizer, lr_scheduler

    def broadcast(self, model: Net, src=0) -> None:
        for v in model.state_dict().values():
            distributed.broadcast(v, src)

    def backward_step(self, loss: torch.Tensor, optimizer: Optimizer):
        self.scaler.scale(loss).backward()

    def autocast_context(self):
        return autocast(enabled=self.use_amp)

    def optimizer_step(self, optimizer: Optimizer, model: Net, grad_norm: float) -> float:
        self.scaler.unscale_(optimizer)
        total_norm = 0.0
        if grad_norm and grad_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        return float(total_norm)
