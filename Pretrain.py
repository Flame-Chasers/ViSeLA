# Pretrain Script for ViSeLA with Refiner Support
# Based on X2-VLM Pretrain.py, adapted for ViSeLA architecture

import argparse
import os
import sys

import yaml
from easydict import EasyDict
import numpy as np
import random
import time
import datetime
import json
import math

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.optim import Optimizer

import utils
from dataset import create_dataset
from scheduler import create_scheduler
from optim import create_optimizer

from utils.checkpointer import Checkpointer
from utils.hdfs_io import hmkdir, hcopy
from accelerators.torch_ddp_accelerator import TorchDDPAccelerator


def reinit_scheduler_properties_mysched(optimizer: Optimizer, scheduler, cfg) -> None:
    """
    with ApexDDP, do re-init to avoid lr_scheduler warning.
    """
    args = cfg

    if scheduler.optimizer == optimizer:
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_training_steps - current_step) / float(
                    max(1, args.num_training_steps - args.num_warmup_steps))
            )

        scheduler.__init__(optimizer, lr_lambda, last_epoch=-1)


def run_image_iter(model, image_batch, optimizer, accelerator, metric_logger, device, config,
                   ret_match_loss=True, return_loss_only=False):
    image, batch = image_batch[0].to(device, non_blocking=True), [t.to(device) if t is not None else None for t in image_batch[1:]]
    text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, idx = batch

    with accelerator.autocast_context():
        loss = model(image, text_ids, text_atts, text_ids_masked=text_ids_masked,
                     masked_pos=masked_pos, masked_ids=masked_ids, ret_match_loss=ret_match_loss, idx=idx)

    if return_loss_only:
        return loss

    optimizer.zero_grad()

    # Base losses
    loss_in_total = loss['loss_itc'] + loss['loss_itm'] + loss['loss_mlm']

    # Add refiner loss if present
    loss_refine_weight = config.get('loss_refine_weight', 1.0)
    if 'loss_refine' in loss:
        loss_in_total = loss_in_total + loss_refine_weight * loss['loss_refine']

    accelerator.backward_step(loss_in_total, optimizer)

    accelerator_clip_grad_norm = float(config['accelerator'].get('CLIP_GRAD_NORM', 0.0))
    accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)

    metric_logger.update(loss_itc=loss['loss_itc'].item())
    metric_logger.update(loss_itm=loss['loss_itm'].item())
    metric_logger.update(loss_mlm=loss['loss_mlm'].item())
    if 'loss_refine' in loss:
        metric_logger.update(loss_refine=loss['loss_refine'].item())


def train(model, image_loader, optimizer, epoch_info, device, scheduler, config, accelerator, checkpointer):
    model.train()
    start_epoch, max_epoch = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_refine', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('lr_large', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train step: [{}]'.format(start_epoch)
    print_freq = 50

    world_size = utils.get_world_size()
    step_per_epoch = math.ceil(config['train_dataset_size']/(config['batch_size']*world_size))
    assert step_per_epoch > 1
    global_step = start_epoch * step_per_epoch

    stop_calc_itm = config.get('stop_calc_itm', float('inf'))
    print(f"### Stop Calculate Matching Loss After {stop_calc_itm} Steps", flush=True)

    for i, batch in enumerate(metric_logger.log_every(image_loader, print_freq, header, step_per_epoch, epoch_info)):

        with torch.no_grad():
            # Compatible with both DDP and non-DDP wrapped models
            temp_param = model.module.temp if hasattr(model, 'module') else model.temp
            temp_param.clamp_(0.001, 0.5)

        ret_match_loss = global_step < stop_calc_itm

        run_image_iter(model, batch, optimizer, accelerator, metric_logger, device, config,
                       ret_match_loss=ret_match_loss)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_large=optimizer.param_groups[2]["lr"])
        scheduler.step()

        current_epoch = global_step // step_per_epoch
        if (global_step+1) % step_per_epoch == 0:
            if utils.is_main_process():
                train_stats = {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch,
                             }

                with open("log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if ((current_epoch+1) % config['ckpt_frequent'] == 0) or (current_epoch+1 == config['schedular']['epochs']):
                    model_without_ddp = model
                    if hasattr(model, 'module'):
                        model_without_ddp = model.module

                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'config': config,
                    }
                    checkpointer.save_checkpoint(model_state=save_obj,
                                                 epoch=current_epoch,
                                                 training_states=optimizer.state_dict())

            dist.barrier()

        if (global_step+1) % config['ckpt_frequent_step'] == 0:
            if utils.is_main_process():
                model_without_ddp = model
                if hasattr(model, 'module'):
                    model_without_ddp = model.module

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'config': config,
                }

                checkpointer.save_checkpoint(model_state=save_obj,
                                             epoch=current_epoch, step=global_step,
                                             training_states=optimizer.state_dict())

            dist.barrier()

        if config['schedular'].get('num_training_steps', False) and (global_step+1 >= config['schedular']['num_training_steps']):
            break

        global_step += 1

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    config['batch_size'] = config['images']['batch_size']

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating dataset", flush=True)
    datasets = create_dataset('pretrain', config)
    image_dataset = datasets[0]

    if utils.is_main_process():
        print(f"### images: {config['train_file']}", flush=True)
        print(f"### batch size, {config['batch_size']} x {int(os.environ.get('WORLD_SIZE', 1))}")

    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=config['images']['batch_size'],
                                               num_workers=config['images']['num_workers'],
                                               pin_memory=True,
                                               drop_last=False,
                                               collate_fn=image_dataset.collate_fn)

    print(f"Creating model {config.get('model_type', 'XVLMPretrain')}", flush=True)

    from models.model_pretrain import XVLMPretrain

    if os.path.exists(args.checkpoint):
        model = XVLMPretrain(config=config, load_vision_params=False, load_text_params=False, pretraining=False)
        model.load_pretrained(args.checkpoint, config, is_domain_pretrain=True)
    else:
        model = XVLMPretrain(config=config)

    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    arg_sche['step_per_epoch'] = math.ceil(config['train_dataset_size'] / (config['batch_size'] * world_size))
    arg_sche['min_rate'] = config['min_lr'] / arg_opt['lr'] if 'min_lr' in config else 0
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    arg_acc = utils.AttrDict(config['accelerator'])
    accelerator = TorchDDPAccelerator(arg_acc, logger=None)

    model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, world_size, rank)
    reinit_scheduler_properties_mysched(optimizer, lr_scheduler, arg_sche)

    checkpointer = Checkpointer(args.output_dir)

    print("### output_dir, ", args.output_dir, flush=True)
    start_time = time.time()

    start_epoch = 0
    max_epoch = config['schedular']['epochs']

    if args.resume:
        training_state_path = os.path.join(args.output_dir, 'training_state_latest.th')
        if os.path.exists(training_state_path):
            print(f"### Resuming from {training_state_path}", flush=True)
            training_state = torch.load(training_state_path, map_location='cpu')
            optimizer.load_state_dict(training_state)
            start_epoch = training_state.get('epoch', 0) + 1
            total_steps_done = start_epoch * arg_sche['step_per_epoch']
            for _ in range(total_steps_done):
                lr_scheduler.step()
            print(f"### Resumed at epoch {start_epoch}, scheduler advanced {total_steps_done} steps", flush=True)
        else:
            print(f"### No training state found at {training_state_path}, starting from scratch", flush=True)

    epoch_info = (start_epoch, max_epoch)

    print("Start training", flush=True)
    train(model, image_loader, optimizer, epoch_info, device, lr_scheduler, config, accelerator, checkpointer)
    dist.barrier()

    if utils.is_main_process():
        os.system("cat log.txt")
        hcopy('log.txt', args.output_dir)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), flush=True)

    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='output/pretrain')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--override_cfg', default="", type=str, help="Use ; to separate keys")
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
    parser.add_argument('--resume', action='store_true', help='resume training from latest checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    utils.update_config(config, args.override_cfg)
    if utils.is_main_process():
        print('config:', json.dumps(config))

    hmkdir(args.output_dir)

    yaml.dump(config, open('config.yaml', 'w'))
    hcopy('config.yaml', args.output_dir)

    config = EasyDict(config)
    main(args, config)
