import os
import random
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import json
from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import cclm_test, metric_eval, save_query_artifact
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
# from model.tbps_model import clip_vitb
from transformers import XLMRobertaTokenizer
from misc.cclm_utils import get_rank, is_main_process

# ---------------------------------------------------------------------------
# Optional environment variables (the legacy ``MMTBPR_*`` prefix is still accepted):
#   VISELA_RUN_NAME               save checkpoints under ``<saved_path>/<run_name>``
#   VISELA_SAVED_PATH             override ``model.saved_path``
#   VISELA_SEED                   override ``misc.seed``
#   VISELA_EVAL_EPOCHS            comma-separated epochs to evaluate during training
#   VISELA_SAVE_EPOCH_CHECKPOINTS save a checkpoint at every evaluated epoch (1/true/yes)
#   VISELA_SAVE_QUERY_ARTIFACT    dump per-query retrieval hits after evaluation (1/true/yes)
#   VISELA_QUERY_ARTIFACT_METHOD  method name recorded in the query artifact (default: ViSeLA)
# ---------------------------------------------------------------------------


def get_project_env(name: str, default=None):
    """Read ``VISELA_<name>``, falling back to the legacy ``MMTBPR_<name>``."""
    for prefix in ('VISELA_', 'MMTBPR_'):
        value = os.environ.get(prefix + name)
        if value is not None:
            return value
    return default


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def build_tokenizer(text_encoder: str):
    tokenizer = XLMRobertaTokenizer.from_pretrained(text_encoder)
    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token, 'eos_token': tokenizer.sep_token})
    return tokenizer


def parse_epoch_list(raw: str):
    if not raw:
        return set()

    epochs = set()
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        epochs.add(int(part))
    return epochs


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def save_checkpoint(model, optimizer, config, filename: str):
    if is_master():
        save_obj = {
            'state_dict': unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
        }
        output_path = os.path.join(config.model.saved_path, filename)
        torch.save(save_obj, output_path)
        print(f"Saved checkpoint: {output_path}")

    if is_using_distributed():
        torch.distributed.barrier()


def _save_rng_states():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_states(states):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch_cpu'])
    if states['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(states['torch_cuda'])


def evaluate_model(model, tokenizer, dataloader, config, epoch_label=None):
    rng_states = _save_rng_states()
    try:
        test_loaders = dataloader['test_loaders']
        results = {'languages': {}, 'avg': {}}
        ref_model = unwrap_model(model)

        for lan, test_loader in test_loaders.items():
            score_matrix_t2i = cclm_test(ref_model, test_loader, tokenizer, config)
            if is_master():
                dataset = test_loader.dataset
                eval_result = metric_eval(score_matrix_t2i, dataset.img2person, dataset.txt2person, config)
                results['languages'][lan] = {
                    'r1': float(eval_result['r1']),
                    'r5': float(eval_result['r5']),
                    'r10': float(eval_result['r10']),
                    'mAP': float(eval_result['mAP']),
                }
                print('{sl}: Acc@1 {top1:.5f} Acc@5 {top5:.5f} Acc@10 {top10:.5f} mAP {mAP:.5f}'
                      .format(sl=lan,
                              top1=eval_result['r1'],
                              top5=eval_result['r5'],
                              top10=eval_result['r10'],
                              mAP=eval_result['mAP']))
                if get_project_env('SAVE_QUERY_ARTIFACT', '').lower() in {'1', 'true', 'yes'}:
                    method = get_project_env('QUERY_ARTIFACT_METHOD', 'ViSeLA')
                    seed = int(getattr(config.misc, 'seed', 0))
                    artifact_path = save_query_artifact(
                        score_matrix_t2i,
                        dataset.img2person,
                        dataset.txt2person,
                        config,
                        method=method,
                        seed=seed,
                        language=lan,
                    )
                    print(f"Saved query artifact: {artifact_path}")

        if is_master() and results['languages']:
            num_langs = len(results['languages'])
            results['avg'] = {
                'r1': sum(item['r1'] for item in results['languages'].values()) / num_langs,
                'r5': sum(item['r5'] for item in results['languages'].values()) / num_langs,
                'r10': sum(item['r10'] for item in results['languages'].values()) / num_langs,
                'mAP': sum(item['mAP'] for item in results['languages'].values()) / num_langs,
            }
            prefix = f"{epoch_label} " if epoch_label else ""
            print(("{prefix}Avg: Acc@1 {top1:.5f} Acc@5 {top5:.5f} "
                   "Acc@10 {top10:.5f} mAP {mAP:.5f}").format(
                       prefix=prefix,
                       top1=results['avg']['r1'],
                       top5=results['avg']['r5'],
                       top10=results['avg']['r10'],
                       mAP=results['avg']['mAP'],
                   ))
            torch.cuda.empty_cache()

        if is_using_distributed():
            torch.distributed.barrier()

        return results
    finally:
        _restore_rng_states(rng_states)


def sync_loaded_model_from_rank0(model, config, load_method: str):
    if not is_using_distributed():
        getattr(model, load_method)(config)
        return

    if is_master():
        getattr(model, load_method)(config)

    # Only rank 0 reads the large checkpoint from disk; other ranks wait here.
    torch.distributed.barrier()

    for tensor in model.state_dict().values():
        if torch.is_tensor(tensor):
            torch.distributed.broadcast(tensor, src=0)

    init_params_payload = [getattr(model, 'init_params', None) if is_master() else None]
    torch.distributed.broadcast_object_list(
        init_params_payload,
        src=0,
        device=torch.device(config.device),
    )
    if init_params_payload[0] is not None:
        model.init_params = init_params_payload[0]

    torch.distributed.barrier()


def run(config):
    print(config)

    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)

    meters = {
        "loss": AverageMeter(),
        "loss_itc": AverageMeter(),
        "loss_sdm": AverageMeter(),
        "loss_itm": AverageMeter(),
        "loss_mlm": AverageMeter(),
        "loss_mim": AverageMeter(),
        "loss_id": AverageMeter(),
        "loss_aux": AverageMeter(),
        "precision": AverageMeter(),
    }

    best_rank_1 = 0.0
    best_epoch = 0
    best_rank_1_tl = 0.0
    best_epoch_tl = 0

    from models.model_retrieval import ViSeLA
    model = ViSeLA(config=config, num_classes=num_classes)
    model = model.to(config.device)
    sync_loaded_model_from_rank0(model, config, load_method='load_pretrained')
    tokenizer = build_tokenizer(config['text_encoder'])

    if is_using_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device],
                                                          find_unused_parameters=True)

    # schedule
    config.schedule.niter_per_ep = len(train_loader)
    lr_schedule = cosine_scheduler(config)

    # optimizer
    optimizer = build_optimizer(config, model)

    # Diversity warm-up: ramp the diversity constraints linearly so they do not
    # conflict with the main objectives early in training.
    div_warmup_epochs = int(getattr(config.refiner, 'div_warmup_epochs', 0) or 0)
    base_lambda_feat_div = float(getattr(config.refiner, 'lambda_feat_div', 0.0) or 0.0)
    base_lambda_attn_div = float(getattr(config.refiner, 'lambda_attn_div', 0.0) or 0.0)

    def _set_div_lambdas(epoch_idx: int):
        # scale in (0, 1]: increases linearly per epoch until the warm-up ends
        if div_warmup_epochs > 0:
            scale = min(1.0, float(epoch_idx + 1) / div_warmup_epochs)
        else:
            scale = 1.0
        # Works for both DDP-wrapped and plain models
        refiner = model.module.refiner if hasattr(model, 'module') else model.refiner
        if refiner is None:
            return
        ref_cfg = refiner.config
        ref_cfg.lambda_feat_div = base_lambda_feat_div * scale
        ref_cfg.lambda_attn_div = base_lambda_attn_div * scale

    # train
    it = 0
    scaler = torch.cuda.amp.GradScaler()
    epoch_times = []  # per-epoch wall time, used for the ETA estimate
    training_start_time = time.time()
    eval_epochs = parse_epoch_list(get_project_env('EVAL_EPOCHS', ''))
    save_epoch_checkpoints = get_project_env('SAVE_EPOCH_CHECKPOINTS', '').lower() in {'1', 'true', 'yes'}
    epoch_eval_records = []

    if is_master() and eval_epochs:
        print(f"Per-epoch evaluation enabled for epochs: {sorted(eval_epochs)}")
        print(f"Per-epoch checkpoint saving: {save_epoch_checkpoints}")

    for epoch in range(config.schedule.epoch):

        _set_div_lambdas(epoch)

        if is_using_distributed():
            dataloader['train_sampler'].set_epoch(epoch)

        epoch_start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for i, batch in enumerate(train_loader):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[it] * param_group['ratio']

            with torch.autocast(device_type='cuda'):
                (image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, idx, lang_ids) = None, None, None, None, None, None, None, None

                if config.mlm.is_mlm:
                    (image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids, idx, lang_ids) = batch
                else:
                    image, text_ids, text_atts, idx, lang_ids = batch
                ret = model(image, text_ids, text_atts, text_ids_masked=text_ids_masked, masked_pos=masked_pos, masked_ids=masked_ids, idx=idx, lang_ids=lang_ids)

                loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = image.shape[0]
            meters['loss'].update(loss.item(), batch_size)
            meters['loss_itc'].update(ret.get('loss_itc', 0), batch_size)
            meters['loss_sdm'].update(ret.get('loss_sdm', 0), batch_size)
            meters['loss_itm'].update(ret.get('loss_itm', 0), batch_size)
            meters['loss_mlm'].update(ret.get('loss_mlm', 0), batch_size)
            meters['loss_id'].update(ret.get('loss_id', 0), batch_size)
            meters['loss_aux'].update(ret.get('loss_aux', 0), batch_size)
            meters['precision'].update(ret.get('precision', 0), batch_size)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.zero_grad()
            optimizer.zero_grad()
            it += 1

            if (i + 1) % config.log.print_period == 0:
                # ETA for the current epoch
                elapsed_batches = i + 1
                total_batches = len(train_loader)
                batch_time_avg = (time.time() - epoch_start_time) / elapsed_batches
                remaining_batches = total_batches - elapsed_batches
                epoch_eta_seconds = batch_time_avg * remaining_batches
                epoch_eta_str = str(timedelta(seconds=int(epoch_eta_seconds)))

                info_str = f"Epoch[{epoch + 1}] Iteration[{i + 1}/{len(train_loader)}]"
                # log loss
                for k, v in meters.items():
                    if v.val != 0:
                        info_str += f", {k}: {v.val:.4f}"
                info_str += f", Base Lr: {param_group['lr']:.2e}"
                info_str += f", Epoch ETA: {epoch_eta_str}"
                print(info_str)

        # End-of-epoch statistics
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # ETA for the whole run
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = config.schedule.epoch - (epoch + 1)
        total_eta_seconds = avg_epoch_time * remaining_epochs
        total_eta_str = str(timedelta(seconds=int(total_eta_seconds)))

        # Estimated wall-clock finish time
        estimated_finish_time = datetime.now() + timedelta(seconds=total_eta_seconds)
        finish_time_str = estimated_finish_time.strftime('%Y-%m-%d %H:%M:%S')

        # Throughput summary
        time_per_batch = epoch_duration / len(train_loader)
        samples_per_sec = train_loader.batch_size / time_per_batch

        print(f"Epoch {epoch + 1} done. Time: {epoch_duration:.1f}s, Speed: {samples_per_sec:.1f} samples/s")
        if remaining_epochs > 0:
            print(f"Avg epoch time: {avg_epoch_time:.1f}s, Remaining: {remaining_epochs} epochs, "
                  f"Total ETA: {total_eta_str}, Est. finish: {finish_time_str}")

        current_epoch = epoch + 1
        if current_epoch in eval_epochs:
            if save_epoch_checkpoints:
                save_checkpoint(model, optimizer, config, f'checkpoint_epoch_{current_epoch:02d}.pth')

            eval_record = evaluate_model(model, tokenizer, dataloader, config, epoch_label=f'Epoch {current_epoch}')
            if is_master():
                eval_record['epoch'] = current_epoch
                epoch_eval_records.append(eval_record)
                summary_path = os.path.join(config.model.saved_path, 'epoch_eval_results.json')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(epoch_eval_records, f, indent=2, ensure_ascii=False)

    if config.schedule.epoch not in eval_epochs:
        evaluate_model(model, tokenizer, dataloader, config, epoch_label='Final')

    save_checkpoint(model, optimizer, config, 'checkpoint.pth')

    # print(f"{sl} best Acc@1: {best_rank_1} at epoch {best_epoch + 1}")
    # print(f"{tl} best Acc@1: {best_rank_1_tl} at epoch {best_epoch_tl + 1}")


def run_eval_only(config):
    print(config)

    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)

    from models.model_retrieval import ViSeLA
    model = ViSeLA(config=config, num_classes=num_classes)
    model = model.to(config.device)
    sync_loaded_model_from_rank0(model, config, load_method='load_finetune')
    tokenizer = build_tokenizer(config['text_encoder'])

    if is_using_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device],
                                                          find_unused_parameters=True)

    evaluate_model(model, tokenizer, dataloader, config, epoch_label='Eval')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ViSeLA training and evaluation')
    parser.add_argument('--config', type=str, default='config/examples/visela_cuhk_pedes.yaml',
                        help='path to the YAML config (default: config/examples/visela_cuhk_pedes.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path for eval-only mode')
    parser.add_argument('--saved-path', type=str, default=None,
                        help='output directory override')
    parser.add_argument('--eval-only', action='store_true',
                        help='run evaluation without training')
    args = parser.parse_args()

    config_path = args.config
    config = parse_config(config_path)
    if args.resume:
        config.model.resume = args.resume
    if args.saved_path:
        config.model.saved_path = args.saved_path
    saved_path_override = get_project_env('SAVED_PATH')
    if saved_path_override:
        config.model.saved_path = saved_path_override
    seed_override = get_project_env('SEED')
    if seed_override:
        config.misc.seed = int(seed_override)

    # ========================================================================
    # Run name: read from the environment and create a dedicated sub-directory
    # ========================================================================
    # When VISELA_RUN_NAME is set, checkpoints are saved under that sub-directory
    # so that different runs get independent, easy-to-trace checkpoint folders.
    run_name = get_project_env('RUN_NAME')
    if run_name:
        # Sanitize the run name: keep only safe characters to prevent path traversal
        import re
        run_name = re.sub(r'[^A-Za-z0-9._+-]', '_', run_name)
        # Strip leading dots/dashes to avoid hidden files and relative paths
        run_name = re.sub(r'^[._-]+', '', run_name)
        # Fall back to a default if nothing is left after sanitizing
        if not run_name:
            run_name = 'default_run'

        original_saved_path = config.model.saved_path
        config.model.saved_path = os.path.join(original_saved_path, run_name)

    Path(config.model.saved_path).mkdir(parents=True, exist_ok=True)

    init_distributed_mode(config)

    # Print run info after distributed init so it appears once, on the main process only
    if run_name and is_main_process():
        print("=" * 72)
        print(f"Run name: {run_name}")
        print(f"Checkpoint directory: {config.model.saved_path}")
        print("=" * 72)

    set_seed(config)

    if args.eval_only:
        run_eval_only(config)
    else:
        run(config)
