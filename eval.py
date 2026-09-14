"""Attention-visualization tool for ViSeLA.

This script loads a fine-tuned checkpoint and renders the refiner's query-to-token
attention as multi-language heatmaps (one sub-plot per language, shared colorbar).

It is *not* the metric evaluation entry point. To compute Rank-1/5/10 and mAP, run::

    python main.py --config config/examples/visela_cuhk_pedes.yaml --eval-only --resume <checkpoint>
"""
import os
import random
import time
from pathlib import Path
from collections import OrderedDict

import torch
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # must be set before importing seaborn/pyplot
import seaborn as sns
import matplotlib.pyplot as plt
import re

from misc.build import load_checkpoint, cosine_scheduler, build_optimizer
from misc.data import build_pedes_data
from misc.eval import cclm_test, metric_eval, get_refiner_attn
from misc.utils import parse_config, init_distributed_mode, set_seed, is_master, is_using_distributed, \
    AverageMeter
# from model.tbps_model import clip_vitb
from transformers import XLMRobertaTokenizer
from misc.cclm_utils import get_rank

# +++ Visualization helpers: begin +++

# Use a font that covers CJK characters so Chinese tokens render in the heatmaps.
# Install it if missing, e.g. the 'Noto Sans CJK JP' package for your platform.
try:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Font setup warning: {e}")
    print("CJK characters in the heatmaps may not render. Please install the 'Noto Sans CJK JP' font.")

def draw_subplot_heatmap(ax, tokens, attention_map, title, cbar=False, vmin=None, vmax=None):
    """
    Draw one attention heatmap on the given Matplotlib Axes.
    Wide horizontal layout with large fonts.
    """
    num_queries, _ = attention_map.shape

    sns.heatmap(
        attention_map,
        xticklabels=tokens,
        yticklabels=[f"Q{i+1}" for i in range(num_queries)],
        cmap="viridis",
        linewidths=.5,
        annot=False,
        ax=ax,
        cbar=cbar,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title, fontsize=52, fontweight='bold', pad=30)
    ax.set_xlabel("", fontsize=1)
    ax.set_ylabel("")

    ax.tick_params(axis='x', labelsize=38, rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), ha="center")
    ax.tick_params(axis='y', labelsize=44, rotation=0)


def draw_multi_language_heatmap(all_lang_data, save_path):
    """
    Draw a single figure with one heatmap sub-plot per language.
    Sub-plots are stacked vertically and share one colorbar.
    """
    num_langs = len(all_lang_data)
    if num_langs == 0:
        print("No attention data available for visualization.")
        return

    # First pass: merge sub-word tokens and compute the global vmin/vmax
    processed_data = []
    global_vmin, global_vmax = float('inf'), float('-inf')

    for lang, data in all_lang_data.items():
        raw_tokens = data["tokens"]
        attention_map = data["attn"]

        indices_to_keep = [
            i for i, token in enumerate(raw_tokens)
            if token not in ['<pad>', '<s>', '</s>']
        ]

        if not indices_to_keep:
            processed_data.append((lang, None, None))
            continue

        # Keep only valid tokens and their attention columns
        filtered_tokens = [raw_tokens[i] for i in indices_to_keep]
        filtered_attn = attention_map[:, indices_to_keep]

        # Merge sub-words: tokens without a leading "▁" are appended to the previous word
        # (non-CJK only). The Chinese tokenizer emits a standalone "▁" at the start of a
        # sentence and no prefix on the following characters.
        merged_tokens = []
        merged_attn_list = []
        current_token = ""
        current_attn_parts = []
        next_is_new_word = False  # whether the next token should start a new word

        def is_chinese_char(char):
            """Return True if the character is in the CJK Unified Ideographs block."""
            return '\u4e00' <= char <= '\u9fff'

        for i, token in enumerate(filtered_tokens):
            clean_token = token.lstrip('▁').lstrip('_').replace(' ', '').strip()

            # Word start is marked by the "▁" prefix
            is_word_start = token.startswith('▁') or token.startswith('_')

            # A standalone "▁" (empty after cleaning) marks the next token as a new word
            if not clean_token:
                if current_token:
                    merged_tokens.append(current_token)
                    merged_attn_list.append(np.mean(current_attn_parts, axis=0))
                    current_token = ""
                    current_attn_parts = []
                next_is_new_word = True
                continue

            # CJK tokens: every character becomes its own word
            if is_chinese_char(clean_token[0]):
                # Flush the word accumulated so far
                if current_token:
                    merged_tokens.append(current_token)
                    merged_attn_list.append(np.mean(current_attn_parts, axis=0))
                    current_token = ""
                    current_attn_parts = []
                # Emit the CJK token as an independent word
                merged_tokens.append(clean_token)
                merged_attn_list.append(filtered_attn[:, i])
                next_is_new_word = False
            elif is_word_start or next_is_new_word:
                # Latin-script languages: a prefix (or a preceding standalone "▁") starts a new word
                if current_token:
                    merged_tokens.append(current_token)
                    merged_attn_list.append(np.mean(current_attn_parts, axis=0))
                current_token = clean_token
                current_attn_parts = [filtered_attn[:, i]]
                next_is_new_word = False
            else:
                # Continuation sub-word: append to the current word
                current_token += clean_token
                current_attn_parts.append(filtered_attn[:, i])

        # Flush the last word
        if current_token:
            merged_tokens.append(current_token)
            merged_attn_list.append(np.mean(current_attn_parts, axis=0))

        # Convert to a numpy array: [num_queries, num_merged_tokens]
        effective_attention_map = np.stack(merged_attn_list, axis=1) if merged_attn_list else None
        display_tokens = merged_tokens

        if effective_attention_map is None:
            processed_data.append((lang, None, None))
            continue

        global_vmin = min(global_vmin, effective_attention_map.min())
        global_vmax = max(global_vmax, effective_attention_map.max())

        processed_data.append((lang, effective_attention_map, display_tokens))

    # Create the figure, leaving room on the right for the shared colorbar
    fig, axes = plt.subplots(num_langs, 1, figsize=(70, 10 * num_langs))

    if num_langs == 1:
        axes = [axes]

    # Draw every sub-plot without its own colorbar
    lang_full_names = {'en': 'English', 'ch': 'Chinese', 'fr': 'French', 'de': 'German'}
    for (lang, attn_map, tokens), ax in zip(processed_data, axes):
        if attn_map is None:
            ax.set_title(f"{lang_full_names.get(lang, lang.upper())} - No valid tokens", fontsize=42)
            ax.axis('off')
            continue

        draw_subplot_heatmap(ax, tokens, attn_map, lang_full_names.get(lang, lang.upper()),
                            cbar=False, vmin=global_vmin, vmax=global_vmax)

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=44)

    plt.tight_layout(rect=[0, 0, 0.92, 1], pad=4.0, h_pad=6.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved multi-language attention heatmap to: {save_path}")
    plt.close(fig)

# +++ Visualization helpers: end +++


def _safe_name(name: str) -> str:
    """Sanitize a file name by removing unsafe characters."""
    name = re.sub(r'[^A-Za-z0-9._+-]', '_', name)
    name = re.sub(r'^[._-]+', '', name)
    return name or "unknown"


def _reduce_attn(attn, reduction: str = "mean"):
    """Reduce attention weights to a [num_queries, seq_len] array.

    Args:
        attn: attention weights, either [num_queries, seq_len] or [num_heads, num_queries, seq_len]
        reduction: aggregation method; only "mean" is supported

    Returns:
        Reduced attention weights of shape [num_queries, seq_len]

    Raises:
        ValueError: if the array rank or the reduction method is unsupported
    """
    arr = np.asarray(attn)

    if arr.ndim == 2:
        # Already [num_queries, seq_len]
        return arr

    if arr.ndim == 3:
        # [num_heads, num_queries, seq_len]: aggregate over heads
        if reduction == "mean":
            return arr.mean(axis=0)
        raise ValueError(f"Unsupported attention reduction: {reduction}")

    raise ValueError(f"Unsupported attention shape: {arr.shape}")


def _select_query_indices(test_loaders, num_queries: int, seed: int, explicit: str = None):
    """Select text-query indices that are valid across all languages.

    To keep the multi-language comparison aligned, indices are drawn from the
    index range shared by every language's test set.

    Args:
        test_loaders: dict of {language: DataLoader}
        num_queries: number of queries to sample
        seed: random seed
        explicit: comma-separated explicit indices; takes precedence over random sampling

    Returns:
        Sorted list of indices

    Raises:
        ValueError: if test_loaders is empty
    """
    if not test_loaders:
        raise ValueError("test_loaders is empty; cannot sample queries")

    # Shared text-index range across languages (the smallest test set)
    common_size = min(len(loader.dataset.text) for loader in test_loaders.values())

    if common_size <= 0:
        return []

    # Parse and validate explicit indices if provided
    if explicit:
        raw_indices = [s.strip() for s in explicit.split(",") if s.strip()]
        indices = []
        for s in raw_indices:
            try:
                idx = int(s)
                if 0 <= idx < common_size:
                    indices.append(idx)
            except ValueError:
                continue

        # Deduplicate, sort, and keep at most num_queries
        return sorted(set(indices))[:max(1, num_queries)]

    # Random sampling
    k = max(1, min(num_queries, common_size))
    rng = random.Random(seed)
    return sorted(rng.sample(range(common_size), k=k))


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def build_tokenizer(text_encoder: str):
    tokenizer = XLMRobertaTokenizer.from_pretrained(text_encoder)
    tokenizer.add_special_tokens({'bos_token': tokenizer.cls_token, 'eos_token': tokenizer.sep_token})
    return tokenizer


def run(config, args):
    print(config)

    # data
    dataloader = build_pedes_data(config)
    train_loader = dataloader['train_loader']
    num_classes = len(train_loader.dataset.person2text)

    from models.model_retrieval import ViSeLA
    model = ViSeLA(config=config, num_classes=num_classes)
    model.load_finetune(config)
    model = model.to(config.device)
    tokenizer = build_tokenizer(config['text_encoder'])

    if is_using_distributed():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device],
                                                          find_unused_parameters=False)
    
    # Output directory for the heatmaps
    heatmap_save_dir = os.path.join(config.model.saved_path, "text_attention_heatmaps")
    if is_master():
        os.makedirs(heatmap_save_dir, exist_ok=True)

    test_loaders = dataloader['test_loaders']

    # Sample query indices (aligned across languages)
    query_indices = _select_query_indices(
        test_loaders=test_loaders,
        num_queries=args.vis_num_queries,
        seed=args.vis_seed,
        explicit=args.vis_query_indices,
    )

    if not query_indices:
        raise ValueError(
            "Could not select any valid query index. "
            "Check that the dataset is not empty and that --vis-query-indices is valid."
        )

    ref_model = model.module if is_using_distributed() else model
    exp_name = _safe_name(Path(config.model.resume).parent.name)

    if is_master():
        print("\nVisualization settings:")
        print(f"  - run name: {exp_name}")
        print(f"  - number of queries: {len(query_indices)}")
        print(f"  - query indices: {query_indices}")
        print(f"  - seed: {args.vis_seed}")
        print(f"  - attention reduction: {args.vis_attn_reduction}")
        print()

    # One multi-language comparison figure per sampled query
    for text_idx in query_indices:
        all_attention_data = OrderedDict()
        pid_for_name = None

        # Collect attention for every language
        for lan, test_loader in test_loaders.items():
            tokens, txt_attn, pid = get_refiner_attn(
                ref_model, test_loader, tokenizer, config, text_index=text_idx
            )
            # Reduce to [num_queries, seq_len]
            txt_attn = _reduce_attn(txt_attn, reduction=args.vis_attn_reduction)
            all_attention_data[lan] = {"tokens": tokens, "attn": txt_attn}

            # Use the first language's person id in the file name
            if pid_for_name is None:
                pid_for_name = pid

        if is_master():
            # File name: multi_lang_attn_<exp>_txt{idx:06d}_pid{pid}
            pid_part = f"_pid{pid_for_name}" if pid_for_name is not None else ""
            base_filename = f"multi_lang_attn_{exp_name}_txt{text_idx:06d}{pid_part}"

            # Save both PDF and PNG
            for ext in ['.pdf', '.png']:
                filename = base_filename + ext
                save_path = os.path.join(heatmap_save_dir, filename)
                print(f"Generating: {filename}")
                draw_multi_language_heatmap(all_attention_data, save_path)

    if is_master():
        print("\nAll attention heatmaps generated.")
        print(f"Output directory: {heatmap_save_dir}")
        print(f"Generated {len(query_indices)} multi-language comparison figure(s)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='ViSeLA refiner attention visualization (metric evaluation lives in main.py --eval-only)')
    parser.add_argument('--config', type=str, default='config/examples/visela_cuhk_pedes.yaml',
                        help='path to the YAML config (default: config/examples/visela_cuhk_pedes.yaml)')
    parser.add_argument('--exp-name', '--name', type=str, default=None,
                        help='run name used to build the checkpoint path: <saved_path>/<exp-name>/checkpoint.pth')
    parser.add_argument('--resume', type=str, default=None,
                        help='explicit checkpoint path (overrides the config)')
    parser.add_argument('--saved-path', type=str, default=None,
                        help='output directory (overrides the config)')

    # Visualization options
    parser.add_argument('--vis-num-queries', type=int, default=3,
                        help='number of text queries to sample')
    parser.add_argument('--vis-seed', type=int, default=1,
                        help='random seed for query sampling')
    parser.add_argument('--vis-query-indices', type=str, default=None,
                        help='explicit comma-separated query indices; overrides random sampling, e.g. 0,10,20')
    parser.add_argument('--vis-attn-reduction', type=str, default='mean', choices=['mean'],
                        help='attention reduction over heads (default: mean -> [num_queries, seq_len])')

    args = parser.parse_args()

    config_path = args.config
    config = parse_config(config_path)

    # ========================================================================
    # Checkpoint selection: command-line arguments take precedence over the config
    # ========================================================================
    if args.resume:
        # Explicit checkpoint path
        config.model.resume = args.resume
        print(f"Using checkpoint from command line: {args.resume}")
    elif args.exp_name:
        # Build the path from the run name
        import re
        exp_name = re.sub(r'[^A-Za-z0-9._+-]', '_', args.exp_name)
        exp_name = re.sub(r'^[._-]+', '', exp_name)
        if not exp_name:
            print("Error: --exp-name contains no valid characters")
            exit(1)

        # Training saves to saved_path/<exp_name>/checkpoint.pth; mirror that layout here
        original_saved_path = config.model.saved_path

        config.model.resume = os.path.join(original_saved_path, exp_name, 'checkpoint.pth')
        config.model.saved_path = os.path.join(original_saved_path, exp_name)

        print(f"Run name: {exp_name}")
        print(f"Checkpoint: {config.model.resume}")
        print(f"Output directory: {config.model.saved_path}")

    # saved_path may be overridden independently
    if args.saved_path:
        config.model.saved_path = args.saved_path
        print(f"Using output directory from command line: {args.saved_path}")

    Path(config.model.saved_path).mkdir(parents=True, exist_ok=True)

    init_distributed_mode(config)

    set_seed(config)

    run(config, args)
