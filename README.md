# ViSeLA: Vision-Grounded Semantic Learning and Alignment for Scalable Multilingual Text-Based Person Retrieval

**[ACM MM 2026]** Official PyTorch implementation of **ViSeLA**. [[OpenReview]](https://openreview.net/forum?id=mKcbH8tKps)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 1.13](https://img.shields.io/badge/PyTorch-1.13-EE4C2C.svg)](https://pytorch.org/)

Resources: [Dataset Download](#dataset-download) · [Released Checkpoints](#released-checkpoints) · [Release v1.0.0](https://github.com/Flame-Chasers/ViSeLA/releases/tag/v1.0.0)

---

## Overview

Multilingual text-based person retrieval (TBPR) retrieves a target person from an image gallery using free-form descriptions written in different languages. Prior work is almost entirely English-only, and the single previous multilingual attempt lacked both a large-scale multilingual pretraining corpus and an architecture designed for multilingual text encoding and cross-modal alignment.

This repository contains the official implementation of **ViSeLA** (Vision-grounded Semantic Learning and Alignment). ViSeLA extends the mixture-of-experts architecture with a *semantic-aware multilingual learning module* that routes equivalent semantics across languages to specialized experts, and adds a *vision-grounded cross-modal alignment module* that uses a semantic probe to capture fine-grained semantics across modalities. The paper also introduces **MultiPedes**, the first large-scale pedestrian-centric multilingual pretraining dataset: 8 million image-text pairs in four languages (English, Chinese, French, German), built by a simple, fully automated pipeline. ViSeLA is pretrained on MultiPedes and fine-tuned on the multilingual and English-only TBPR benchmarks below.

---

## Environment

All experiments were run on 4 NVIDIA A40 (48 GB) GPUs with CUDA 11.7, Python 3.10, and PyTorch 1.13.

Create the environment with conda:

```sh
conda env create -f environment_tbps_py310.yml
conda activate tbps_py310
```

or install the pinned packages with pip (the file includes the PyTorch CUDA 11.7 index):

```sh
pip install -r requirements.txt
```

Regenerating code-switch files additionally requires `spacy` and its language models; see the comments at the end of `requirements.txt`.

---

## Download

### Dataset Download

| Download | Contents |
| --- | --- |
| [visela_downstream_multilingual_annos.tar.gz](https://github.com/Flame-Chasers/ViSeLA/releases/download/v1.0.0/visela_downstream_multilingual_annos.tar.gz) | 32 train/test JSON files for CUHK-PEDES, ICFG-PEDES, RSTPReid, and UFineBench in English, Chinese, French, and German; extracts to `DATASET/annos/`. |
| [visela_downstream_codeswitch_annos.tar.gz](https://github.com/Flame-Chasers/ViSeLA/releases/download/v1.0.0/visela_downstream_codeswitch_annos.tar.gz) | 16 code-switch training JSON files for the same four datasets and languages; extracts to `DATASET/code_switch/`. |
| [multipedes_10m_aligned_parquet.tar.gz](https://github.com/Flame-Chasers/ViSeLA/releases/download/v1.0.0/multipedes_10m_aligned_parquet.tar.gz) | 297 aligned HAM and WebPerson Parquet files, plus a portable WebPerson manifest. |
| [multipedes_ham1m_4langs_json.tar.gz](https://github.com/Flame-Chasers/ViSeLA/releases/download/v1.0.0/multipedes_ham1m_4langs_json.tar.gz) | HAM four-language JSON with 999,007 images and 1,997,789 caption values per language. |

The downstream annotation filenames use `en`, `ch`, `fr`, and `de` for English, Chinese, French, and German. UFineBench code-switch annotations cover all English and Chinese training records; the French and German files contain 45 and 314 fewer records, respectively, than their multilingual counterparts.

The aligned Parquet sources contain 1,187,383 HAM and 1,312,617 WebPerson rows per target language. Counting English once (`source_caption`) plus German, French, and Chinese yields 10,000,000 image-text pairs. These combined 10M sources and the separate HAM JSON are identified separately from the paper's 8M corpus description. The HAM JSON preserves 652 empty English caption values.

Each archive includes a guide to its layout. Obtain images separately under the original dataset terms:

- [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description)
- [ICFG-PEDES](https://github.com/zifyloo/SSAN)
- [RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset)
- [UFineBench](https://github.com/Zplusdragon/UFineBench)

Resolve relative image paths against the corresponding image roots. For pretraining, convert the annotations to the JSON Lines format in [Data Layout](#data-layout).

ICFG-PEDES and RSTPReid provide a single original annotation JSON file. To prepare English-only annotations from those sources, use:

```sh
python scripts/prepare_dataset.py --dataset icfg --icfg-json /path/to/ICFG-PEDES/ICFG-PEDES.json --output-root data
python scripts/prepare_dataset.py --dataset rstp --rstp-json /path/to/RSTPReid/data_captions.json --output-root data
```

To generate code-switch files locally, use `tools/offline_code_switch.py`, which uses `facebook/m2m100_418M` for word-level translation:

```sh
python tools/offline_code_switch.py \
  --lang en \
  --input /path/to/CUHK-PEDES/annos/train_reid_en.json \
  --output /path/to/CUHK-PEDES/code_switch/train_reid_en_cs.json \
  --device cuda:0 --num-variants 2 --seed 42
```

Generate all four languages of one dataset with the batch scripts by setting `INPUT_DIR` and `OUTPUT_DIR`:

```sh
INPUT_DIR=/path/to/CUHK-PEDES/annos OUTPUT_DIR=/path/to/CUHK-PEDES/code_switch \
  bash scripts/generate_multilingual_cs.sh
INPUT_DIR=/path/to/UFineBench/annos OUTPUT_DIR=/path/to/UFineBench/code_switch \
  bash scripts/generate_ufine_cs.sh
```

### Released Checkpoints

The following checkpoints include their training configurations. Use the embedded configuration for each model and adapt its relative paths to your installation.

| Download | Training configuration | Checkpoint contents |
| --- | --- | --- |
| [visela_pretrain_ham_luperson_refiner_ep1.th](https://github.com/Flame-Chasers/ViSeLA/releases/download/v1.0.0/visela_pretrain_ham_luperson_refiner_ep1.th) | HAM multilingual shards and LUPerson bilingual data, epoch 1; four experts, four queries, `refiner.sim_type: lsp_max`, `routing_temp: 1.0`, batch size 72 per GPU. | `model` and `config`; pretraining weights for downstream fine-tuning. |
| [visela_cuhk_pedes_baseline_router.pth](https://github.com/Flame-Chasers/ViSeLA/releases/download/v1.0.0/visela_cuhk_pedes_baseline_router.pth) | CUHK-PEDES multilingual baseline initialized from CCLM-X2VLM; 10 epochs, four experts, four queries, `refiner.sim_type: avg`, `routing_temp: 0.85`, batch size 64 per GPU, text length 77, code-switch probability 0.3. | `state_dict` and `config`; optimizer state excluded. |

Both checkpoints retain the original FP32 model tensors and int64 buffers. The released CUHK-PEDES baseline router has the following retrieval scores from its original training log (percent):

| en R@1 | ch R@1 | fr R@1 | de R@1 | Avg R@1 | Avg mAP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 78.22 | 76.07 | 76.74 | 76.10 | 76.78 | 68.67 |

The [paper results](#results) report ViSeLA with MultiPedes pretraining.

Download [SHA256SUMS.txt](https://github.com/Flame-Chasers/ViSeLA/releases/download/v1.0.0/SHA256SUMS.txt) alongside all four data archives and both model files, then verify them with:

```sh
sha256sum -c SHA256SUMS.txt
```

### Pretrained Encoders

- **Text encoder:** [XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base).
- **CCLM-X2VLM base checkpoint:** [cclm_x2vlm_base.th](https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/x2vlm_ckpts_2release/cclm_x2vlm_base.th), the starting point for MultiPedes pretraining. It can also be used as `model.resume` to fine-tune directly.
- **BEiT-2 weights (optional):** [BEiT-2](https://github.com/microsoft/unilm/tree/master/beit2). The vision encoder configuration in `config/config_beit2_base.json` points to `data/beitv2_base_patch16_224_pt1k_ft21k.pth`. These weights are loaded when `Pretrain.py` runs without `--checkpoint`; CCLM-X2VLM and ViSeLA checkpoints already include the vision encoder.

---

## Data Layout

Arrange each fine-tuning dataset as follows (paths are configurable, this is the layout assumed by the example configs):

```text
CUHK-PEDES/
  imgs/
  annos/
    train_reid_en.json   train_reid_ch.json   train_reid_fr.json   train_reid_de.json
    test_reid_en.json    test_reid_ch.json    test_reid_fr.json    test_reid_de.json
  code_switch/
    train_reid_en_cs.json  train_reid_ch_cs.json  train_reid_fr_cs.json  train_reid_de_cs.json
```

Each annotation file is a JSON list. `file_path` is joined with `image_dir`, `id` is the person identity, and `captions` holds one or more descriptions in that file's language:

```json
{
  "file_path": "CUHK01/0363004.png",
  "id": 363,
  "captions": [
    "A man wearing a black jacket, blue jeans and white sneakers.",
    "The man has short hair and carries a backpack."
  ]
}
```

Code-switch files repeat `id` and add `captions_cs`, a list aligned with `captions` whose entries are the augmented variants of each caption.

Pretraining reads JSON Lines shards listed under `train_file` (`images.image_key: binary`, `images.caption_key: caption`, `images.is_image_rpath: true`). Each line is one JSON object with:

- `binary`: path to the image file (with `is_image_rpath: true` the value is a readable path, not base64 data);
- `caption`: a string, a list of strings (one is sampled per step), or a dictionary mapping a language code to a caption (`en`, `zh`, `de`, `fr`). The dictionary form is what MultiPedes uses; one language is sampled per step and its language id is passed to the model. Note that pretraining uses `zh` while the fine-tuning annotations use `ch`.

```json
{"binary": "/path/to/MultiPedes/images/000001.jpg", "caption": {"en": "...", "zh": "...", "fr": "...", "de": "..."}}
```

`train_dataset_size: 10000000` is the retained epoch sampling budget, while MultiPedes's
8M describes the corpus size. `Pretrain.py` computes
`steps_per_epoch = ceil(train_dataset_size / (images.batch_size * world_size))`.
The data loader repeats the listed shards and samples one caption per image visit;
an epoch therefore need not visit each unique image-text pair exactly once. With
72 samples per GPU and 4 GPUs, the example runs 34,723 steps per epoch, or 69,446
steps over 2 epochs. This setting controls the training schedule; it does not
establish an additional 2M pairs or a particular mixture of datasets.

---

## Configuration

Sanitized example configs live in `config/examples/`. They are derived from the exact configs behind the paper numbers; only path-bearing keys were replaced with placeholders.

| Example config | Dataset | `refiner.num_query` | `num_of_experts` | Notes |
| --- | --- | ---: | ---: | --- |
| `config/examples/visela_cuhk_pedes.yaml` | CUHK-PEDES (multilingual) | 4 | 4 | Table 2/4/5 ViSeLA rows, `text_length` 77 |
| `config/examples/visela_icfg_pedes.yaml` | ICFG-PEDES (multilingual) | 4 | 4 | Table 2 ViSeLA rows |
| `config/examples/visela_ufinebench.yaml` | UFineBench (multilingual) | 8 | 8 | Table 2 ViSeLA rows, `text_length` 168, batch size 32 |
| `config/examples/pretrain_multipedes.yaml` | MultiPedes | 4 | 4 | `Pretrain.py`, 2 epochs, semantic MoE |

Copy an example and fill in your local paths (copies named `local_*.yaml` are git-ignored):

```sh
cp config/examples/visela_cuhk_pedes.yaml local_visela_cuhk_pedes.yaml
```

Edit these six keys in the copy:

```yaml
model:
  resume: /path/to/visela_multipedes_pretrained.th   # or cclm_x2vlm_base.th to skip pretraining
  saved_path: ckpts/CUHK-PEDES/visela                  # where checkpoints and logs are written
text_encoder: /path/to/xlm-roberta-base
anno_dir: /path/to/CUHK-PEDES/annos
image_dir: /path/to/CUHK-PEDES/imgs
data:
  code_switch_dir: /path/to/CUHK-PEDES/code_switch
```

For the pretraining example, set `text_encoder` and list every MultiPedes shard under `train_file`; the CCLM-X2VLM checkpoint is passed on the command line.

**English-only runs (Table 3).** Use the same fine-tuning config with `data.languages: [en]`. Only the English annotation (and, if `data.use_code_switch` is `true`, the English code-switch file) is loaded.

---

## Training

Fine-tune on 4 GPUs with an edited config:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --rdzv_id=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  --nnodes=1 --nproc_per_node=4 \
  main.py --config local_visela_cuhk_pedes.yaml
```

Checkpoints and logs are written under `model.saved_path`. Optional environment variables (the legacy `MMTBPR_*` names are still accepted):

| Variable | Effect |
| --- | --- |
| `VISELA_RUN_NAME` | save checkpoints under `<saved_path>/<run_name>` |
| `VISELA_SAVED_PATH` | override `model.saved_path` |
| `VISELA_SEED` | override `misc.seed` |
| `VISELA_EVAL_EPOCHS` | comma-separated epochs to evaluate during training |
| `VISELA_SAVE_EPOCH_CHECKPOINTS` | save a checkpoint at every evaluated epoch (`1`/`true`/`yes`) |

## Evaluation

Compute Rank-1/5/10 and mAP for a fine-tuned checkpoint without training:

```sh
python main.py \
  --config local_visela_cuhk_pedes.yaml \
  --eval-only \
  --resume /path/to/checkpoint.pth
```

Multilingual configs report per-language metrics and their average.

## Pretraining

Pretrain on MultiPedes starting from the CCLM-X2VLM checkpoint:

```sh
torchrun --nproc_per_node=4 Pretrain.py \
  --config config/examples/pretrain_multipedes.yaml \
  --checkpoint /path/to/cclm_x2vlm_base.th \
  --output_dir output/pretrain/visela \
  --epoch 2
```

`Pretrain.py` writes `model_state_epoch_<N>.th` into `--output_dir` (and `training_state_latest.th` for `--resume`). Point `model.resume` of the fine-tuning configs at the final `model_state_epoch_*.th` to reproduce the "w/ pretraining" rows.

## Attention Visualization

`eval.py` is not the metric evaluation entry point; it renders the refiner's query-to-token attention as per-language heatmaps for a fine-tuned checkpoint:

```sh
python eval.py --config local_visela_cuhk_pedes.yaml --resume /path/to/checkpoint.pth
```

Instead of `--resume`, `--exp-name <run_name>` loads `<saved_path>/<run_name>/checkpoint.pth` (the layout written by training with `VISELA_RUN_NAME`) and writes next to it. Figures are saved as PDF and PNG in a `text_attention_heatmaps/` folder inside the output directory (`model.saved_path`, or `--saved-path`); `--vis-num-queries`, `--vis-seed`, and `--vis-query-indices` control which text queries are drawn.

---

## Results

Results reported in the paper for ViSeLA with MultiPedes pretraining (percent).

### Multilingual TBPR (ViSeLA with MultiPedes pretraining, Table 2)

| Dataset | Avg R@1 | Avg mAP | en R@1 | ch R@1 | fr R@1 | de R@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUHK-PEDES (M) | 77.94 | 70.16 | 79.37 | 77.29 | 77.66 | 77.45 |
| ICFG-PEDES (M) | 69.66 | 43.45 | 71.35 | 69.26 | 68.80 | 69.23 |
| UFineBench (M) | 91.52 | 90.56 | 92.27 | 91.55 | 91.17 | 91.10 |

### English-only TBPR (ViSeLA with MultiPedes pretraining, Table 3)

| Dataset | R@1 | R@5 | R@10 | mAP |
| --- | ---: | ---: | ---: | ---: |
| CUHK-PEDES | 79.42 | 92.11 | 95.48 | 71.53 |
| ICFG-PEDES | 70.53 | 83.52 | 88.10 | 44.65 |
| UFineBench | 91.65 | 97.00 | 98.26 | 90.83 |

---

## Citation

```bibtex
@inproceedings{
anonymous2026visiongrounded,
title={Vision-Grounded Semantic Learning and Alignment for Scalable Multilingual Text-Based Person Retrieval},
author={Anonymous},
booktitle={34th ACM International Conference on Multimedia},
year={2026},
url={https://openreview.net/forum?id=mKcbH8tKps}
}
```

---

## Acknowledgements

This repository builds on [Bi-IRRA](https://github.com/Flame-Chasers/Bi-IRRA), [TBPS-CLIP](https://github.com/Flame-Chasers/TBPS-CLIP), [X2-VLM](https://github.com/zengyan-97/X2-VLM), and [BEiT-2](https://github.com/microsoft/unilm/tree/master/beit2). We thank the authors for their open-source contributions.

---

## License

This project is licensed under the MIT License; see `LICENSE`. Third-party components and their licenses are listed in `THIRD_PARTY_NOTICES.md`.
