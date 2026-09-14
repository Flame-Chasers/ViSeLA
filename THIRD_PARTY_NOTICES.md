# Third-Party Notices

ViSeLA's original contributions are released under the MIT License (see [LICENSE](LICENSE)).
Parts of this repository are derived from, or adapted from, the projects listed below.
Existing copyright notices and license headers are retained in the affected source
files. The corresponding upstream licenses continue to apply to those portions;
copies of the license texts are distributed in [licenses/](licenses/).

## ByteDance X-VLM / X2-VLM (BSD 3-Clause)

- Upstream: https://github.com/zengyan-97/X-VLM and https://github.com/zengyan-97/X2-VLM
- Copyright (c) 2022-2023, ByteDance Inc. All rights reserved.
- License: BSD 3-Clause License. The upstream `LICENSE` files use the standard
  three-clause BSD text, including a non-endorsement clause that names
  `Salesforce.com` (inherited from the ALBEF code base).
- Full texts: [X-VLM (2022)](licenses/X-VLM-BSD-3-Clause.txt) and
  [X2-VLM (2023)](licenses/X2-VLM-BSD-3-Clause.txt), copied without changes from
  their upstream `LICENSE` files. The X2-VLM copy was checked against local revision
  `6981be21e63e3ffbd65875d9076b48c6022f1707`.
- Files derived from this code base:
  - `Pretrain.py`
  - `optim.py`
  - `scheduler.py`
  - `accelerators/accelerator.py`
  - `accelerators/torch_ddp_accelerator.py` (native-DDP replacement of the upstream accelerator)
  - `dataset/__init__.py`
  - `dataset/dist_dataset.py`
  - `dataset/pretrain_dataset.py`
  - `dataset/randaugment.py`
  - `dataset/utils.py`
  - `dataset/tokenizers/__init__.py`
  - `dataset/tokenizers/bert_tokenizer_with_dropout.py`
  - `misc/cclm_utils.py` (distributed/logging helpers from the X-VLM `utils` package)
  - `models/__init__.py`
  - `models/xvlm.py`
  - `models/model_pretrain.py` (pretraining model built on `XVLMBase`)
  - `utils/__init__.py`
  - `utils/checkpointer.py`
  - `utils/hdfs_io.py`
  - `utils/torch_io.py`

## Microsoft Swin Transformer (MIT)

- Upstream: https://github.com/microsoft/Swin-Transformer
- Copyright (c) 2021 Microsoft. Written by Ze Liu.
- License: MIT License.
- Full text: [Swin Transformer MIT license](licenses/Swin-Transformer-MIT.txt),
  copied from the upstream `LICENSE`, including the Microsoft copyright notice.
- Files: `models/swin_transformer.py`

## Microsoft BEiT-2 / unilm (MIT)

- Upstream: https://github.com/microsoft/unilm/tree/master/beit2
- Copyright (c) 2022 Microsoft. By Zhiliang Peng. Based on BEiT, timm, DeiT and DINO.
- License: MIT License.
- Full text: [unilm MIT license](licenses/unilm-MIT.txt), copied from the upstream
  `LICENSE`, including the Microsoft copyright notice.
- Files: `models/beit2.py`, `config/config_beit2_base.json`

## HuggingFace Transformers / Google AI Language Team / NVIDIA (Apache-2.0)

- Upstream: https://github.com/huggingface/transformers
- Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
  Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
- License: Apache License, Version 2.0.
- Full text: [Transformers Apache-2.0 license](licenses/Transformers-Apache-2.0.txt),
  copied from Transformers v4.40.0 and verified against the installed distribution.
  That upstream release contains no separate `NOTICE` file.
- Files: `models/xbert.py` (BERT), `models/xroberta.py` (RoBERTa / XLM-RoBERTa), each
  adapted through X-VLM/X2-VLM. Modification notices are included in their headers.

## OpenAI CLIP via HuggingFace Transformers (Apache-2.0)

- Upstream: https://github.com/openai/CLIP and https://github.com/huggingface/transformers
- Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
- License: Apache License, Version 2.0.
- Full text: [Transformers Apache-2.0 license](licenses/Transformers-Apache-2.0.txt).
  This is the license of the HuggingFace implementation adapted by X2-VLM; it is
  not a statement that OpenAI's separate CLIP repository is Apache-2.0 licensed.
- Files: `models/clip_vit.py`

## timm - PyTorch Image Models (Apache-2.0)

- Upstream: https://github.com/huggingface/pytorch-image-models
- Copyright 2019-2023 Ross Wightman.
- License: Apache License, Version 2.0.
- Full text of Apache-2.0 is included in
  [licenses/Transformers-Apache-2.0.txt](licenses/Transformers-Apache-2.0.txt).
- Usage: imported as a dependency (`timm.models.layers`, `timm.models.registry`) by
  `models/beit2.py`, `models/swin_transformer.py`, and `models/xvlm.py`.
  The BEiT-2 source header also credits timm among its upstream code bases.

## Bi-IRRA (MIT declared in README)

- Upstream: https://github.com/Flame-Chasers/Bi-IRRA
- License evidence: the upstream [README License section](https://github.com/Flame-Chasers/Bi-IRRA/blob/76c6c23cfa7d40ca0a28d6bcfd1c43c4c1e3344a/README.md#license)
  declares the MIT License. Both the local checkout at
  `e3b7a144597bb20f6893399af8521622a4d41ae5` and the linked upstream revision lack
  a standalone license file and a project-wide copyright notice.
- The [standard MIT permission and disclaimer text](licenses/MIT-permission-text.txt)
  is included for that declaration; it is not an upstream license file, and no
  missing copyright holder or year has been invented.
- Usage: the multilingual training/evaluation loop in `main.py`, the data pipeline in
  `misc/build.py`, `misc/caption_dataset.py`, `misc/data.py`, `misc/eval.py`,
  `misc/utils.py`, and the multilingual annotation format (`train_reid_{lang}.json`) follow
  the Bi-IRRA code base, which this project extends.

## TBPS-CLIP (MIT declared in README)

- Upstream: https://github.com/Flame-Chasers/TBPS-CLIP
- Usage: the config-driven experiment structure (`misc.utils.parse_config`, `experiment` /
  `schedule` config groups) originates from TBPS-CLIP via Bi-IRRA.
- License evidence: the upstream [README License section](https://github.com/Flame-Chasers/TBPS-CLIP/blob/ba090bc7dcb3b2787e80f701fc34e19f9205cee4/README.md#license)
  declares the MIT License. The linked revision has no standalone license file or
  project-wide copyright notice. The
  [standard MIT permission and disclaimer text](licenses/MIT-permission-text.txt)
  is included for this declaration as well.
- This attribution records TBPS-CLIP's own README declaration; it does not infer
  permission from Bi-IRRA's license. Upstream third-party portions retain their
  respective terms.

## Pretrained Weights and Datasets

Pretrained weights and datasets referenced in the README (`xlm-roberta-base`,
CCLM-X2VLM, BEiT-2, CUHK-PEDES, ICFG-PEDES, RSTPReid, UFineBench, and the `facebook/m2m100_418M`
translation model used by `tools/offline_code_switch.py`) are not redistributed here and are
subject to their own licenses and terms of use.
