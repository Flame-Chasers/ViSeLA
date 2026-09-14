import csv
import os

import torch
import torch.nn.functional as F

from typing import Union, List, Optional, Tuple
from misc.cclm_utils import MetricLogger, get_world_size, get_rank
import torch.distributed as dist
from misc.utils import is_master
import numpy as np
np.set_printoptions(
    threshold=np.inf,      # print every element
    linewidth=200,         # wider lines to avoid wrapping
    precision=3,           # more decimal places
    suppress=True          # no scientific notation for small values
)

@torch.no_grad()
def cclm_test(model, data_loader, tokenizer, config):
    device = config.device
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config.data.test_batch_size  # 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config.experiment.text_length,
                               return_tensors="pt").to(device)

        text_feat, _ = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_embed = model.get_features(text_embeds=text_feat)

        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    for image in data_loader:
        image = image.to(device)

        image_feat, _, _ = model.get_vision_embeds(image)
        image_embed = model.get_features(image_embeds=image_feat)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()

    num_tasks = get_world_size()
    rank = get_rank()

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)), -100.0).to(device)

    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 1000, header)):
        topk_sim, topk_idx = sims.topk(k=config.model.k_test, dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        output = model.get_cross_embeds(image_embeds=encoder_output, image_atts=encoder_att,
                                        text_embeds=text_feats[start + i].repeat(config.model.k_test, 1, 1),
                                        text_atts=text_atts[start + i].repeat(config.model.k_test, 1))
        score = model.itm_head(output[:, 0, :])[:, 1]

        score_matrix_t2i[start + i, topk_idx] = score
    dist.barrier()
    torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    score_matrix_t2i = score_matrix_t2i.cpu().numpy()
    return score_matrix_t2i


@torch.no_grad()
def metric_eval(scores_t2i, img2person, txt2person, config):
    device = config.device
    scores_t2i = torch.Tensor(scores_t2i).to(device)
    img2person = img2person.to(device)
    txt2person = txt2person.to(device)

    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def acc_k(matches, k=1):
        matches_k = matches[:, :k].sum(dim=-1)
        matches_k = torch.sum((matches_k > 0))
        return 100.0 * matches_k / matches.size(0)

    # Compute metrics
    ir1 = acc_k(matches, k=1).item()
    ir5 = acc_k(matches, k=5).item()
    ir10 = acc_k(matches, k=10).item()
    ir_mean = (ir1 + ir5 + ir10) / 3

    real_num = matches.sum(dim=-1)
    tmp_cmc = matches.cumsum(dim=-1).float()
    order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long).to(device)
    tmp_cmc /= order
    tmp_cmc *= matches
    AP = tmp_cmc.sum(dim=-1) / real_num
    mAP = AP.mean() * 100.0

    eval_result = {'r1': ir1,
                   'r5': ir5,
                   'r10': ir10,
                   'r_mean': ir_mean,
                   'mAP': mAP.item()
                   }

    return eval_result


def save_query_artifact(scores_t2i, img2person, txt2person, config, method: str, seed: int, language: str):
    device = config.device
    scores_t2i = torch.Tensor(scores_t2i).to(device)
    img2person = img2person.to(device)
    txt2person = txt2person.to(device)

    index = torch.argsort(scores_t2i, dim=-1, descending=True)
    pred_person = img2person[index]
    matches = (txt2person.view(-1, 1).eq(pred_person)).long()

    def hit_at(k: int):
        return (matches[:, :k].sum(dim=-1) > 0).long()

    real_num = matches.sum(dim=-1).clamp_min(1)
    tmp_cmc = matches.cumsum(dim=-1).float()
    order = torch.arange(start=1, end=matches.size(1) + 1, dtype=torch.long).to(device)
    tmp_cmc /= order
    tmp_cmc *= matches
    ap = tmp_cmc.sum(dim=-1) / real_num

    artifact_dir = os.path.join(config.model.saved_path, "query_artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    output_path = os.path.join(artifact_dir, f"{language}.csv")

    person_ids = txt2person.detach().cpu().tolist()
    hit1 = hit_at(1).detach().cpu().tolist()
    hit5 = hit_at(5).detach().cpu().tolist()
    hit10 = hit_at(10).detach().cpu().tolist()
    ap_values = ap.detach().cpu().tolist()

    fieldnames = ["method", "seed", "language", "query_index", "person_id", "hit@1", "hit@5", "hit@10", "AP"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for query_index, person_id in enumerate(person_ids):
            writer.writerow({
                "method": method,
                "seed": seed,
                "language": language,
                "query_index": query_index,
                "person_id": person_id,
                "hit@1": hit1[query_index],
                "hit@5": hit5[query_index],
                "hit@10": hit10[query_index],
                "AP": f"{ap_values[query_index]:.10f}",
            })

    return output_path

@torch.no_grad()
def get_refiner_attn(
    model,
    data_loader,
    tokenizer,
    config,
    *,
    text_index: int = 0
) -> Tuple[List[str], np.ndarray, Optional[int]]:
    """Return the refiner attention for one text query.

    Args:
        model: model instance
        data_loader: test DataLoader
        tokenizer: tokenizer
        config: config object
        text_index: index of the text query (default 0)

    Returns:
        tokens: list of tokens
        txt_attn: text attention weights [num_queries, seq_len]
        pid: person ID (may be None)

    Raises:
        ValueError: if dataset.text is empty
        IndexError: if text_index is out of range
    """
    device = config.device
    model.eval()

    texts = data_loader.dataset.text

    # Bounds checks
    if not texts:
        raise ValueError("dataset.text is empty; cannot visualize refiner attention")
    if text_index < 0 or text_index >= len(texts):
        raise IndexError(f"text_index out of range: {text_index} not in [0, {len(texts)})")

    text = texts[text_index]

    text_input = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=config.experiment.text_length,
        return_tensors="pt"
    ).to(device)

    text_feat, _ = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
    tokens = tokenizer.convert_ids_to_tokens(text_input.input_ids[0])
    txt_attn = model.refiner.get_attn(text_feat)
    txt_attn = txt_attn.cpu().numpy()[0]

    # Person ID, if available
    pid: Optional[int] = None
    if hasattr(data_loader.dataset, "txt2person"):
        try:
            pid = int(data_loader.dataset.txt2person[text_index])
        except Exception:
            pid = None

    return tokens, txt_attn, pid
