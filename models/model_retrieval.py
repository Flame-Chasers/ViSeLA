import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.xvlm import XVLMBase, XVLMPlusBase
from models.xvlm import QuickGELU, LayerNorm
from models.queries import query_feature_diversity_loss, attention_diversity_loss

class ViSeLA(XVLMBase):
    def __init__(self, config, num_classes=None):
        # Initialize the base XVLM model for parallel data
        super().__init__(config, num_classes=num_classes, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=config.mlm.is_mlm,
                         use_bbox_loss=False)
        self.itm_weight = config.itm.weight
        self.mim_weight = config.mim.weight
        self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []

    def _should_compute_refiner_contrastive_loss(self):
        return self.refiner is not None and getattr(self.refiner.config, 'use_contrastive_loss', True)

    def _should_collect_refiner_aux(self):
        if self.refiner is None:
            return False
        return any(
            float(getattr(self.refiner.config, key, 0.0) or 0.0) > 0
            for key in ('lambda_feat_div', 'lambda_attn_div')
        )

    def forward(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None,
                masked_ids=None, idx=None, lang_ids=None):
        ret = dict()
        # Get image and source language text embeddings
        image_embeds, image_atts, image_refine = self.get_vision_embeds(image)
        text_embeds, text_refine = self.get_text_embeds(text_ids, text_atts, lang_ids=lang_ids)
        image_feat, text_feat = self.get_features(image_embeds, text_embeds)

        loss_refine = None
        aux = None
        if self._should_compute_refiner_contrastive_loss():
            if self.training:
                loss_refine, aux = self.refiner.forward(
                    image_embeds, text_embeds, text_atts, idx, self.temp, return_aux=True
                )
            else:
                loss_refine = self.refiner.forward(
                    image_embeds, text_embeds, text_atts, idx, self.temp, return_aux=False
                )
        elif self.training and self._should_collect_refiner_aux():
            _, aux = self.refiner.forward(
                image_embeds, text_embeds, text_atts, idx, self.temp, return_aux=True
            )

        # Calculate contrastive and matching losses for source language
        loss_itc_sl = self.get_contrastive_loss(image_feat, text_feat, idx=idx)
        loss_itm_sl = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx)

        # If using masked language modeling, calculate MLM loss for both languages
        if self.use_mlm:
            loss_mlm_sl = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)
            ret['loss_mlm'] = loss_mlm_sl
        if loss_refine is None:
            ret['loss_itc'] = loss_itc_sl
        else:
            ret['loss_itc'] = (loss_itc_sl + loss_refine) / 2

        # Add diversity losses
        if aux is not None:
            loss_aux = 0.0

            # Diversity-constraint settings
            diversity_mode = getattr(self.refiner.config, 'diversity_mode', 'hard')
            # Separate thresholds for feature / attention diversity (falls back to the legacy shared key)
            feat_threshold = getattr(self.refiner.config, 'feat_threshold',
                                    getattr(self.refiner.config, 'diversity_threshold', 0.5))
            attn_threshold = getattr(self.refiner.config, 'attn_threshold',
                                    getattr(self.refiner.config, 'diversity_threshold', 0.5))

            if self.refiner.config.lambda_feat_div > 0:
                # Feature diversity loss (hard/soft mode, dedicated threshold)
                loss_feat_div = (query_feature_diversity_loss(
                                    aux['image_attrs'],
                                    mode=diversity_mode,
                                    threshold=feat_threshold) +
                                 query_feature_diversity_loss(
                                    aux['text_attrs'],
                                    mode=diversity_mode,
                                    threshold=feat_threshold)) / 2
                loss_aux = loss_aux + self.refiner.config.lambda_feat_div * loss_feat_div

            if self.refiner.config.lambda_attn_div > 0:
                # Attention diversity loss (padding-aware, hard/soft mode, dedicated threshold)
                loss_attn_div = (attention_diversity_loss(
                                    aux['image_attn'],
                                    attention_mask=None,
                                    mode=diversity_mode,
                                    threshold=attn_threshold) +
                                 attention_diversity_loss(
                                    aux['text_attn'],
                                    attention_mask=text_atts,
                                    mode=diversity_mode,
                                    threshold=attn_threshold)) / 2
                loss_aux = loss_aux + self.refiner.config.lambda_attn_div * loss_attn_div

            # Tensor-safe zero check. main.py sums every key containing "loss",
            # so record loss_aux exactly once to avoid double counting.
            if isinstance(loss_aux, torch.Tensor) and loss_aux.numel() == 1:
                if loss_aux.item() != 0:
                    ret['loss_aux'] = loss_aux
            elif loss_aux != 0:
                ret['loss_aux'] = loss_aux

        ret['loss_itm'] = loss_itm_sl * self.itm_weight
        return ret


Bi_IRRA = ViSeLA  # backward-compatible alias
