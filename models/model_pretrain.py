# Pretrain Model with Refiner Support
# Based on X2-VLM architecture, adapted for ViSeLA with Refiner module

import torch
from einops import rearrange
from models.xvlm import XVLMBase, load_pretrained


class XVLMPretrain(XVLMBase):
    """
    Pretrain model supporting ITC, ITM, MLM losses with optional Refiner loss.
    Inherits from XVLMBase which already includes the Refiner module.
    """

    def __init__(self, config, load_vision_params=True, load_text_params=True, pretraining=True):
        super().__init__(config, load_vision_params=load_vision_params, load_text_params=load_text_params,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=True, use_bbox_loss=True,
                         config_text=None, pretraining=pretraining)

    def forward_multimodal(self, image, text_ids, text_atts, text_ids_masked=None, masked_pos=None, masked_ids=None,
                           image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None,
                           ret_bbox_loss=False, ret_match_loss=True, idx=None):
        """Forward for image-text multimodal pretraining"""

        if ret_bbox_loss:
            image_embeds, image_atts, image_embeds_fullatts = \
                self.get_vision_embeds(image, image_atts=image_atts, idx_to_group_img=idx_to_group_img)
        else:
            image_embeds, image_atts, _ = self.get_vision_embeds(image)

        text_embeds, _ = self.get_text_embeds(text_ids, text_atts)

        image_feat, text_feat = self.get_features(image_embeds, text_embeds)

        loss_itc = self.get_contrastive_loss(image_feat, text_feat, idx=idx)

        if ret_match_loss:
            loss_itm = self.get_matching_loss(image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=idx)
        else:
            # Use same device as loss_itc to avoid device mismatch
            loss_itm = loss_itc.new_zeros(())

        loss_mlm = self.get_mlm_loss(text_ids_masked, text_atts, image_embeds, image_atts, masked_pos, masked_ids)

        loss = {'loss_itc': loss_itc, 'loss_itm': loss_itm, 'loss_mlm': loss_mlm}

        # Add Refiner loss (only contrastive, no diversity)
        # Can be disabled via refiner.use_contrastive_loss (ablation: routing only, no contrastive loss)
        if hasattr(self, 'refiner') and self.refiner is not None:
            if getattr(self.refiner.config, 'use_contrastive_loss', True):
                loss_refine = self.refiner.forward(image_embeds, text_embeds, text_atts, idx, self.temp, return_aux=False)
                loss['loss_refine'] = loss_refine

        if ret_bbox_loss:
            output_coord = self.predict_bbox(image_embeds_fullatts, text_embeds, text_atts)
            loss_bbox, loss_giou = self.get_bbox_loss(output_coord, target_bbox, is_image=is_image)
            loss['loss_bbox'] = loss_bbox
            loss['loss_giou'] = loss_giou

        return loss

    def forward_text(self, text_ids=None, text_atts=None,
                     text_ids_masked=None, masked_pos=None, masked_ids=None):
        """Forward for text-only pretraining (MLM only)"""
        loss = self.get_mlm_loss(text_ids_masked, text_atts, None, None, masked_pos, masked_ids)
        return {'loss_mlm': loss}

    def forward(self, image=None, text_ids=None, text_atts=None,
                text_ids_masked=None, masked_pos=None, masked_ids=None,
                image_atts=None, idx_to_group_img=None, target_bbox=None, is_image=None,
                ret_bbox_loss=False, ret_match_loss=True, idx=None):

        if image is None:  # text only
            loss = self.forward_text(text_ids, text_atts, text_ids_masked,
                                     masked_pos, masked_ids)
        else:
            loss = self.forward_multimodal(image, text_ids, text_atts, text_ids_masked, masked_pos, masked_ids,
                                           image_atts, idx_to_group_img, target_bbox, is_image, ret_bbox_loss,
                                           ret_match_loss=ret_match_loss, idx=idx)

        return loss

    def load_pretrained(self, ckpt_rpath, config, is_domain_pretrain=False, text_ckpt_rpath=''):
        """Load pretrained weights, compatible with X2-VLM checkpoint format"""
        print('load checkpoint from %s' % ckpt_rpath)
        state_dict = load_pretrained(self, ckpt_rpath, config, is_eval=False, load_text=True)

        # Handle cross_encoder weights remapping for 18-layer text encoder
        temp_dict = {}
        for i in range(6):
            for key, value in state_dict.items():
                if key.startswith(f"cross_encoder.encoder.layer.{i}."):
                    new_key = key.replace(f"cross_encoder.encoder.layer.{i}", f"text_encoder.bert.encoder.layer.{i + 12}")
                    temp_dict[new_key] = value
        keys_to_remove = [key for key in state_dict.keys() if key.startswith("cross_encoder.encoder.layer.")]
        for key in keys_to_remove:
            del state_dict[key]

        # Handle text_encoder prefix for MLM
        keys_to_remove = []
        for key, value in state_dict.items():
            if key.startswith(f"text_encoder."):
                new_key = key.replace(f"text_encoder", f"text_encoder.bert")
                temp_dict[new_key] = value
                keys_to_remove.append(key)
            elif key.startswith(f"mlm_head."):
                new_key = key.replace(f"mlm_head", f"text_encoder.mlm_head")
                temp_dict[new_key] = value
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del state_dict[key]

        state_dict.update(temp_dict)

        # Replace text encoder if specified
        if text_ckpt_rpath:
            print(f"### Replacing text encoder from {text_ckpt_rpath}")
            text_state_dict = torch.load(text_ckpt_rpath, map_location='cpu')
            for k in list(text_state_dict.keys()):
                if k.startswith('roberta.'):
                    new_k = 'text_encoder.bert.' + k[8:]
                    text_state_dict[new_k] = text_state_dict[k]
                    del text_state_dict[k]
            state_dict.update(text_state_dict)

        msg = self.load_state_dict(state_dict, strict=False)
        print("unexpected_keys: ", msg.unexpected_keys)
        missing_keys = [p for p in msg.missing_keys]
        self.update_init_params(missing_keys)
        print("train from scratch: ", sorted(self.init_params))
