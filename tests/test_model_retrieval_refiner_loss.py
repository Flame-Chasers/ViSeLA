from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.model_retrieval import ViSeLA


class DummyRefiner:
    def __init__(self, use_contrastive_loss=True):
        self.config = SimpleNamespace(
            use_contrastive_loss=use_contrastive_loss,
            lambda_feat_div=0.0,
            lambda_attn_div=0.0,
            diversity_mode='soft',
            feat_threshold=0.6,
            attn_threshold=0.01,
        )
        self.calls = []

    def forward(self, image_embeds, text_embeds, text_atts, idx, temp, return_aux=False):
        self.calls.append(return_aux)
        loss = torch.tensor(4.0)
        if return_aux:
            aux = {
                'image_attrs': torch.zeros(2, 4, 3),
                'text_attrs': torch.zeros(2, 4, 3),
                'image_attn': torch.zeros(2, 4, 3),
                'text_attn': torch.zeros(2, 4, 3),
            }
            return loss, aux
        return loss


def build_stub_model(refiner):
    model = ViSeLA.__new__(ViSeLA)
    torch.nn.Module.__init__(model)
    model.refiner = refiner
    model.use_mlm = False
    model.itm_weight = 4.0
    model.temp = torch.tensor(0.07)
    model.get_vision_embeds = lambda image: (torch.ones(2, 3, 4), None, None)
    model.get_text_embeds = lambda text_ids, text_atts, lang_ids=None: (torch.ones(2, 5, 4), None)
    model.get_features = lambda image_embeds, text_embeds: (torch.ones(2, 4), torch.ones(2, 4))
    model.get_contrastive_loss = lambda image_feat, text_feat, idx=None: torch.tensor(6.0)
    model.get_matching_loss = lambda *args, **kwargs: torch.tensor(2.0)
    model.train()
    return model


def run_forward(model):
    image = torch.ones(2, 3, 4)
    text_ids = torch.ones(2, 5, dtype=torch.long)
    text_atts = torch.ones(2, 5, dtype=torch.long)
    idx = torch.arange(2, dtype=torch.long)
    return model(image, text_ids, text_atts, idx=idx)


class RetrievalRefinerLossTest(unittest.TestCase):
    def test_retrieval_uses_refiner_loss_when_enabled(self):
        refiner = DummyRefiner(use_contrastive_loss=True)
        model = build_stub_model(refiner)

        ret = run_forward(model)

        self.assertTrue(torch.isclose(ret['loss_itc'], torch.tensor(5.0)))
        self.assertTrue(torch.isclose(ret['loss_itm'], torch.tensor(8.0)))
        self.assertEqual(refiner.calls, [True])

    def test_retrieval_skips_refiner_loss_when_disabled(self):
        refiner = DummyRefiner(use_contrastive_loss=False)
        model = build_stub_model(refiner)

        ret = run_forward(model)

        self.assertTrue(torch.isclose(ret['loss_itc'], torch.tensor(6.0)))
        self.assertTrue(torch.isclose(ret['loss_itm'], torch.tensor(8.0)))
        self.assertEqual(refiner.calls, [])

    def test_retrieval_runs_without_refiner(self):
        model = build_stub_model(refiner=None)

        ret = run_forward(model)

        self.assertTrue(torch.isclose(ret['loss_itc'], torch.tensor(6.0)))
        self.assertTrue(torch.isclose(ret['loss_itm'], torch.tensor(8.0)))


if __name__ == '__main__':
    unittest.main()
