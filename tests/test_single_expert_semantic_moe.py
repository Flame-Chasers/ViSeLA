from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

# Support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from models.xbert import BertIntermediate, BertLayer, BertOutput


class SingleExpertSemanticMoETest(unittest.TestCase):
    def test_single_expert_ignores_extra_refiner_queries(self):
        config = SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            layer_norm_eps=1e-12,
        )
        layer = BertLayer.__new__(BertLayer)
        nn.Module.__init__(layer)
        layer.num_experts = 1
        layer.routing_temp = 1.0
        layer.intermediate = nn.ModuleList([BertIntermediate(config)])
        layer.output = nn.ModuleList([BertOutput(config, drop_path_rate=0.0)])

        hidden = torch.randn(2, 5, config.hidden_size)
        attn = torch.randn(2, 4, 5)

        output = layer.semantic_moe_forward(hidden, attn)
        expected = layer.output[0](
            layer.intermediate[0](hidden.reshape(-1, config.hidden_size)),
            hidden.reshape(-1, config.hidden_size),
        ).reshape_as(hidden)

        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
