"""Run with the optional local dependencies installed. No checkpoint/API needed."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:
    torch = None

from jitrl_decision import LocalDecisionScorer


@unittest.skipIf(torch is None, 'Optional local model dependencies not installed')
class LocalForwardTests(unittest.TestCase):
    def test_exactly_one_forward_and_candidate_only_softmax(self):
        torch.manual_seed(7)
        scorer = LocalDecisionScorer.__new__(LocalDecisionScorer)
        scorer.torch = torch
        scorer.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)).eval()
        scorer.labels = [('A', 4), ('B', 7)]
        scorer.tokenizer = Mock()
        scorer.tokenizer.apply_chat_template.return_value = 'prompt'
        inputs = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.ones(1, 3, dtype=torch.long)}
        scorer.tokenizer.return_value = inputs
        calls = []
        handle = scorer.model.register_forward_hook(lambda *args: calls.append(args[-1]))
        labels, logp = scorer.score('room', [], ['north', 'south'], 'Win')
        handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertEqual(labels, ['A', 'B'])
        expected = torch.log_softmax(calls[0].logits[0, -1, [4, 7]].float(), dim=0)
        torch.testing.assert_close(torch.tensor(logp), expected)
        self.assertAlmostEqual(torch.tensor(logp).exp().sum().item(), 1, places=6)
        self.assertFalse(scorer.tokenizer.call_args.kwargs['add_special_tokens'])
        self.assertFalse(scorer.tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])

    def test_no_truncation_of_candidate_list(self):
        scorer = LocalDecisionScorer.__new__(LocalDecisionScorer)
        scorer.labels = [('A', 4)]
        with self.assertRaisesRegex(ValueError, 'exceed'):
            scorer.score('room', [], ['north', 'south'], 'Win')


if __name__ == '__main__':
    unittest.main()
