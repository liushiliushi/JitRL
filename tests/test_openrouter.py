import ast
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jitrl_decision.openrouter import OpenRouterDecisionScorer


def response(labels='ABCD', reasoning=0):
    return io.BytesIO(json.dumps({'choices': [{'logprobs': {'content': [{
        'top_logprobs': [{'token': label, 'logprob': -i-1.0} for i, label in enumerate(labels)]
    }]}}], 'usage': {'completion_tokens': 1, 'completion_tokens_details': {'reasoning_tokens': reasoning}}}).encode())


class OpenRouterTests(unittest.TestCase):
    def test_jericho_legal_actions_reach_api_every_step(self):
        # Exercise the real agent entry point -> backend factory -> HTTP payload.
        path = Path(__file__).resolve().parents[1] / 'Jericho/src/jitrl_agent.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'JitRLAgent')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_action')
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        agent = SimpleNamespace(args=SimpleNamespace(decision_mode='single_forward', decision_backend='openrouter',
            decision_model='openai/gpt-4o-mini'), guiding_prompt='Explore safely', game_history=[],
            update_scores=Mock(return_value={}), _add_to_game_history=Mock())
        lists = [['open mailbox', 'north', 'south', 'look'], ['read leaflet', 'drop leaflet', 'east', 'west']]
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test-only'}), \
             patch('urllib.request.urlopen', side_effect=[response(), response()]) as send:
            for actions in lists:
                action, trace = namespace['generate_action'](agent, SimpleNamespace(state='West of house'), info={'valid': actions})
                self.assertIn(action, actions)
                self.assertAlmostEqual(sum(json.loads(trace)['base_probabilities']), 1)
        self.assertEqual(send.call_count, 2)
        for call, actions in zip(send.call_args_list, lists):
            payload = json.loads(call.args[0].data)
            self.assertEqual(payload['max_tokens'], 1)
            self.assertTrue(payload['logprobs'])
            self.assertEqual(payload['top_logprobs'], 20)
            prompt = payload['messages'][1]['content']
            self.assertIn('West of house', prompt)
            for label, action in zip('ABCD', actions):
                self.assertIn(f'{label}: {action}', prompt)
            self.assertNotIn('test-only', call.args[0].data.decode())

    def test_missing_candidates_rejected(self):
        scorer = OpenRouterDecisionScorer('openai/gpt-4o-mini', api_key='test-only')
        with patch('urllib.request.urlopen', return_value=response('ABC')), self.assertRaisesRegex(RuntimeError, 'missing labels: D'):
            scorer.score('state', [], ['a', 'b', 'c', 'd'], 'goal')

    def test_large_action_set_not_truncated(self):
        scorer = OpenRouterDecisionScorer('model', api_key='test-only')
        with patch('urllib.request.urlopen') as send, self.assertRaises(ValueError):
            scorer.score('state', [], [str(i) for i in range(21)], 'goal')
        send.assert_not_called()

    def test_reasoning_rejected(self):
        scorer = OpenRouterDecisionScorer('model', api_key='test-only')
        with patch('urllib.request.urlopen', return_value=response(reasoning=1)), self.assertRaisesRegex(RuntimeError, 'reasoning'):
            scorer.score('state', [], ['a', 'b'], 'goal')

    def test_missing_key(self):
        with patch.dict('os.environ', {}, clear=True), self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY'):
            OpenRouterDecisionScorer('model')


if __name__ == '__main__':
    unittest.main()
