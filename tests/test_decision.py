import argparse
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jitrl_decision import add_decision_arguments, decide, softmax, validate_candidates


class DecisionTests(unittest.TestCase):
    def agent(self, advantages=None):
        agent = SimpleNamespace(
            args=SimpleNamespace(decision_beta=1.0), game_history=[], guiding_prompt='Win',
            _decision_scorer=SimpleNamespace(score=Mock(return_value=(['A', 'B'], [math.log(.8), math.log(.2)]))),
            update_scores=Mock(return_value=advantages or {}), _add_to_game_history=Mock())
        return agent

    def test_base_distribution_and_one_score_call(self):
        agent = self.agent()
        action, response = decide(agent, SimpleNamespace(state='room'), info={'valid': ['north', 'south']})
        self.assertEqual(action, 'north')
        agent._decision_scorer.score.assert_called_once()
        self.assertEqual(json.loads(response)['policy_probabilities'], [.8, .2])
        agent._add_to_game_history.assert_called_once()

    def test_memory_is_applied_in_log_space(self):
        agent = self.agent({1: {'normalized_advantage': -1}, 2: {'normalized_advantage': 1},
                            3: {'normalized_advantage': 100, 'action': 'invalid'}})
        action, response = decide(agent, SimpleNamespace(state='room'), info={'valid': ['north', 'south']})
        result = json.loads(response)
        self.assertEqual(action, 'south')
        expected = .2 * math.exp(1) / (.8 * math.exp(-1) + .2 * math.exp(1))
        self.assertAlmostEqual(result['policy_probabilities'][1], expected)
        self.assertAlmostEqual(sum(result['policy_probabilities']), 1)
        self.assertEqual(len(result['policy_probabilities']), 2)

    def test_invalid_candidates_fail_before_scoring(self):
        for actions in [None, [], [''], ['a', 'a'], ['a', 2], 'north']:
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                validate_candidates(actions)

    def test_invalid_beta(self):
        for beta in [0, -1, float('nan'), float('inf')]:
            agent = self.agent()
            agent.args.decision_beta = beta
            with self.assertRaises(ValueError):
                decide(agent, SimpleNamespace(state='room'), info={'valid': ['north', 'south']})
            agent._decision_scorer.score.assert_not_called()

    def test_web_provider_and_normalization(self):
        agent = self.agent()
        agent.args.decision_candidate_provider = 'example:choices'
        provider = Mock(return_value=['click("1")', 'click("2")'])
        with patch('jitrl_decision.importlib.import_module', return_value=SimpleNamespace(choices=provider)):
            action, _ = decide(agent, SimpleNamespace(state='page', instruction='Buy'),
                url='https://example.com', web=True,
                normalize_action=lambda a, s: {'normalized_action': 'normalized:' + a})
        self.assertEqual(action, 'click("1")')
        self.assertEqual(agent.update_scores.call_args.args[1][1]['normalized_action'], 'normalized:click("1")')
        self.assertEqual(provider.call_args.kwargs['url'], 'https://example.com')

    def test_jericho_provider_cannot_escape_valid_actions(self):
        agent = self.agent()
        agent.args.decision_candidate_provider = 'example:choices'
        with patch('jitrl_decision.importlib.import_module', return_value=SimpleNamespace(choices=lambda **kw: ['invalid'])):
            with self.assertRaises(ValueError):
                decide(agent, SimpleNamespace(state='room'), info={'valid': ['north']})

    def test_legacy_default(self):
        parser = argparse.ArgumentParser()
        add_decision_arguments(parser)
        self.assertEqual(parser.parse_args([]).decision_mode, 'legacy')

    def test_numerical_stability(self):
        self.assertEqual(softmax([10000, 10000]), [.5, .5])
        with self.assertRaises(ValueError):
            softmax([float('nan')])

    def test_real_agent_dispatch_skips_generation(self):
        # Compile the actual methods without importing game/API dependencies.
        root = Path(__file__).resolve().parents[1]
        for path, class_name, web in [
            ('Jericho/src/jitrl_agent.py', 'JitRLAgent', False),
            ('WebArena/memory_agents/jitrl_agent.py', 'BrowserGymJitRLAgent', True),
        ]:
            tree = ast.parse((root / path).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_action')
            # Exercise legacy branch entry without any model calls.
            namespace = {}
            exec(compile(ast.Module(body=[method], type_ignores=[]), path, 'exec'), namespace)
            agent = self.agent()
            agent.args.decision_mode = 'legacy'
            agent.get_prompts = Mock(side_effect=RuntimeError('legacy-entry'))
            with self.assertRaisesRegex(RuntimeError, 'legacy-entry'):
                namespace['generate_action'](agent, SimpleNamespace(state='room'))
            if not web:
                agent.args.decision_mode = 'single_forward'
                with patch('jitrl_decision.decide', return_value=('north', '{}')) as scorer:
                    result = namespace['generate_action'](agent, SimpleNamespace(state='room'), info={'valid': ['north']})
                self.assertEqual(result[0], 'north')
                scorer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
