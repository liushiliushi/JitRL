"""Small paired Jericho evaluation of actual legacy and remote decision paths.

Run from repo root with game/API dependencies installed. API key is read from
environment or a hidden prompt, never saved in benchmark outputs.
"""
import argparse
import contextlib
import getpass
import io
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Jericho'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+', default=['zork1', 'library'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--model', default='openai/gpt-4o-mini')
    parser.add_argument('--prompt-key', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    from src.env import JerichoEnv
    from src.jitrl_agent import JitRLAgent
    from openai.resources.chat.completions import Completions
    key = getpass.getpass('OpenRouter key (hidden): ') if args.prompt_key else os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise SystemExit('OPENROUTER_API_KEY is required.')
    # Legacy helpers expect this name even though their endpoint is OpenRouter.
    os.environ['OPENROUTER_API_KEY'] = key
    os.environ['OPENAI_API_KEY'] = key
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    sdk_create, urlopen = Completions.create, urllib.request.urlopen
    results_file = output / 'results.json'
    results = json.loads(results_file.read_text()) if args.resume and results_file.exists() else []
    for game in args.games:
        for seed in args.seeds:
            for mode in ['single_forward', 'legacy']:
                if any(r['game'] == game and r['seed'] == seed and r['mode'] == mode
                       and r['model'] == args.model and r['step_limit'] == args.steps for r in results):
                    continue
                usage = []
                def sdk_meter(self, *a, **kw):
                    # Bound API retries so a failed baseline does not silently run forever.
                    self._client.max_retries = 0
                    kw['timeout'] = 30
                    try:
                        result = sdk_create(self, *a, **kw)
                    except Exception as error:
                        raise RuntimeError('Legacy API request failed: ' + type(error).__name__) from None
                    if result.usage:
                        usage.append(result.usage.model_dump())
                    return result

                def url_meter(*a, **kw):
                    kw['timeout'] = 30
                    with urlopen(*a, **kw) as response:
                        content = response.read()
                    data = json.loads(content)
                    if data.get('usage'):
                        usage.append(data['usage'])
                    return io.BytesIO(content)

                config = SimpleNamespace(decision_mode=mode, decision_backend='openrouter',
                    decision_model=args.model, llm_model=args.model, decision_beta=1.0,
                    enable_cross_mem=False, update_guiding_prompt=False, use_history_prompt=False,
                    use_valid_actions=True, confidence_mode='verbalized', top_actions=3,
                    llm_temperature=0.0, max_memory=30,
                    retrieval_top_k=10, retrieval_threshold=0.95)
                random.seed(seed)
                steps, failure, score, done = [], None, 0, False
                run_name = f'{game}-{seed}-{mode}'
                print(f'START {run_name}', flush=True)
                start = time.monotonic()
                env = None
                with (output / f'{run_name}.log').open('w') as log, contextlib.redirect_stdout(log), \
                     patch.object(Completions, 'create', sdk_meter), patch('urllib.request.urlopen', url_meter):
                    try:
                        env = JerichoEnv(str(ROOT / 'Jericho/jericho-games' / f'{game}.z5'), seed, args.steps)
                        env.use_parallel = False
                        env.env.seed(seed)  # Repo wrapper stores seed but does not apply it.
                        observation, info = env.reset()
                        agent = JitRLAgent(config)
                        agent.start_episode()
                        for index in range(args.steps):
                            candidates = list(info['valid'])
                            step_start = time.monotonic()
                            try:
                                with patch('src.jitrl_agent.time.time_ns', return_value=seed * 1000 + index):
                                    action, raw = agent.generate_action(SimpleNamespace(state=observation), info=info)
                            except Exception as error:
                                failure = str(error).replace(key, '[REDACTED]')
                                steps.append({'step': index, 'state': observation, 'valid': candidates,
                                    'error': failure, 'decision_seconds': time.monotonic()-step_start})
                                break
                            seconds = time.monotonic()-step_start
                            before = observation
                            observation, reward, done, info = env.step(action)
                            score = info['score']
                            agent.update_game_history_reward(reward, score)
                            steps.append({'step': index, 'state': before, 'valid': candidates,
                                'action': action, 'allowed': action in candidates, 'reward': reward,
                                'score': score, 'decision_seconds': seconds, 'response': raw})
                            print(f'STEP {index} score={score} action={action}', file=sys.__stdout__, flush=True)
                            if done:
                                break
                    except Exception as error:
                        failure = str(error).replace(key, '[REDACTED]')
                    finally:
                        if env:
                            env.close()
                result = {'game': game, 'seed': seed, 'mode': mode, 'model': args.model,
                    'step_limit': args.steps, 'cross_episode_memory': False, 'score': score,
                    'completed_steps': sum('action' in s for s in steps), 'failure': failure,
                    'elapsed_seconds': time.monotonic()-start, 'api_calls': len(usage),
                    'prompt_tokens': sum(u.get('prompt_tokens', 0) for u in usage),
                    'completion_tokens': sum(u.get('completion_tokens', 0) for u in usage),
                    'reported_cost_usd': sum(u.get('cost') or 0 for u in usage),
                    'cost_entries': sum(u.get('cost') is not None for u in usage),
                    'decision_seconds': sum(s['decision_seconds'] for s in steps), 'steps': steps}
                results.append(result)
                (output / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
                print(json.dumps({k: v for k, v in result.items() if k != 'steps'}), flush=True)


if __name__ == '__main__':
    main()
