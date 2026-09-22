"""Optional decision backend. Legacy agents do not import model dependencies."""

import importlib
import json
import math


def add_decision_arguments(parser):
    parser.add_argument('--decision_mode', choices=['legacy', 'single_forward'], default='legacy')
    parser.add_argument('--decision_backend', choices=['local', 'openrouter'], default='local')
    parser.add_argument('--decision_model', help='Hugging Face model/path or OpenRouter model ID; required in single_forward mode.')
    parser.add_argument('--decision_candidate_provider', help='Optional module:function accepting state, history, info, url; returns action strings. Required for WebArena.')
    parser.add_argument('--decision_beta', type=float, default=1.0, help='Positive KL temperature for memory advantage adjustment.')


def validate_candidates(actions):
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError('single_forward requires a nonempty list of candidate action strings.')
    if any(not isinstance(a, str) or not a.strip() for a in actions):
        raise ValueError('Every candidate must be a nonempty action string.')
    if len(set(actions)) != len(actions):
        raise ValueError('Candidate actions must be unique.')
    return list(actions)


def softmax(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('Decision scores must be finite and nonempty.')
    maximum = max(values)
    weights = [math.exp(v - maximum) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def decision_prompt(state, history, actions, labels, instruction):
    return (f'Goal: {instruction}\nState: {state}\nRecent history: '
            + json.dumps(history, ensure_ascii=False, default=str)
            + '\nLegal candidate actions (choose only from this list):\n'
            + '\n'.join(f'{label}: {action}' for label, action in zip(labels, actions))
            + '\nReply with exactly one option label, without explanation or whitespace.')


class LocalDecisionScorer:
    """One model forward over the prompt; no generate(), sampling or reasoning trace."""

    def __init__(self, model_name):
        if not model_name:
            raise ValueError('--decision_model is required for single_forward.')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype='auto', device_map='auto').eval()
        # Labels are mapped to actual single vocabulary tokens, never multi-token IDs.
        self.labels = []
        seen = set()
        for label in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + [str(i) for i in range(1000)]:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in seen and ids[0] not in self.tokenizer.all_special_ids:
                self.labels.append((label, ids[0]))
                seen.add(ids[0])

    def score(self, state, history, actions, instruction):
        actions = validate_candidates(actions)
        if len(actions) > len(self.labels):
            raise ValueError(f'{len(actions)} candidates exceed {len(self.labels)} available single-token labels; provide a smaller valid candidate set.')
        labels = self.labels[:len(actions)]
        prompt = decision_prompt(state, history, actions, [label for label, _ in labels], instruction)
        encoded = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(encoded, return_tensors='pt', add_special_tokens=False)
        limit = getattr(self.model.config, 'max_position_embeddings', None)
        if isinstance(limit, int) and inputs['input_ids'].shape[-1] > limit:
            raise ValueError('Decision prompt exceeds model context; reduce history/candidates.')
        device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            # Full vocabulary at ONE decision position, then candidate-only softmax.
            logits = self.model(**inputs, use_cache=False).logits[0, -1]
            selected = logits[[token for _, token in labels]].float()
            log_probs = self.torch.log_softmax(selected, dim=0).cpu().tolist()
        return [label for label, _ in labels], log_probs


def decide(agent, state_node, info=None, url=None, web=False, normalize_action=None):
    beta = getattr(agent.args, 'decision_beta', 1.0)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('--decision_beta must be finite and positive.')
    history = [{k: step.get(k) for k in ('state', 'action', 'reward')}
               for step in agent.game_history[-20:]]
    provider = getattr(agent.args, 'decision_candidate_provider', None)
    instruction = getattr(state_node, 'instruction', None) or agent.guiding_prompt
    if provider:
        module, name = provider.rsplit(':', 1)
        actions = getattr(importlib.import_module(module), name)(
            state=state_node.state, history=history, info=info, url=url, instruction=instruction)
    else:
        actions = (info or {}).get('valid')
    actions = validate_candidates(actions)
    if not web and info and info.get('valid'):
        if any(a not in info['valid'] for a in actions):
            raise ValueError('Candidate provider returned an action outside the environment valid set.')
    if not hasattr(agent, '_decision_scorer'):
        if getattr(agent.args, 'decision_backend', 'local') == 'openrouter':
            from .openrouter import OpenRouterDecisionScorer
            agent._decision_scorer = OpenRouterDecisionScorer(getattr(agent.args, 'decision_model', None))
        else:
            agent._decision_scorer = LocalDecisionScorer(getattr(agent.args, 'decision_model', None))
    labels, log_probs = agent._decision_scorer.score(state_node.state, history, actions, instruction)
    if len(log_probs) != len(actions) or len(labels) != len(actions):
        raise ValueError('Scorer must return one label/log probability per candidate.')
    base_probs = softmax(log_probs)
    options = {i: {'action': action, 'token': labels[i-1], 'logprob': log_probs[i-1],
                   'normalized_prob': base_probs[i-1]} for i, action in enumerate(actions, 1)}
    if web:
        for option in options.values():
            option['normalized_action'] = normalize_action(option['action'], state_node.state).get('normalized_action', option['action'])
    memory_text = json.dumps(history, ensure_ascii=False, default=str)
    kwargs = {'current_url': url} if web else {'info': info}
    updated = agent.update_scores(state_node, options,
        k=getattr(agent.args, 'retrieval_top_k', 10),
        r=getattr(agent.args, 'retrieval_threshold', 0.8), memory_text=memory_text, **kwargs)
    # Reuse the existing advantage estimator, not legacy probability-plus-advantage scores.
    # Closed set: historical actions added by update_scores cannot enter the policy.
    advantages = [updated.get(i, {}).get('normalized_advantage', 0.0) for i in range(1, len(actions)+1)]
    final_probs = softmax([lp + advantage / beta for lp, advantage in zip(log_probs, advantages)])
    choice = max(range(len(actions)), key=final_probs.__getitem__)
    payload = {'decision_mode': 'single_forward', 'best_action': choice+1,
               'labels': labels, 'base_probabilities': base_probs,
               'policy_probabilities': final_probs, 'advantages': advantages}
    payload.update({f'option{i}': a for i, a in enumerate(actions, 1)})
    response = json.dumps(payload, ensure_ascii=False)
    if web:
        agent._add_to_game_history(state_node.state, actions[choice], response,
                                  task_goal=instruction, url=url)
    else:
        agent._add_to_game_history(state_node.state, actions[choice], response)
    return actions[choice], response
