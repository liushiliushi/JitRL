"""One-output-token remote scorer. Requires complete candidate logprob coverage."""

import json
import math
import os
import urllib.error
import urllib.request

from . import decision_prompt, validate_candidates


class OpenRouterDecisionScorer:
    def __init__(self, model_name, api_key=None):
        if not model_name:
            raise ValueError('--decision_model is required for OpenRouter.')
        self.model_name = model_name
        self._api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        if not self._api_key:
            raise ValueError('Set OPENROUTER_API_KEY before using the OpenRouter decision backend.')

    def score(self, state, history, actions, instruction):
        actions = validate_candidates(actions)
        # API top-logprob limits cannot expose an arbitrary-size full vocabulary.
        if len(actions) > 20:
            raise ValueError('OpenRouter exposes at most 20 top token logprobs. '
                             'Use the local backend for this full legal action set, '
                             'or explicitly provide a smaller valid subset with decision_candidate_provider.')
        labels = list('ABCDEFGHIJKLMNOPQRST')[:len(actions)]
        body = {'model': self.model_name, 'messages': [
            {'role': 'system', 'content': 'Select the best legal action. Output only one uppercase option letter.'},
            {'role': 'user', 'content': decision_prompt(state, history, actions, labels, instruction)}],
            'max_tokens': 1, 'temperature': 1, 'logprobs': True, 'top_logprobs': 20,
            'provider': {'require_parameters': True}}
        request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(body).encode(), headers={
                'Authorization': 'Bearer ' + self._api_key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            # Do not log request headers or an upstream error body containing credentials.
            raise RuntimeError(f'OpenRouter decision request failed (HTTP {error.code}). '
                               'Check model logprobs support, API credentials and credits.') from None
        if data.get('error'):
            raise RuntimeError('OpenRouter returned an error instead of decision logprobs.')
        choices = data.get('choices') or []
        positions = ((choices[0].get('logprobs') or {}).get('content') or []) if choices else []
        usage = data.get('usage') or {}
        if len(positions) != 1 or usage.get('completion_tokens', 1) != 1:
            raise RuntimeError('Expected exactly one output-token logprob position; use a non-reasoning model with logprobs support.')
        if (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0):
            raise RuntimeError('Decision response used reasoning tokens; choose a non-reasoning model.')
        records = positions[0].get('top_logprobs') or []
        scores = {}
        for item in records:
            token, value = item.get('token'), item.get('logprob')
            # Exact token labels: never silently combine whitespace variants or zero-fill omissions.
            if token in labels and isinstance(value, (int, float)) and math.isfinite(value) and value > -9999:
                scores[token] = value
        missing = [label for label in labels if label not in scores]
        if missing:
            raise RuntimeError('Incomplete candidate logprobs; missing labels: ' + ', '.join(missing)
                               + '. Use the local backend or another supporting model; no actions were dropped.')
        return labels, [scores[label] for label in labels]
