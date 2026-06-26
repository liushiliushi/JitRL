"""
Exploration Method — Module 2

Code-based strategy path selection.
Supports: ucb, thompson, epsilon_greedy.
Selected via --exploration_method CLI arg.
"""

import math
import random
from typing import List

from .strategy_space import MilestoneNode, StrategySpace


class Explorer:
    """Select a strategy path from the strategy space using an exploration method."""

    def __init__(self, method: str = "thompson", c: float = 1.414, epsilon: float = 0.2):
        """
        Args:
            method: "ucb", "thompson", or "epsilon_greedy"
            c: UCB exploration constant
            epsilon: epsilon-greedy exploration rate
        """
        self.method = method
        self.c = c
        self.epsilon = epsilon

    def select_path(self, space: StrategySpace, domain: str = None) -> List[MilestoneNode]:
        """Select a strategy path from the space."""
        paths = space.get_paths(domain)
        if not paths:
            return []
        if len(paths) == 1:
            return paths[0]

        if self.method == "ucb":
            return self._select_ucb(paths)
        elif self.method == "thompson":
            return self._select_thompson(paths)
        elif self.method == "epsilon_greedy":
            return self._select_epsilon_greedy(paths)
        else:
            raise ValueError(f"Unknown exploration method: {self.method}")

    def _path_score(self, path: List[MilestoneNode]) -> float:
        """Aggregate success rate of a path (use leaf node)."""
        if not path:
            return 0.5
        return path[-1].success_rate()

    def _path_visits(self, path: List[MilestoneNode]) -> int:
        """Total visits of a path (use leaf node)."""
        if not path:
            return 0
        return path[-1].visit_count

    def _select_ucb(self, paths: List[List[MilestoneNode]]) -> List[MilestoneNode]:
        """UCB1 selection: score(path) = avg_success_rate + c * sqrt(ln(N) / n)"""
        total_visits = sum(self._path_visits(p) for p in paths)
        if total_visits == 0:
            return random.choice(paths)

        best_path = None
        best_score = float('-inf')

        for path in paths:
            n = self._path_visits(path)
            if n == 0:
                return path  # Unvisited path gets priority
            avg_success = self._path_score(path)
            ucb_score = avg_success + self.c * math.sqrt(math.log(total_visits) / n)
            if ucb_score > best_score:
                best_score = ucb_score
                best_path = path

        return best_path

    def _select_thompson(self, paths: List[List[MilestoneNode]]) -> List[MilestoneNode]:
        """Thompson Sampling: sample from Beta(alpha, beta) for each path."""
        best_path = None
        best_sample = float('-inf')

        for path in paths:
            if not path:
                continue
            leaf = path[-1]
            sample = random.betavariate(max(leaf.alpha, 0.01), max(leaf.beta_param, 0.01))
            if sample > best_sample:
                best_sample = sample
                best_path = path

        return best_path or paths[0]

    def _select_epsilon_greedy(self, paths: List[List[MilestoneNode]]) -> List[MilestoneNode]:
        """Epsilon-greedy: with probability epsilon pick random, otherwise pick best."""
        if random.random() < self.epsilon:
            return random.choice(paths)

        best_path = max(paths, key=lambda p: self._path_score(p))
        return best_path
