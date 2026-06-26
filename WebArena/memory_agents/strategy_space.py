"""
Strategy Space — Module 0

Stores structured strategy knowledge as MilestoneNodes.
Supports 4 structures: flat, tree, dag, action_tree.
All selected via --strategy_structure CLI arg.
"""

import json
import os
import random
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class MilestoneNode:
    id: str
    milestone: str                    # "在 OneStopShop 搜索目标商品"
    key_actions: List[str] = field(default_factory=list)   # ["click [search_bar]", "type [search_bar] '{query}'"]
    preconditions: List[str] = field(default_factory=list)  # ["已打开首页"]
    success_signal: str = ""          # "搜索结果页面显示商品列表"
    pitfalls: List[str] = field(default_factory=list)       # ["搜索框可能被 overlay 遮挡"]
    domain: str = ""                  # "shopping" / "reddit" / "gitlab" / "cms" / "map"
    visit_count: int = 0
    alpha: float = 1.0                # Beta(alpha, beta) for Thompson Sampling
    beta_param: float = 1.0
    children: List[str] = field(default_factory=list)       # child node IDs
    parents: List[str] = field(default_factory=list)        # parent node IDs (DAG)

    def success_rate(self) -> float:
        total = self.alpha + self.beta_param - 2  # subtract priors
        if total <= 0:
            return 0.5
        return (self.alpha - 1) / total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'MilestoneNode':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class StrategySpace:
    """
    Unified strategy space supporting flat/tree/dag/action_tree structures.
    """

    def __init__(self, structure: str = "dag", seed_file: str = None, save_dir: str = None):
        """
        Args:
            structure: One of "flat", "tree", "dag", "action_tree"
            seed_file: Path to initial strategy seeds JSON
            save_dir: Directory for persistence
        """
        self.structure = structure
        self.save_dir = save_dir
        self.nodes: Dict[str, MilestoneNode] = {}
        self._next_id = 0

        # Load from save first, then seed if no save exists
        loaded = False
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "strategy_space.json")
            if os.path.exists(save_path):
                self.load(save_path)
                loaded = True

        if not loaded and seed_file and os.path.exists(seed_file):
            self._load_seeds(seed_file)

    def _generate_id(self) -> str:
        self._next_id += 1
        return f"node_{self._next_id}"

    # ── Node CRUD ──

    def add_node(self, milestone: str, domain: str = "", parent_id: str = None,
                 key_actions: List[str] = None, preconditions: List[str] = None,
                 success_signal: str = "", pitfalls: List[str] = None) -> str:
        """Add a new node. Returns node ID."""
        node_id = self._generate_id()
        node = MilestoneNode(
            id=node_id,
            milestone=milestone,
            key_actions=key_actions or [],
            preconditions=preconditions or [],
            success_signal=success_signal,
            pitfalls=pitfalls or [],
            domain=domain,
        )

        if parent_id and parent_id in self.nodes:
            parent = self.nodes[parent_id]
            parent.children.append(node_id)
            node.parents.append(parent_id)

        self.nodes[node_id] = node
        return node_id

    def add_edge(self, from_id: str, to_id: str):
        """Add edge (DAG only). Creates parent-child link."""
        if self.structure not in ("dag", "action_tree"):
            print(f"[StrategySpace] add_edge only supported for dag/action_tree, got {self.structure}")
            return
        if from_id in self.nodes and to_id in self.nodes:
            if to_id not in self.nodes[from_id].children:
                self.nodes[from_id].children.append(to_id)
            if from_id not in self.nodes[to_id].parents:
                self.nodes[to_id].parents.append(from_id)

    def remove_edge(self, from_id: str, to_id: str):
        """Remove edge between two nodes."""
        if from_id in self.nodes and to_id in self.nodes:
            if to_id in self.nodes[from_id].children:
                self.nodes[from_id].children.remove(to_id)
            if from_id in self.nodes[to_id].parents:
                self.nodes[to_id].parents.remove(from_id)

    def prune(self, node_id: str):
        """Remove a node and all its descendants (tree/dag)."""
        if node_id not in self.nodes:
            return
        # Remove from parents' children lists
        node = self.nodes[node_id]
        for pid in node.parents:
            if pid in self.nodes and node_id in self.nodes[pid].children:
                self.nodes[pid].children.remove(node_id)
        # Remove descendants recursively
        to_remove = [node_id]
        while to_remove:
            nid = to_remove.pop()
            if nid in self.nodes:
                to_remove.extend(self.nodes[nid].children)
                # Clean up parent references
                for cid in self.nodes[nid].children:
                    if cid in self.nodes and nid in self.nodes[cid].parents:
                        self.nodes[cid].parents.remove(nid)
                del self.nodes[nid]

    def update_node(self, node_id: str, **kwargs):
        """Update fields of an existing node."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            for k, v in kwargs.items():
                if hasattr(node, k) and k != 'id':
                    setattr(node, k, v)

    # ── Path operations ──

    def _get_roots(self, domain: str = None) -> List[str]:
        """Get root nodes (nodes with no parents)."""
        roots = []
        for nid, node in self.nodes.items():
            if not node.parents:
                if domain is None or node.domain == domain or node.domain == "":
                    roots.append(nid)
        return roots

    def get_paths(self, domain: str = None) -> List[List[MilestoneNode]]:
        """
        Get all root-to-leaf paths for a domain.

        For flat structure: each root is a standalone strategy (single-node path).
        For tree/dag/action_tree: enumerate root-to-leaf paths.
        """
        if self.structure == "flat":
            # Each node is a standalone strategy
            paths = []
            for nid, node in self.nodes.items():
                if domain is None or node.domain == domain or node.domain == "":
                    paths.append([node])
            return paths

        # Tree / DAG / action_tree: DFS from roots
        roots = self._get_roots(domain)
        all_paths = []
        for root_id in roots:
            self._dfs_paths(root_id, [], all_paths, domain)
        return all_paths

    def _dfs_paths(self, node_id: str, current_path: List[MilestoneNode],
                   all_paths: List[List[MilestoneNode]], domain: str = None):
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        if domain and node.domain != domain and node.domain != "":
            return
        current_path = current_path + [node]

        # Filter children by domain
        valid_children = [
            cid for cid in node.children
            if cid in self.nodes and (domain is None or self.nodes[cid].domain == domain or self.nodes[cid].domain == "")
        ]

        if not valid_children:
            # Leaf node — record path
            all_paths.append(current_path)
        else:
            for child_id in valid_children:
                self._dfs_paths(child_id, current_path, all_paths, domain)

    # ── Statistics update ──

    def update_path_stats(self, path: List[MilestoneNode], success: bool, gamma: float = 0.9):
        """
        Update visit count and Beta distribution parameters along a path.

        For tree/dag: gamma-discounted backpropagation from leaf to root.
        For flat: direct update on the single node.
        """
        if not path:
            return

        # Update from leaf to root with discount
        discount = 1.0
        for node in reversed(path):
            if node.id in self.nodes:
                real_node = self.nodes[node.id]
                real_node.visit_count += 1
                if success:
                    real_node.alpha += discount
                else:
                    real_node.beta_param += discount
                discount *= gamma

    # ── Text rendering ──

    def to_text(self, path: List[MilestoneNode]) -> str:
        """Convert a strategy path to human-readable text for prompt injection."""
        lines = []
        for i, node in enumerate(path):
            lines.append(f"Step {i + 1}: {node.milestone}")
            if node.key_actions:
                lines.append(f"  Key actions: {'; '.join(node.key_actions)}")
            if node.pitfalls:
                lines.append(f"  Watch out: {'; '.join(node.pitfalls)}")
            if node.success_signal:
                lines.append(f"  Success signal: {node.success_signal}")
        return "\n".join(lines)

    def space_summary(self, domain: str = None) -> str:
        """Get a text summary of the entire strategy space for a domain."""
        paths = self.get_paths(domain)
        if not paths:
            return "No strategies available."

        lines = [f"Strategy Space ({len(paths)} strategies for domain '{domain or 'all'}'):"]
        for i, path in enumerate(paths):
            milestones = " → ".join([n.milestone for n in path])
            last_node = path[-1]
            sr = last_node.success_rate()
            visits = sum(n.visit_count for n in path)
            lines.append(f"  Strategy {i + 1}: {milestones} [success_rate={sr:.2f}, visits={visits}]")
        return "\n".join(lines)

    # ── Persistence ──

    def save(self, path: str = None):
        if path is None and self.save_dir:
            path = os.path.join(self.save_dir, "strategy_space.json")
        if path is None:
            return

        data = {
            "structure": self.structure,
            "next_id": self._next_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.structure = data.get("structure", self.structure)
        self._next_id = data.get("next_id", 0)
        self.nodes = {}
        for nid, ndata in data.get("nodes", {}).items():
            self.nodes[nid] = MilestoneNode.from_dict(ndata)

    def _load_seeds(self, seed_file: str):
        """Load initial strategies from a seed file."""
        if not os.path.exists(seed_file):
            return
        with open(seed_file, 'r', encoding='utf-8') as f:
            seeds = json.load(f)

        # Seeds format: list of strategy dicts with "domain", "milestones" list
        for strategy in seeds:
            domain = strategy.get("domain", "")
            milestones = strategy.get("milestones", [])
            parent_id = None
            for ms in milestones:
                node_id = self.add_node(
                    milestone=ms.get("milestone", ""),
                    domain=domain,
                    parent_id=parent_id,
                    key_actions=ms.get("key_actions", []),
                    preconditions=ms.get("preconditions", []),
                    success_signal=ms.get("success_signal", ""),
                    pitfalls=ms.get("pitfalls", []),
                )
                parent_id = node_id
