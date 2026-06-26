"""
Space Evolution — Module 3

Evolves the strategy space over time.
Supports: reflection, dpm (Decision Point Mining).
Selected via --evolution_method CLI arg.
"""

import json
from typing import List, Dict, Optional

from .strategy_space import MilestoneNode, StrategySpace
from .utils.openai_helpers import chat_completion_with_retries, extract_json_from_response


class Evolution:
    """Evolves the strategy space based on episode outcomes."""

    def __init__(self, method: str = "reflection", interval: int = 5,
                 llm_model: str = "gpt-4o"):
        """
        Args:
            method: "reflection" or "dpm"
            interval: For reflection — trigger every N episodes
            llm_model: Model used for LLM-based evolution
        """
        self.method = method
        self.interval = interval
        self.llm_model = llm_model
        self.episode_count = 0
        self.episode_buffer: List[Dict] = []  # Store recent episode summaries

    def on_episode_end(self, episode_data: Dict, success: bool,
                       strategy_space: StrategySpace, domain: str = ""):
        """
        Called after each episode ends.

        Args:
            episode_data: {"task_goal": str, "game_history": list, "path": list of node dicts}
            success: Whether the episode was successful
            strategy_space: The strategy space to evolve
            domain: Current domain
        """
        self.episode_count += 1
        self.episode_buffer.append({
            "episode_num": self.episode_count,
            "success": success,
            "task_goal": episode_data.get("task_goal", ""),
            "summary": self._summarize_episode(episode_data),
            "domain": domain,
        })

        # Keep buffer manageable
        if len(self.episode_buffer) > 20:
            self.episode_buffer = self.episode_buffer[-20:]

        if self.method == "reflection":
            self._reflection_evolve(strategy_space, domain)
        elif self.method == "dpm":
            self._dpm_evolve(episode_data, success, strategy_space, domain)
        else:
            raise ValueError(f"Unknown evolution method: {self.method}")

    def _summarize_episode(self, episode_data: Dict) -> str:
        """Create a brief summary of an episode."""
        history = episode_data.get("game_history", [])
        if not history:
            return "No actions taken."

        actions = []
        for entry in history[-10:]:  # Last 10 steps
            action = entry.get('action', 'N/A')
            url = entry.get('url', '')
            actions.append(f"  {action} (url: {url})" if url else f"  {action}")

        return f"Task: {episode_data.get('task_goal', 'N/A')}\nActions ({len(history)} steps, showing last {min(10, len(history))}):\n" + "\n".join(actions)

    # ── Reflection ──

    def _reflection_evolve(self, space: StrategySpace, domain: str):
        """Free Reflection: periodically ask LLM to improve the strategy space."""
        if self.episode_count % self.interval != 0:
            return

        print(f"[Evolution] Triggering reflection after {self.episode_count} episodes")

        # Prepare context
        space_summary = space.space_summary(domain)
        recent_episodes = "\n\n".join([
            f"Episode {ep['episode_num']} ({'SUCCESS' if ep['success'] else 'FAILED'}):\n{ep['summary']}"
            for ep in self.episode_buffer[-self.interval:]
        ])

        sys_prompt = """You are a strategy evolution agent. You analyze past episode outcomes and suggest improvements to a strategy space.

You must respond with a JSON object containing a list of operations to perform on the strategy space.

Each operation is a dict with:
- "op": one of "add_child", "add_branch", "prune", "update_node"
- "parent_id": (for add_child/add_branch) the parent node ID, or null for new root
- "node_id": (for prune/update_node) the target node ID
- "milestone": (for add_child/add_branch) the new milestone text
- "key_actions": (for add_child/add_branch/update_node) list of key action strings
- "pitfalls": (for add_child/add_branch/update_node) list of pitfall strings
- "success_signal": (for add_child/add_branch/update_node) success signal string

Respond with ONLY valid JSON: {"operations": [...]}"""

        prompt = f"""Current strategy space:
{space_summary}

Recent episodes:
{recent_episodes}

Based on the successes and failures above, suggest operations to improve the strategy space.
Focus on:
1. Adding new strategies for situations where existing ones failed
2. Pruning strategies that consistently fail
3. Updating key_actions or pitfalls based on observed patterns

Output your operations as JSON."""

        try:
            res = chat_completion_with_retries(
                model=self.llm_model,
                sys_prompt=sys_prompt,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7,
                max_retries=2,
                retry_interval_sec=10
            )
            if res and hasattr(res, 'choices') and res.choices:
                response_text = res.choices[0].message.content
                ops = extract_json_from_response(response_text)
                self._apply_operations(ops, space, domain)
        except Exception as e:
            print(f"[Evolution] Reflection failed: {e}")

    # ── Decision Point Mining ──

    def _dpm_evolve(self, episode_data: Dict, success: bool,
                    space: StrategySpace, domain: str):
        """Decision Point Mining: analyze failures to find alternative branches."""
        if success:
            return  # Only mine failures

        print(f"[Evolution] DPM: analyzing failed episode {self.episode_count}")

        history = episode_data.get("game_history", [])
        if len(history) < 2:
            return

        # Build trajectory summary
        trajectory = []
        for i, entry in enumerate(history):
            action = entry.get('action', 'N/A')
            url = entry.get('url', '')
            state_snippet = entry.get('state', '')[:200]
            trajectory.append(f"Step {i}: {action}\n  URL: {url}\n  State: {state_snippet}...")

        trajectory_text = "\n".join(trajectory)
        space_summary = space.space_summary(domain)

        sys_prompt = """You are a decision point analyzer. Given a failed web task trajectory, identify the key decision points where an alternative action could have led to success.

Respond with ONLY valid JSON:
{
  "decision_points": [
    {
      "step": <step number>,
      "original_action": "<what was done>",
      "alternative_milestone": "<what should be done instead>",
      "key_actions": ["<specific web actions to try>"],
      "reasoning": "<why this alternative might work>"
    }
  ]
}"""

        prompt = f"""Task: {episode_data.get('task_goal', 'N/A')}

Failed trajectory:
{trajectory_text}

Current strategy space:
{space_summary}

Identify 1-3 key decision points where a different approach could lead to success.
For each, suggest a new strategy branch to add to the space."""

        try:
            res = chat_completion_with_retries(
                model=self.llm_model,
                sys_prompt=sys_prompt,
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7,
                max_retries=2,
                retry_interval_sec=10
            )
            if res and hasattr(res, 'choices') and res.choices:
                response_text = res.choices[0].message.content
                result = extract_json_from_response(response_text)
                self._apply_dpm_results(result, space, domain)
        except Exception as e:
            print(f"[Evolution] DPM analysis failed: {e}")

    def _apply_dpm_results(self, result: Dict, space: StrategySpace, domain: str):
        """Apply DPM decision point results to the strategy space."""
        decision_points = result.get("decision_points", [])
        for dp in decision_points:
            milestone = dp.get("alternative_milestone", "")
            key_actions = dp.get("key_actions", [])
            if milestone:
                node_id = space.add_node(
                    milestone=milestone,
                    domain=domain,
                    key_actions=key_actions,
                    pitfalls=[dp.get("reasoning", "")],
                )
                print(f"[Evolution] DPM added new branch: {milestone} (id={node_id})")

    def _apply_operations(self, ops: Dict, space: StrategySpace, domain: str):
        """Apply LLM-suggested operations to the strategy space."""
        operations = ops.get("operations", [])
        for op in operations:
            op_type = op.get("op", "")
            try:
                if op_type == "add_child":
                    parent_id = op.get("parent_id")
                    node_id = space.add_node(
                        milestone=op.get("milestone", ""),
                        domain=domain,
                        parent_id=parent_id,
                        key_actions=op.get("key_actions", []),
                        pitfalls=op.get("pitfalls", []),
                        success_signal=op.get("success_signal", ""),
                    )
                    print(f"[Evolution] Added child node: {op.get('milestone', '')} (id={node_id})")

                elif op_type == "add_branch":
                    # Add as new root (independent strategy)
                    node_id = space.add_node(
                        milestone=op.get("milestone", ""),
                        domain=domain,
                        key_actions=op.get("key_actions", []),
                        pitfalls=op.get("pitfalls", []),
                        success_signal=op.get("success_signal", ""),
                    )
                    print(f"[Evolution] Added new branch: {op.get('milestone', '')} (id={node_id})")

                elif op_type == "prune":
                    node_id = op.get("node_id", "")
                    if node_id:
                        space.prune(node_id)
                        print(f"[Evolution] Pruned node: {node_id}")

                elif op_type == "update_node":
                    node_id = op.get("node_id", "")
                    if node_id:
                        updates = {}
                        if "key_actions" in op:
                            updates["key_actions"] = op["key_actions"]
                        if "pitfalls" in op:
                            updates["pitfalls"] = op["pitfalls"]
                        if "success_signal" in op:
                            updates["success_signal"] = op["success_signal"]
                        if "milestone" in op:
                            updates["milestone"] = op["milestone"]
                        space.update_node(node_id, **updates)
                        print(f"[Evolution] Updated node {node_id}")

            except Exception as e:
                print(f"[Evolution] Failed to apply operation {op_type}: {e}")
