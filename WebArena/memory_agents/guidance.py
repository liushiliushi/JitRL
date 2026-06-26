"""
Guidance Mode — Module 1

Converts selected strategy path into prompt text.
Supports: full_plan, step_by_step, hierarchical.
Selected via --guidance_mode CLI arg.

Also includes MilestoneTracker for step_by_step and hierarchical modes.
"""

from typing import List, Optional

from .strategy_space import MilestoneNode, StrategySpace
from .utils.openai_helpers import chat_completion_with_retries


class MilestoneTracker:
    """
    Tracks progress through a strategy path's milestones.
    Uses lightweight LLM call to check if current milestone is completed.
    """

    def __init__(self, path: List[MilestoneNode], llm_model: str = "gpt-4o"):
        self.path = path
        self.current_idx = 0
        self.llm_model = llm_model

    @property
    def current_milestone(self) -> Optional[MilestoneNode]:
        if self.current_idx < len(self.path):
            return self.path[self.current_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_idx >= len(self.path)

    def check_completion(self, observation: str) -> bool:
        """
        Check if the current milestone is completed based on observation.
        Returns True if milestone was completed and tracker advanced.
        """
        milestone = self.current_milestone
        if milestone is None:
            return False

        if not milestone.success_signal:
            # No success signal defined — skip to next
            self.current_idx += 1
            return True

        # Lightweight LLM check
        sys_prompt = "You are a milestone completion checker. Answer only 'yes' or 'no'."
        prompt = f"""Has the following milestone been completed?

Milestone: {milestone.milestone}
Success signal: {milestone.success_signal}

Current observation (partial):
{observation[:2000]}

Answer 'yes' if the milestone appears to be completed, 'no' otherwise."""

        try:
            res = chat_completion_with_retries(
                model=self.llm_model,
                sys_prompt=sys_prompt,
                prompt=prompt,
                max_tokens=10,
                temperature=0.0,
                max_retries=2,
                retry_interval_sec=5
            )
            if res and hasattr(res, 'choices') and res.choices:
                answer = res.choices[0].message.content.strip().lower()
                if answer.startswith('yes'):
                    self.current_idx += 1
                    print(f"[MilestoneTracker] Milestone completed: {milestone.milestone}")
                    return True
        except Exception as e:
            print(f"[MilestoneTracker] Error checking milestone: {e}")

        return False

    def advance(self):
        """Manually advance to next milestone."""
        if self.current_idx < len(self.path):
            self.current_idx += 1


class Guidance:
    """Generate prompt text from a strategy path based on guidance mode."""

    def __init__(self, mode: str = "hierarchical"):
        """
        Args:
            mode: "full_plan", "step_by_step", or "hierarchical"
        """
        self.mode = mode

    def generate(self, path: List[MilestoneNode], space: StrategySpace,
                 milestone_idx: int = 0, observation: str = "") -> str:
        """
        Generate guidance text for the current state.

        Args:
            path: The selected strategy path
            space: The full strategy space (for context)
            milestone_idx: Current milestone index (for step_by_step / hierarchical)
            observation: Current observation text (unused for now)

        Returns:
            Guidance text to inject into the prompt.
        """
        if not path:
            return ""

        if self.mode == "full_plan":
            return self._full_plan(path)
        elif self.mode == "step_by_step":
            return self._step_by_step(path, milestone_idx)
        elif self.mode == "hierarchical":
            return self._hierarchical(path, milestone_idx)
        else:
            raise ValueError(f"Unknown guidance mode: {self.mode}")

    def _full_plan(self, path: List[MilestoneNode]) -> str:
        """Full Plan Upfront: all milestones injected at once."""
        lines = ["## Strategy Plan (follow this step by step):\n"]
        for i, node in enumerate(path):
            status = f"Step {i + 1}"
            lines.append(f"### {status}: {node.milestone}")
            if node.key_actions:
                lines.append(f"  Key actions: {'; '.join(node.key_actions)}")
            if node.pitfalls:
                lines.append(f"  Pitfalls to avoid: {'; '.join(node.pitfalls)}")
            if node.success_signal:
                lines.append(f"  Success signal: {node.success_signal}")
            lines.append("")
        return "\n".join(lines)

    def _step_by_step(self, path: List[MilestoneNode], milestone_idx: int) -> str:
        """Step-by-Step: only show current milestone."""
        if milestone_idx >= len(path):
            return "## All milestones completed. Verify the result and submit your answer.\n"

        node = path[milestone_idx]
        lines = [
            f"## Current Milestone ({milestone_idx + 1}/{len(path)}): {node.milestone}\n",
        ]
        if node.key_actions:
            lines.append(f"Key actions to try: {'; '.join(node.key_actions)}")
        if node.pitfalls:
            lines.append(f"Watch out for: {'; '.join(node.pitfalls)}")
        if node.success_signal:
            lines.append(f"Move on when: {node.success_signal}")

        # Show progress
        if milestone_idx > 0:
            completed = [p.milestone for p in path[:milestone_idx]]
            lines.append(f"\nCompleted milestones: {', '.join(completed)}")

        remaining = len(path) - milestone_idx - 1
        if remaining > 0:
            lines.append(f"Remaining milestones: {remaining}")

        return "\n".join(lines)

    def _hierarchical(self, path: List[MilestoneNode], milestone_idx: int) -> str:
        """Hierarchical: high-level overview always visible, detailed for current step."""
        lines = ["## Strategy Overview:"]

        # High-level: show all milestones with status
        for i, node in enumerate(path):
            if i < milestone_idx:
                status = "DONE"
            elif i == milestone_idx:
                status = "CURRENT"
            else:
                status = "upcoming"
            lines.append(f"  {i + 1}. [{status}] {node.milestone}")

        lines.append("")

        # Detailed: current milestone
        if milestone_idx < len(path):
            node = path[milestone_idx]
            lines.append(f"## Current Focus: {node.milestone}")
            if node.key_actions:
                lines.append(f"  Key actions: {'; '.join(node.key_actions)}")
            if node.pitfalls:
                lines.append(f"  Pitfalls: {'; '.join(node.pitfalls)}")
            if node.success_signal:
                lines.append(f"  Move on when: {node.success_signal}")
        else:
            lines.append("## All milestones completed. Submit your answer.")

        return "\n".join(lines)
