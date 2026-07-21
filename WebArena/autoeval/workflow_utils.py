"""
Utilities for parsing BrowserGym experiment.log files into structured
trajectories (per-step reasoning and actions).

Adapted from Agent Workflow Memory (https://github.com/zorazrw/agent-workflow-memory),
webarena/autoeval/evaluate_trajectory.py, where these helpers are defined inline.
Used by autoeval/evaluate_trajectory.py for post-hoc trajectory evaluation.
"""

import ast

_LOOP_LOGGER_MARKER = "browsergym.experiments.loop - INFO -"


def load_blocks(path: str, handle_incomplete: bool = False) -> list[list[str]]:
    """Load blank-line separated blocks from the log file.

    Blocks alternate between agent reasoning (logged by browsergym.experiments.loop)
    and executed actions. Returns a list of blocks, each a list of stripped lines.
    """
    blocks, block = [], []
    with open(path, 'r') as f:
        for line in f:
            if line.strip() == "":
                if block:
                    blocks.append(block)
                block = []
            else:
                block.append(line.strip())
    # A log that ends mid-block (e.g. crashed episode) has no trailing blank line
    if block:
        blocks.append(block)
    if not handle_incomplete:
        assert len(blocks) % 2 == 0
    return blocks


def _is_valid_literal_arg(text: str) -> bool:
    """Check that an extracted argument is a string literal."""
    try:
        return isinstance(ast.literal_eval(text), str)
    except (ValueError, SyntaxError):
        return False


def remove_invalid_steps(actions: list[str], filter_scroll_noop: bool = False) -> list[str]:
    """Remove invalid steps from the action sequence.

    Drops click()/fill() calls whose element argument is not a string literal
    (i.e. malformed lines picked up from the log). With filter_scroll_noop=True,
    also drops scroll() and noop() actions, which carry no task progress signal.
    """
    valid_actions = []
    for a in actions:
        if filter_scroll_noop and ("scroll(" in a or "noop(" in a):
            continue
        if "click(" in a:
            if "(" not in a or ")" not in a:
                continue
            arg = a[a.index("(") + 1: a.index(")")]
            if _is_valid_literal_arg(arg):
                valid_actions.append(a)
        elif "fill(" in a:
            if "(" not in a or "," not in a:
                continue
            arg = a[a.index("(") + 1: a.index(",")].strip()
            if _is_valid_literal_arg(arg):
                valid_actions.append(a)
        else:
            valid_actions.append(a)
    return valid_actions


def extract_think_and_action(
    path: str,
    filter_scroll_noop: bool = False,
    handle_incomplete: bool = False,
) -> tuple[list[str], list[str]]:
    """Extract the task trajectory from the log file.

    Args:
        path: Path to experiment.log.
        filter_scroll_noop: Drop scroll()/noop() actions from each step.
        handle_incomplete: Tolerate truncated logs (crashed episodes): unpaired
            trailing blocks are ignored and think blocks missing the loop-logger
            marker fall back to the raw line instead of raising.

    Returns:
        (think_list, action_list): per-step reasoning strings and per-step
        lists of action strings, aligned by index.
    """
    blocks = load_blocks(path, handle_incomplete=handle_incomplete)
    think_list, action_list = [], []
    for i in range(1, len(blocks), 2):
        # action block: first line is a header, remaining lines are actions
        actions = remove_invalid_steps(blocks[i][1:], filter_scroll_noop=filter_scroll_noop)
        if len(actions) == 0:
            continue
        # think block: reasoning is on the last line after the loop-logger marker
        think_line = blocks[i - 1][-1]
        if _LOOP_LOGGER_MARKER in think_line:
            idx = think_line.index(_LOOP_LOGGER_MARKER)
            think = think_line[idx + len(_LOOP_LOGGER_MARKER):].strip()
        elif handle_incomplete:
            think = think_line
        else:
            raise ValueError(f"Log line missing '{_LOOP_LOGGER_MARKER}': {think_line}")
        action_list.append(actions)
        think_list.append(think)

    assert len(think_list) == len(action_list)
    return think_list, action_list
