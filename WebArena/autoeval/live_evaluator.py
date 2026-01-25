"""
Live evaluator for real-time task evaluation during agent execution.
This evaluator runs during the agent's task execution to provide immediate feedback.
"""
import json
from pathlib import Path
from typing import Any, Dict, List
from playwright.sync_api import Page

from autoeval.enhanced_evaluator import (
    evaluator_router,
    create_trajectory_from_info,
    PseudoPage,
    Action,
    StateInfo
)


def evaluate_on_live_page(
    page: Page,
    eval_config: Dict[str, Any],
    current_url: str = "",
    last_message: str = "",
    action_history: List[str] = None,
    config_file: str = None,
    task_name: str = None
) -> float:
    """
    Evaluate the current page state against evaluation criteria.

    Args:
        page: Playwright page object
        eval_config: Evaluation configuration dict
        current_url: Current URL of the page
        last_message: Last message from agent
        action_history: List of actions taken
        config_file: Path to config file (optional, for loading full config)
        task_name: Task name (e.g., "webarena.700") to infer config file

    Returns:
        score: Float between 0.0 and 1.0 indicating task completion
    """
    try:
        # Infer config file from task_name if not provided
        if not config_file and task_name:
            import re
            import os
            task_id = re.search(r'\.(\d+)$', task_name)
            if task_id:
                # Try config_files directory first
                config_file = os.path.join("config_files", f"{task_id.group(1)}.json")
                if not os.path.exists(config_file):
                    # Fallback to config_files_lite
                    config_file = os.path.join("config_files_lite", f"{task_id.group(1)}.json")
                    if not os.path.exists(config_file):
                        config_file = None

        # Get evaluation types
        eval_types = eval_config.get("eval_types", [])

        if not eval_types:
            # No evaluation configured
            return 0.0

        # Create a minimal trajectory for evaluation
        trajectory = []

        # Add a state info (empty for now, since we're evaluating live)
        state_info = StateInfo(state="")
        trajectory.append(state_info)

        # Add final action with last message
        final_action = Action(
            action_type="send_msg_to_user",
            answer=last_message
        )
        trajectory.append(final_action)

        # Get current URL from page if not provided
        if not current_url:
            current_url = page.url if page else "about:blank"

        # For live evaluation, DON'T use PseudoPage - use the real page directly
        # This way we can access the current form state
        score = 1.0

        print(f"\n{'='*80}")
        print(f"🔴 LIVE EVALUATION - Current URL: {current_url}")
        print(f"{'='*80}")

        # URL match evaluation
        if "url_match" in eval_types:
            ref_url = eval_config.get("reference_url", "")
            if ref_url:
                from autoeval.enhanced_evaluator import URLEvaluator, replace_url_placeholders

                # Clean URLs
                def clean_url(url: str) -> str:
                    return str(url).rstrip("/")

                pred_url = clean_url(current_url)
                ref_urls = [clean_url(replace_url_placeholders(u)) for u in ref_url.split(" |OR| ")]

                # Simple contains check
                url_match = any(ref in pred_url for ref in ref_urls)
                url_score = 1.0 if url_match else 0.0

                print(f"✓ URL Match: {url_match} (score: {url_score})")
                print(f"  Expected: {ref_urls}")
                print(f"  Got: {pred_url}")

                score *= url_score

        # String match evaluation
        if "string_match" in eval_types:
            ref_answers = eval_config.get("reference_answers", {})
            if ref_answers and last_message:
                from autoeval.enhanced_evaluator import StringEvaluator

                string_score = 1.0
                for approach, value in ref_answers.items():
                    if approach == "exact_match":
                        match = StringEvaluator.exact_match(ref=value, pred=last_message)
                        string_score *= match
                    elif approach == "must_include":
                        if isinstance(value, list):
                            for v in value:
                                match = StringEvaluator.must_include(ref=v, pred=last_message)
                                string_score *= match

                print(f"✓ String Match: score={string_score}")
                score *= string_score

        # Program HTML evaluation - requires config file
        if "program_html" in eval_types and config_file:
            try:
                print(f"📋 Using config file: {config_file}")
                # Use the full evaluator router for program_html
                # IMPORTANT: Pass the real page object, NOT PseudoPage
                # This allows evaluator to access current form state
                enhanced_evaluator = evaluator_router(config_file)
                html_score = enhanced_evaluator(trajectory, config_file, page, client=None)

                print(f"✓ HTML Content: score={html_score}")
                score *= html_score
            except Exception as e:
                print(f"✗ HTML evaluation error: {e}")
                import traceback
                traceback.print_exc()
                # Don't fail completely, just skip HTML checks

        print(f"\n{'='*80}")
        print(f"LIVE EVALUATION SCORE: {score}")
        print(f"{'='*80}\n")

        return score

    except Exception as e:
        print(f"Error in live evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0
