"""
WARNING DEPRECATED WILL BE REMOVED SOON
"""

import os
import json
import shutil
import sys, subprocess
import argparse
from pathlib import Path
import logging
from datetime import datetime
import re

# Load WebArena environment variables BEFORE any browsergym imports
def _load_webarena_env():
    """Load WebArena environment variables from env_setup.txt or .env"""
    env_file = Path(__file__).parent / "env_setup.txt"
    if not env_file.exists():
        env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line = line[7:]
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

_load_webarena_env()

# Suppress browsergym warnings and info logs
logging.getLogger('browsergym.core.env').setLevel(logging.ERROR)
logging.getLogger('browsergym.experiments.loop').setLevel(logging.WARNING)

# NOTE: Configuration is managed by switch_config.sh script
# Use ./switch_config.sh <config_dir> to switch between config_files and config_files_lite
# This avoids conflicts and ensures consistent configuration across runs
#
# The code below is DISABLED to prevent automatic config switching
# If you want to use --config_dir parameter, uncomment the code below
# But be aware that it may override your switch_config.sh settings

from browsergym.experiments import ExpArgs, EnvArgs
from browsergym.core.env import BrowserEnv

from memory_agents.memory_agent import MemoryAgentArgs
from memory_agents.utils.openai_helpers import TokenLimitExceededError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    # Environment settings
    parser.add_argument(
        "--task_name",
        type=str, default="webarena.0", help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'"
    )
    
    parser.add_argument(
        "--start_url",
        type=str, default="https://www.google.com", help="Starting URL (only for the openended task)."
    )
    
    parser.add_argument(
        "--slow_mo", 
        type=int, default=30, help="Slow motion delay for the playwright actions."
    )
    
    parser.add_argument(
        "--headless",
        type=str2bool, default=True, help="Run the experiment in headless mode (hides the browser windows)."
    )
    
    parser.add_argument(
        "--demo_mode",
        type=str2bool, default=True, help="Add visual effects when the agents performs actions."
    )
    
    parser.add_argument(
        "--use_html",
        type=str2bool, default=False, help="Use HTML in the agent's observation space."
    )

    parser.add_argument(
        "--save_html",
        type=str2bool, default=False, help="Save HTML content of each step to txt files in the results directory."
    )

    parser.add_argument(
        "--use_ax_tree",
        type=str2bool, default=True, help="Use AX tree in the agent's observation space."
    )
    
    parser.add_argument(
        "--use_screenshot",
        type=str2bool, default=True, help="Use screenshot in the agent's observation space."
    )
    
    parser.add_argument(
        "--multi_actions", 
        type=str2bool, default=True, help="Allow multi-actions in the agent."
    )
    
    parser.add_argument(
        "--action_space",
        type=str, default="bid", choices=["python", "bid", "coord", "bid+coord", "bid+nav", "coord+nav", "bid+coord+nav"], help=""
    )
    
    parser.add_argument(
        "--use_history",
        type=str2bool, default=True, help="Use history in the agent's observation space."
    )
    
    parser.add_argument(
        "--use_thinking",
        type=str2bool, default=True, help="Use thinking in the agent (chain-of-thought prompting)."
    )
    
    parser.add_argument(
        "--max_steps",
        type=int, default=10, help="Maximum number of steps to take for each task.",
    )
    
    parser.add_argument(
        "--workflow_path",
        type=str, default=None, help="Path to the memory file to load for the agent.",
    )
    
    parser.add_argument(
        '--seed',
        default=0, type=int, help="Random seed for reproducibility. If None, a random seed is used."
    )

    # LLM settings
    parser.add_argument(
        "--llm_model",
        type=str, default="gpt-4o", help="LLM model for web agent. Can be API model name (e.g., 'gpt-4o', 'claude-3-opus') or local model path (e.g., '/path/to/local/model')",
    )

    parser.add_argument(
        "--llm_eval",
        type=str, default=None, help="LLM model for evaluation (if not specified, uses llm_model)",
    )

    parser.add_argument(
        "--llm_extract",
        type=str, default=None, help="LLM model for extracting effective action sequences (if not specified, uses llm_model)",
    )

    parser.add_argument(
        "--use_screenshot_eval",
        type=str2bool, default=False, help="Use screenshots for visual evaluation of steps (requires vision-capable model like gpt-4o)",
    )

    parser.add_argument(
        "--use_screenshot_action",
        type=str2bool, default=False, help="Use screenshots when generating actions (requires vision-capable model like gpt-4o)",
    )

    parser.add_argument(
        "--disable_memory",
        type=str2bool, default=False, help="Disable memory functionality (no retrieval, storage, or step evaluation). Agent relies purely on model capabilities.",
    )

    parser.add_argument(
        '--top_actions',
        default=3, type=int, help="Number of potential action."
    )
    
    parser.add_argument(
        '--llm_temperature',
        default=0.8, type=float, help="Temperature for the agent's LLM."
    )

    parser.add_argument(
        '--logit_mode',
        type=str, default="verbalized", choices=["token", "verbalized"],
        help="Method to extract action probabilities: 'token' uses token logits (GPT), 'verbalized' uses model-provided confidence scores (Gemini)."
    )

    parser.add_argument(
        '--max_memory',
        default=30, type=int, help="Maximum number of past states to keep in memory for the agent."
    )
    
    parser.add_argument(
        '--gamma',
        default=0.1, type=float, help="Discount factor for computing returns in cross-episode memory."
    )

    parser.add_argument(
        '--max_trajectory_window',
        default=5, type=int, help="Maximum window size for trajectory comparison (uses sliding window for longer trajectories)."
    )

    # Debug options
    parser.add_argument(
        '--debug_info',
        default=False, action=argparse.BooleanOptionalAction, help='Print detailed info updates during game episodes.'
    )
    
    parser.add_argument(
        '--track_valid_changes',
        default=False, action=argparse.BooleanOptionalAction, help='Track valid action changes (if applicable).'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--agent_type',
        type=str,
        default='memory',
        choices=['memory'],
        help='Agent implementation to use. "memory" uses logit adjustment with cross-episode memory.'
    )
    
    parser.add_argument(
        '--eval_runs',
        type=int, default=50, help='Number of episodes to run for statistical evaluation.'
    )
    
    parser.add_argument(
        '--test_case_start',
        type=int, default=None, help='Starting test case index (e.g., 0). If specified, will run test cases from start to end.'
    )
    
    parser.add_argument(
        '--test_case_end',
        type=int, default=None, help='Ending test case index (e.g., 10). If specified, will run test cases from start to end.'
    )
    
    parser.add_argument(
        '--repeat_runs',
        type=int, default=1, help='Number of times to repeat the test cases (for computing cumulative accuracy).'
    )
    
    parser.add_argument(
        '--evol_temperature',
        default=0.7, type=float, help="Temperature for the evolutionary's LLM."
    )
    
    # Summary agent parameters
    parser.add_argument(
        '--summary_temperature',
        type=float, default=0.8, help='Temperature for the summarization LLM.'
    )
    
    parser.add_argument(
        '--summary_max_tokens',
        type=int, default=300, help='Maximum tokens for summarization response.'
    )
    
    # RAG agent parameters
    parser.add_argument(
        '--retrieval_top_k',
        type=int, default=3, help='Number of top-k most relevant history entries to retrieve.'
    )
    
    parser.add_argument(
        '--retrieval_threshold',
        type=float, default=0.1, help='Similarity threshold for retrieving relevant history entries.'
    )

    parser.add_argument(
        '--embedding_api_key',
        type=str, default='None', help='API key for embedding API (if different from main LLM API key).'
    )
    
    parser.add_argument(
        '--rag_temperature',
        type=float, default=0.4, help='Temperature for the RAG enhancement LLM.'
    )
    
    parser.add_argument(
        '--rag_max_tokens',
        type=int, default=400, help='Maximum tokens for RAG enhancement response.'
    )
    
    parser.add_argument(
        '--initial_prompts_file',
        default='initial_prompts.json', type=str, help='JSON file with initial prompts to seed the pool (relative to project root or absolute).'
    )
    
    parser.add_argument(
        '--exploration_constant',
        default=1.0, type=float, help='Exploration constant for UCB calculation in tree-based agent.'
    )

    parser.add_argument(
        '--depth_constant',
        default=0.8, type=float, help='Decay factor of exploration term in tree-based agent.'
    )

    parser.add_argument(
        '--freeze_on_win',
        default=True, action=argparse.BooleanOptionalAction, help='Once any node reaches win_freeze_threshold, stop exploration and reuse the best prompt thereafter.'
    )
    parser.add_argument(
        '--win_freeze_threshold',
        type=int, default=0, help='Score threshold to freeze on win (e.g., 310 for detective). 0 disables freezing.'
    )
    
    parser.add_argument(
        '--force_best_after_drop',
        default=True, action=argparse.BooleanOptionalAction, help='If the last episode score drops far below best, force exploiting the best prompt next episode.'
    )
    
    parser.add_argument(
        '--drop_threshold',
        type=int, default=50, help='Score drop margin vs best to trigger forced exploit.'
    )

    # Cross-episode memory toggle (few-shot positives + negative contrast during evolution)
    parser.add_argument(
        '--enable_cross_mem',
        default=True, action=argparse.BooleanOptionalAction, help='Enable cross-episode memory: store successful/failed snippets across episodes, few-shot retrieval, and negative-contrast evolution.'
    )

    parser.add_argument(
        '--save_memory',
        default=True, action=argparse.BooleanOptionalAction, help='Save memory at the end of each episode. When False, memory retrieval still works but no new memories are saved.'
    )

    parser.add_argument(
        '--task_similarity_threshold',
        type=float, default=0.27,
        help='Task similarity threshold for memory retrieval. Higher values require more similar tasks to share memories. Default: 0.27'
    )

    # Config directory selection
    parser.add_argument(
        '--config_dir',
        type=str, default='config_files',
        choices=['config_files_lite', 'config_files'],
        help='Configuration directory: "config_files_lite" for WebArena-Lite (165 tasks) or "config_files" for full WebArena (812 tasks).'
    )

    parser.add_argument(
        '--use_llm_success_eval',
        type=str2bool, default=False,
        help='Use LLM to evaluate task success instead of config-based ground truth evaluation. When enabled, the LLM will judge whether the task was completed successfully based on the trajectory and task goal.'
    )

    parser.add_argument(
        '--result_dir',
        type=str, default='results',
        help='Directory to save experiment results. Default: "results"'
    )

    return parser.parse_args()


# Pickleable wrapper class for make_agent (must be at module level, not inside function)
class MakeAgentWrapper:
    """Wrapper class to make agent and save reference to exp_args."""
    def __init__(self, original_make_agent, exp_args_ref):
        self.original_make_agent = original_make_agent
        self.exp_args_ref = exp_args_ref
    
    def __call__(self, *args_inner, **kwargs):
        agent = self.original_make_agent(*args_inner, **kwargs)
        if self.exp_args_ref is not None:
            self.exp_args_ref.agent = agent
            # Store exp_dir reference in agent for screenshot access during inference
            # This will be used to pass screenshots_dir to retrieve_similar
            agent._exp_args = self.exp_args_ref
        return agent
    
    def __getstate__(self):
        """Custom pickle state - exclude exp_args_ref to avoid RLock pickle errors."""
        state = self.__dict__.copy()
        # Don't pickle exp_args_ref as it may contain unpickleable objects (like RLock)
        state['exp_args_ref'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # exp_args_ref will be None after unpickling, but that's okay
        # The agent will still be created correctly


def run_single_episode(args, task_name, reused_agent=None):
    """Run a single episode and return the success status."""
    if (args.workflow_path is not None) and (not os.path.exists(args.workflow_path)):
        open(args.workflow_path, "w").close()

    # Load task config to get site information
    task_id = task_name.split('.')[-1]
    config_path = os.path.join(args.config_dir, f"{task_id}.json")
    task_sites = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            task_sites = config.get('sites', [])
    args.task_sites = task_sites  # Pass site info to agent via args

    # Monkey patch BrowserEnv.step() to disable mid-episode termination
    # This prevents program_html tasks from terminating early when score > 0
    # BUT still allows termination when agent explicitly stops (send_msg_to_user or infeasible)
    _original_step = BrowserEnv.step

    def patched_step(self, action: str):
        """Modified step with live evaluation on current page."""
        # Call original step to preserve all validation and page checks
        obs, reward, terminated, truncated, info = _original_step(self, action)

        # Check if agent explicitly requested to stop
        # This happens when last chat message is from "assistant" or "infeasible" role
        agent_requested_stop = False
        last_message = ""
        if self.chat.messages:
            last_msg_role = self.chat.messages[-1].get("role", "")
            if last_msg_role in ["assistant", "infeasible"]:
                agent_requested_stop = True
                # Extract message for string_match evaluation
                last_message = self.chat.messages[-1].get("content", "")

        # ✨ NEW: Live evaluation on current page
        try:
            from autoeval.live_evaluator import evaluate_on_live_page

            # Get current URL
            current_url = obs.get("url", "")
            if not current_url and hasattr(self, 'page') and self.page:
                try:
                    current_url = self.page.url
                except:
                    current_url = ""

            # Load task config for evaluation
            config_path = None
            if hasattr(self, 'task') and isinstance(self.task, dict):
                eval_config = self.task.get('eval', {})
            elif hasattr(self, 'task') and hasattr(self.task, 'config_file'):
                # task is a Task object with config_file attribute
                config_path = self.task.config_file
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        eval_config = config.get('eval', {})
                else:
                    eval_config = {}
            elif hasattr(self, 'task_name'):
                # Load config from file
                import re
                task_id = re.search(r'\.(\d+)$', self.task_name)
                if task_id:
                    config_path = os.path.join(args.config_dir, f"{task_id.group(1)}.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            eval_config = config.get('eval', {})
                    else:
                        eval_config = {}
                else:
                    eval_config = {}
            else:
                eval_config = {}

            # Only perform live evaluation if program_html is in eval_types
            # Other evaluation types (string_match, url_match, etc.) use post-hoc evaluation
            eval_types = eval_config.get('eval_types', [])
            if 'program_html' in eval_types and eval_config and hasattr(self, 'page') and self.page:
                live_score = evaluate_on_live_page(
                    page=self.page,
                    eval_config=eval_config,
                    current_url=current_url,
                    last_message=last_message,
                    action_history=[],
                    config_file=config_path
                )

                # Update reward with live evaluation result
                reward = live_score
                info['live_evaluation'] = True
                info['live_eval_score'] = live_score

                # If task is complete (score = 1.0), mark as terminated
                if live_score >= 1.0:
                    print(f"\n{'='*80}")
                    print(f"✅ LIVE EVALUATION: TASK COMPLETED SUCCESSFULLY (score={live_score})")
                    print(f"{'='*80}\n")
                    terminated = True
                    info['success'] = True
                    # Force agent stop by adding assistant message
                    agent_requested_stop = True
                elif live_score > 0.0:
                    print(f"\n[Live Eval] Partial progress: score={live_score}")
        except Exception as e:
            print(f"\n[Live Eval] Error during live evaluation: {e}")
            import traceback
            traceback.print_exc()

        # Override terminated to prevent early termination based on score > 0
        # But preserve termination for:
        # 1. Agent explicitly requested stop (send_msg_to_user)
        # 2. Infeasible action
        # 3. Live eval determined task is complete
        if not agent_requested_stop and not (self.terminate_on_infeasible and self.infeasible_message_received):
            terminated = False

        return obs, reward, terminated, truncated, info

    BrowserEnv.step = patched_step
    print("🔧 Patched BrowserEnv.step() with LIVE EVALUATION enabled")
    print("   - Only enabled for tasks with 'program_html' evaluation")
    print("   - Evaluates on current page without reloading")
    print("   - Verifies temporary states (forms, selections, etc.)")
    print("   - Auto-terminates when task is complete (score=1.0)")
    print("   - Other eval types (string_match, url_match) use post-hoc evaluation")

    env_args = EnvArgs(
        task_name=task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,
        viewport={"width": 1500, "height": 1280},
        slow_mo=args.slow_mo,
    )

    # Select agent args based on --agent_type
    # Map agent types to their constructors
    _AGENT_MAP = {
        'memory': lambda args: MemoryAgentArgs(args=args),
    }

    constructor = _AGENT_MAP.get(args.agent_type)

    if constructor:
        try:
            selected_agent_args = constructor(args)
        except Exception as e:
            print(f"[WARNING] Failed to initialize {args.agent_type} ({e}). Falling back to MemoryAgent.")
            selected_agent_args = MemoryAgentArgs(args=args)
    else:
        print(f"[WARNING] Unsupported agent type: {args.agent_type}. Falling back to MemoryAgent.")
        selected_agent_args = MemoryAgentArgs(args=args)

    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=selected_agent_args,
    )

    # Clear entire results directory for this task before starting
    results_dir = Path(args.result_dir) / task_name
    if results_dir.exists():
        shutil.rmtree(results_dir)
        print(f"[INFO] Removed old results directory: {results_dir}")

    # Create the results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare exp_args but skip pickle to avoid RLock errors
    # We'll manually do the preparation steps without pickle
    if exp_args.env_args.task_seed is None:
        import numpy as np
        SEED_MAX = 2**31 - 1
        exp_args.env_args.task_seed = np.random.randint(0, SEED_MAX)

    if exp_args.exp_name is None:
        task_name = exp_args.env_args.task_name
        exp_args.exp_name = f"{exp_args.agent_args.agent_name}_on_{task_name}_{exp_args.env_args.task_seed}"

    # Generate exp_dir manually (simplified version)
    exp_args.make_id()
    exp_args.exp_date = datetime.now()
    
    # Use our custom results_dir directly
    exp_args.exp_dir = results_dir
    exp_args.exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using exp_dir: {exp_args.exp_dir}")
    
    # Skip pickle.save to avoid RLock errors - we don't need to save exp_args.pkl
    # The important data (states, results) will be saved separately

    # Wrap make_agent to save agent reference
    _original_make_agent = exp_args.agent_args.make_agent
    exp_args.agent_args.make_agent = MakeAgentWrapper(_original_make_agent, exp_args)

    exp_args.run()
    
    # Save state information to results folder
    agent = exp_args.agent
    if hasattr(agent, 'game_history') and agent.game_history:
        states_data = []
        for idx, entry in enumerate(agent.game_history):
            state_info = {
                'step': idx,
                'state': entry.get('state', ''),
                'action': entry.get('action', ''),
                'reward': entry.get('reward', 0),
                'score': entry.get('score', 0),
                'full_response': entry.get('full_response', ''),
                'url': entry.get('url', '')
            }
            states_data.append(state_info)

        # Extract the final URL from the last step's observation
        # This is critical for program_html evaluation which needs to check the final page state
        # The game_history stores URLs BEFORE actions, but we need the URL AFTER the last action
        final_url = states_data[-1]['url'] if states_data else 'about:blank'

        # Try to get the final URL from the last step pickle file
        # The pickle contains the observation after the action was executed
        import pickle, gzip
        last_step_num = len(states_data)  # steps are 0-indexed, but we want the observation after the last action
        last_step_file = results_dir / f"step_{last_step_num}.pkl.gz"

        if last_step_file.exists():
            try:
                with gzip.open(last_step_file, 'rb') as f:
                    step_data = pickle.load(f)
                    # step_data is a StepInfo object with 'obs' field containing observation dict
                    if hasattr(step_data, 'obs') and isinstance(step_data.obs, dict):
                        if 'url' in step_data.obs:
                            final_url = step_data.obs['url']
                            print(f"✓ Extracted final URL from {last_step_file.name}: {final_url}")
                        else:
                            print(f"⚠ No URL found in observation from {last_step_file.name}")
                    else:
                        print(f"⚠ Unexpected step data structure in {last_step_file.name}")
            except Exception as e:
                print(f"⚠ Warning: Could not extract final URL from {last_step_file.name}: {e}")
                print(f"  Will use URL from last game_history entry: {final_url}")
        else:
            print(f"⚠ Last step file not found: {last_step_file}")
            print(f"  Will use URL from last game_history entry: {final_url}")

        # Update the last state's URL to be the final URL (after the last action)
        if states_data:
            states_data[-1]['url'] = final_url
            print(f"✓ Updated last state (step {len(states_data)-1}) URL to: {final_url}")

        # Save to states.json
        states_file = results_dir / "states.json"
        with open(states_file, 'w', encoding='utf-8') as f:
            json.dump(states_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(states_data)} states to {states_file}")

        # Save HTML content to txt files if --save_html is enabled
        if args.save_html:
            print(f"\n[INFO] Saving HTML content to txt files...")
            html_saved_count = 0
            # Iterate through all step files
            step_idx = 0
            while True:
                step_file = results_dir / f"step_{step_idx}.pkl.gz"
                if not step_file.exists():
                    break
                try:
                    with gzip.open(step_file, 'rb') as f:
                        step_data = pickle.load(f)
                        if hasattr(step_data, 'obs') and isinstance(step_data.obs, dict):
                            # Try to get HTML content (prefer pruned_html, fallback to dom_txt)
                            html_content = step_data.obs.get('pruned_html') or step_data.obs.get('dom_txt') or step_data.obs.get('html')
                            if html_content:
                                html_file = results_dir / f"html_step_{step_idx}.txt"
                                with open(html_file, 'w', encoding='utf-8') as hf:
                                    hf.write(html_content)
                                html_saved_count += 1
                except Exception as e:
                    print(f"  ⚠ Warning: Could not extract HTML from step_{step_idx}.pkl.gz: {e}")
                step_idx += 1

            if html_saved_count > 0:
                print(f"  ✓ Saved HTML for {html_saved_count} steps to html_step_*.txt files")
            else:
                print(f"  ⚠ No HTML content found in step files. Make sure to use --use_html true")

    # Evaluate the task success/failure
    agent = exp_args.agent

    try:
        # Load the task configuration
        task_id = task_name.split('.')[-1]
        config_path = os.path.join(args.config_dir, f"{task_id}.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Check the evaluation type
        eval_types = config.get('eval', {}).get('eval_types', [])

        # If contains program_html, use the Live Evaluation result
        if 'program_html' in eval_types:
            print(f"\n{'='*80}")
            print(f"Detected program_html evaluation type")
            print(f"Use Live Evaluation result (skip Post-hoc Evaluation)")
            print(f"{'='*80}\n")

            # Read the Live Evaluation result from summary_info.json
            summary_path = results_dir / "summary_info.json"
            with open(summary_path, 'r') as f:
                summary_info = json.load(f)

            # cum_reward from Live Evaluation
            live_reward = summary_info.get('cum_reward', 0.0)
            success = live_reward >= 1.0

            print(f"📊 Live Evaluation result:")
            print(f"   - cum_reward: {live_reward}")
            print(f"   - success: {success}")
            print(f"   - threshold: 1.0\n")

            analysis = f"Live evaluation reward: {live_reward}, success: {success}"

            # Save the evaluation result to autoeval.json file (for compatibility)
            result_dir = Path(args.result_dir) / task_name
            # The file name is named after the evaluation model (if not specified, it falls back to the generation model)
            safe_model_name = (args.llm_eval if args.llm_eval else args.llm_model).replace('/', '_')
            eval_file = result_dir / f"{safe_model_name}_autoeval.json"

            eval_info = [{
                "idx": task_id,
                "gt": live_reward,
                "rm": success,
                "thoughts": analysis,
                "uid": task_name,
            }]

            with open(eval_file, 'w') as f:
                json.dump(eval_info, f, indent=4)
            print(f"💾 Evaluation result saved to: {eval_file}\n")

        elif args.use_llm_success_eval:
            # Use LLM to directly evaluate the task success/failure (based on the trajectory and task goal)
            print(f"\n{'='*80}")
            print(f"Use LLM to evaluate the task success/failure")
            print(f"{'='*80}\n")

            task_goal = config.get('intent', 'Unknown task')

            # Call evaluate_step_scores_with_llm to get the LLM's success judgment
            # Note: Here we need to call end_episode first to trigger the evaluation, but don't pass success yet
            # Actually we need to have a complete game_history before calling evaluate_step_scores_with_llm
            from memory_agents.utils.utils import evaluate_step_scores_with_llm

            # Call the evaluation function to get the LLM judgment
            result = evaluate_step_scores_with_llm(
                game_history=agent.game_history,
                state=agent.game_history[-1].get('state', '') if agent.game_history else '',
                final_score=0,  # Temporarily not used
                success=False,  # This value will be overwritten by the LLM judgment
                llm_analysis='',
                llm_model=args.llm_eval if args.llm_eval else args.llm_model,
                temperature=0.3,
                task_goal=task_goal,
                evaluate_success=True  # Key: Enable LLM success evaluation
            )

            # Unpack the result
            if len(result) == 3:
                scores, reasonings, llm_success = result
            else:
                print(f"[Warning] evaluate_step_scores_with_llm returned unexpected format, using default failure")
                scores, reasonings = result
                llm_success = False

            success = llm_success
            analysis = f"LLM evaluated task as: {'SUCCESS' if success else 'FAILURE'}"

            print(f"\n✅ LLM evaluation completed: success = {success}\n")

            result_dir = Path(args.result_dir) / task_name
            safe_model_name = (args.llm_eval if args.llm_eval else args.llm_model).replace('/', '_')
            eval_file = result_dir / f"{safe_model_name}_autoeval.json"

            eval_info = [{
                "idx": task_id,
                "gt": None,  # LLM evaluation does not have ground truth
                "rm": success,
                "thoughts": analysis,
                "uid": task_name,
            }]

            with open(eval_file, 'w') as f:
                json.dump(eval_info, f, indent=4)
            print(f"💾 Save evaluation result to: {eval_file}\n")

        else:
            # Use the original evaluation method based on the config file ground truth (Post-hoc Evaluation)
            from autoeval.evaluate_trajectory import evaluate_trajectory_direct

            # Use llm_model as the saved file name, so that test_webarena_lite.py can find it
            # And use llm_eval (if specified) for actual evaluation
            llm_eval_model = args.llm_eval if args.llm_eval is not None else args.llm_model
            eval_prompt = "text"

            print(f"\n{'='*80}")
            print(f"Use config file Ground Truth to evaluate the task success/failure (Post-hoc Evaluation)")
            print(f"{'='*80}")
            print(f"   - result_dir: {args.result_dir}/{task_name}")
            print(f"   - llm_eval: {llm_eval_model}")
            print(f"   - prompt: {eval_prompt}")
            print(f"   - config_dir: {args.config_dir}\n")

            # Directly call the evaluation function
            eval_info, success = evaluate_trajectory_direct(
                result_dir=f"{args.result_dir}/{task_name}",
                llm_eval=llm_eval_model,  # Use the specified evaluation model name, the file name is the same
                prompt=eval_prompt,
                config_dir=args.config_dir
            )

            print(f"\n✅ Config file evaluation completed: success = {success}\n")

            # Extract the analysis information
            if isinstance(eval_info, list) and len(eval_info) > 0:
                analysis = eval_info[0].get('thoughts', 'No analysis available')
            else:
                analysis = eval_info.get('thoughts', 'No analysis available')

        # Notify the agent that the task has ended
        exp_args.agent.end_episode(
            state=agent.game_history[-1].get('state', '') if agent.game_history else '',
            score=10 if success else 0,
            success=success,
            llm_analysis=analysis,
            user_instruction=None
        )

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Output the URL and action summary for each step (using the normalized action)
    print("\n" + "="*80)
    print("Step-by-Step URL and Action Summary")
    print("="*80)

    agent = exp_args.agent
    from memory_agents.utils.utils import normalize_action

    for idx, entry in enumerate(agent.game_history):
        url = entry.get('url', 'N/A')
        raw_action = entry.get('action', 'N/A')
        state_text = entry.get('state', '')

        # Normalize the action
        try:
            # Keep the original message content of send_msg_to_user in the log
            normalized_data = normalize_action(raw_action, state_text, normalize_send_msg=False)
            action_display = normalized_data.get('normalized_action', raw_action)
        except:
            action_display = raw_action

        print(f"Step {idx}:")
        print(f"  URL: {url}")
        print(f"  Action: {action_display}")
        print()

    print("="*80 + "\n")

    return success, exp_args.agent


def main():
    args = parse_args()

    # Check if we're running multiple test cases
    if args.test_case_start is not None and args.test_case_end is not None:
        # Extract base task name (e.g., "webarena" from "webarena.0")
        if '.' in args.task_name:
            base_task_name = args.task_name.rsplit('.', 1)[0]
        else:
            base_task_name = args.task_name
        
        # Initialize statistics
        all_results = []
        cumulative_success = 0
        total_runs = 0
        evaluation_status = []  # For reflexion agent

        print(f"\n{'='*80}")
        print(f"Running test cases {args.test_case_start} to {args.test_case_end}, repeating {args.repeat_runs} times")
        print(f"{'='*80}\n")

        # Keep track of the agent across episodes for memory
        agent_dict = {}
        
        # Repeat runs
        for run_idx in range(args.repeat_runs):
            print(f"\n{'='*80}")
            print(f"Run {run_idx + 1}/{args.repeat_runs}")
            print(f"{'='*80}\n")
            
            run_results = []
            
            # Run each test case
            for test_case_idx in range(args.test_case_start, args.test_case_end + 1):
                task_name = f"{base_task_name}.{test_case_idx}"
                print(f"\n[Run {run_idx + 1}/{args.repeat_runs}] Test case {test_case_idx}: {task_name}")
                
                try:
                    if args.agent_type in ['evotest', 'reference']:
                        print(f"Reused agent: {agent_dict.get(task_name, None)}")
                        if agent_dict.get(task_name, None) is not None:
                            agent_dict.get(task_name, None).start_episode()
                        success, agent = run_single_episode(args, task_name, reused_agent=agent_dict.get(task_name, None))
                    elif args.agent_type in ['reflexion']:
                        print(f"Reused agent: {agent_dict.get(task_name, None)}")
                        if agent_dict.get(task_name, None) is not None:
                            agent_dict.get(task_name, None).start_episode(evaluation_status=evaluation_status)
                        success, agent = run_single_episode(args, task_name, reused_agent=agent_dict.get(task_name, None))
                    else:
                        print(f"New agent: {agent_dict.get(task_name, None)}")
                        success, agent = run_single_episode(args, task_name)

                    agent_dict[task_name] = agent
                    run_results.append({
                        'run': run_idx + 1,
                        'test_case': test_case_idx,
                        'task_name': task_name,
                        'success': success
                    })

                    cumulative_success += (1 if success else 0)
                    total_runs += 1
                    cumulative_accuracy = cumulative_success / total_runs

                    print(f"  Result: {'Success' if success else 'Failed'}")
                    print(f"  Cumulative Accuracy: {cumulative_accuracy:.2%} ({cumulative_success}/{total_runs})")

                except TokenLimitExceededError as e:
                    # Token limit exceeded - terminate immediately
                    print(f"\n{'='*80}")
                    print(f"  FATAL ERROR: Token limit exceeded!")
                    print(f"  Terminating run.py immediately...")
                    print(f"  Error details: {e}")
                    print(f"{'='*80}\n")
                    # Exit with specific code to signal token limit error
                    sys.exit(2)
                except Exception as e:
                    print(f"  Error: {e}")
                    run_results.append({
                        'run': run_idx + 1,
                        'test_case': test_case_idx,
                        'task_name': task_name,
                        'success': False,
                        'error': str(e)
                    })
                    total_runs += 1
            
            all_results.extend(run_results)
        
        # Calculate and display final statistics
        print(f"\n\n{'='*80}")
        print(f"Execution Complete - Final Statistics")
        print(f"{'='*80}")
        print(f"Test Case Range: {args.test_case_start} - {args.test_case_end}")
        print(f"Repeat Runs: {args.repeat_runs}")
        print(f"Total Runs: {total_runs}")
        print(f"Successful Runs: {cumulative_success}")
        print(f"Cumulative Accuracy: {cumulative_accuracy:.2%}")
        print(f"{'='*80}\n")
        
        # Calculate accuracy by test case
        print(f"{'='*80}")
        print(f"Accuracy by Test Case:")
        print(f"{'='*80}")
        
        test_case_stats = {}
        for result in all_results:
            test_case = result['test_case']
            if test_case not in test_case_stats:
                test_case_stats[test_case] = {'success': 0, 'total': 0}
            test_case_stats[test_case]['total'] += 1
            if result['success']:
                test_case_stats[test_case]['success'] += 1
        
        for test_case in sorted(test_case_stats.keys()):
            stats = test_case_stats[test_case]
            accuracy = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  Test Case {test_case}: {accuracy:.2%} ({stats['success']}/{stats['total']})")
        
        print(f"{'='*80}\n")
        
        # Save detailed results to JSON
        results_summary_path = f"{args.result_dir}/cumulative_results_{args.test_case_start}_{args.test_case_end}_x{args.repeat_runs}.json"
        
        # Prepare test case statistics for JSON
        test_case_accuracy = {
            str(tc): {
                'success': stats['success'],
                'total': stats['total'],
                'accuracy': stats['success'] / stats['total'] if stats['total'] > 0 else 0
            }
            for tc, stats in test_case_stats.items()
        }
        
        with open(results_summary_path, 'w') as f:
            json.dump({
                'test_case_range': [args.test_case_start, args.test_case_end],
                'repeat_runs': args.repeat_runs,
                'total_runs': total_runs,
                'cumulative_success': cumulative_success,
                'cumulative_accuracy': cumulative_accuracy,
                'test_case_accuracy': test_case_accuracy,
                'detailed_results': all_results
            }, f, indent=2)
        
        print(f"Detailed results saved to: {results_summary_path}\n")
        
    else:
        evaluation_status = []
        # Original single episode execution (with optional repeat_runs support)
        if args.repeat_runs > 1:
            # Repeat runs for single task
            print(f"\n{'='*80}")
            print(f"Running task {args.task_name}, repeating {args.repeat_runs} times")
            print(f"{'='*80}\n")
            
            # Keep track of the agent across episodes for memory (especially for evotest and reference)
            reused_agent = None
            all_results = []
            cumulative_success = 0
            total_runs = 0
            agent_dict = {}
            cumulative_accuracy = 0.0  # Initialize to avoid UnboundLocalError
            task_name = args.task_name  # Define task_name variable
            
            # Repeat runs
            for run_idx in range(args.repeat_runs):
                print(f"\n{'='*80}")
                print(f"Run {run_idx + 1}/{args.repeat_runs}")
                print(f"{'='*80}\n")
                
                try:
                    if args.agent_type in ['evotest', 'reference']:
                        print(f"Reused agent: {reused_agent is not None}")
                        if reused_agent is not None:
                            reused_agent.start_episode()
                        success, agent = run_single_episode(args, task_name, reused_agent=reused_agent)
                        reused_agent = agent  # Save agent for next run
                    elif args.agent_type in ['reflexion']:
                        print(f"Reused agent: {reused_agent is not None}")
                        if reused_agent is not None:
                            reused_agent.start_episode(evaluation_status=evaluation_status) # Reflexion utilized evaluation status from previous runs
                        success, agent = run_single_episode(args, task_name, reused_agent=reused_agent)
                        reused_agent = agent 
                    else:
                        print(f"New agent: {agent_dict.get(task_name, None)}")
                        success, agent = run_single_episode(args, task_name)
                    
                    all_results.append({
                        'run': run_idx + 1,
                        'task_name': args.task_name,
                        'success': success
                    })
                    
                    cumulative_success += (1 if success else 0)
                    total_runs += 1
                    cumulative_accuracy = cumulative_success / total_runs if total_runs > 0 else 0.0
                    
                    print(f"  Result: {'Success' if success else 'Failed'}")
                    print(f"  Cumulative Accuracy: {cumulative_accuracy:.2%} ({cumulative_success}/{total_runs})")
                    
                except TokenLimitExceededError as e:
                    # Token limit exceeded - terminate immediately
                    print(f"\n{'='*80}")
                    print(f"  FATAL ERROR: Token limit exceeded!")
                    print(f"  Terminating run.py immediately...")
                    print(f"  Error details: {e}")
                    print(f"{'='*80}\n")
                    # Exit with specific code to signal token limit error
                    sys.exit(2)
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        'run': run_idx + 1,
                        'task_name': args.task_name,
                        'success': False,
                        'error': str(e)
                    })
                    total_runs += 1
                    # Update cumulative_accuracy even on error
                    cumulative_accuracy = cumulative_success / total_runs if total_runs > 0 else 0.0
            
            # Print final statistics
            print(f"\n\n{'='*80}")
            print(f"Execution Complete - Final Statistics")
            print(f"{'='*80}")
            print(f"Task: {args.task_name}")
            print(f"Repeat Runs: {args.repeat_runs}")
            print(f"Total Runs: {total_runs}")
            print(f"Successful Runs: {cumulative_success}")
            print(f"Cumulative Accuracy: {cumulative_accuracy:.2%}")
            print(f"{'='*80}\n")
            
            # Save detailed results to JSON
            results_summary_path = f"{args.result_dir}/cumulative_results_{args.task_name.replace('.', '_')}_x{args.repeat_runs}.json"
            
            with open(results_summary_path, 'w') as f:
                json.dump({
                    'task_name': args.task_name,
                    'repeat_runs': args.repeat_runs,
                    'total_runs': total_runs,
                    'cumulative_success': cumulative_success,
                    'cumulative_accuracy': cumulative_accuracy,
                    'detailed_results': all_results
                }, f, indent=2)
            
            print(f"Detailed results saved to: {results_summary_path}\n")
        else:
            # Single run (original behavior)
            try:
                success, agent = run_single_episode(args, args.task_name, reused_agent=None)
                print(f"\nTask {args.task_name} {'succeeded' if success else 'failed'}")
            except TokenLimitExceededError as e:
                print(f"\n{'='*80}")
                print(f"FATAL ERROR: Token limit exceeded!")
                print(f"Terminating run.py immediately...")
                print(f"Error details: {e}")
                print(f"{'='*80}\n")
                # Exit with specific code to signal token limit error
                sys.exit(2)


if __name__ == "__main__":
    main()
