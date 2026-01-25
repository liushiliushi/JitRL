#!/usr/bin/env python
"""
Test WebArena-Lite tasks (165 tasks) with parallel execution support and early stopping.

Features:
- Early stopping: When a task succeeds once, remaining repeats are automatically skipped
- Parallel execution: Run multiple tasks simultaneously for faster completion
- Comprehensive logging: Detailed results saved to JSON and summary files

Usage:
    # Test all WebArena-Lite tasks once (parallel, 4 workers)
    python test_webarena_lite.py --model google/gemini-2.5-flash-preview-09-2025 --workers 4

    # Test tasks 0-10, running each up to 3 times with early stopping (parallel, 8 workers)
    # Each task stops after first success, saving time
    python test_webarena_lite.py --start 0 --end 10 --repeat 3 --workers 8 --model google/gemini-2.5-flash-preview-09-2025

    # Test specific tasks (serial execution)
    python test_webarena_lite.py --tasks 0,1,2,25 --workers 1 --model google/gemini-2.5-flash-preview-09-2025
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import logging

# Suppress browsergym warnings and info logs
logging.getLogger('browsergym.core.env').setLevel(logging.ERROR)
logging.getLogger('browsergym.experiments.loop').setLevel(logging.WARNING)

# Load WebArena environment variables
def load_webarena_env():
    """Load WebArena environment variables from env_setup.txt"""
    from pathlib import Path
    env_file = Path(__file__).parent / "env_setup.txt"
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
                    if '$BASE_URL' in value and 'BASE_URL' in os.environ:
                        value = value.replace('$BASE_URL', os.environ['BASE_URL'])
                    os.environ[key] = value

# Load environment at module level
load_webarena_env()




def get_available_tasks(config_dir=None, sites_filter=None):
    """Get list of available task IDs from webarena package or local config files.

    Args:
        config_dir: If None (default), reads from webarena package. Otherwise reads from local directory.
        sites_filter: If provided, only return tasks that include this site (e.g., 'shopping_admin', 'map', 'wikipedia').
                     Special value 'mixed' returns tasks that involve 2 or more sites.

    Returns:
        list: List of task IDs that match the criteria
    """
    if config_dir is None:
        # Try to read from webarena package first, fall back to local config_files
        try:
            import importlib.resources
            import webarena

            all_configs_str = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
            all_configs = json.loads(all_configs_str)

            # Filter by sites if specified
            if sites_filter:
                filtered_configs = []
                for conf in all_configs:
                    if 'task_id' not in conf:
                        continue
                    sites = conf.get('sites', [])
                    # Special case: 'mixed' means tasks with 2+ sites
                    if sites_filter == 'mixed':
                        if len(sites) >= 2:
                            filtered_configs.append(conf)
                    # Include if sites contains the specified site (changed from exact match)
                    elif sites_filter in sites:
                        filtered_configs.append(conf)
                task_ids = sorted(set(conf.get('task_id') for conf in filtered_configs))
            else:
                task_ids = sorted(set(conf.get('task_id') for conf in all_configs if 'task_id' in conf))

            return task_ids
        except (ImportError, Exception) as e:
            # Fall back to local config files
            print(f"Warning: webarena package not available, using local config_files directory")
            config_dir = "config_files"

    # Read from local config directory (either explicitly specified or fallback from above)
    config_path = Path(config_dir)
    if not config_path.exists():
        print(f"Error: Config directory {config_dir} not found")
        return []

    task_ids = []
    for json_file in config_path.glob("*.json"):
        try:
            task_id = int(json_file.stem)

            # Filter by sites if specified
            if sites_filter:
                with open(json_file, 'r') as f:
                    conf = json.load(f)
                    sites = conf.get('sites', [])
                    # Special case: 'mixed' means tasks with 2+ sites
                    if sites_filter == 'mixed':
                        if len(sites) >= 2:
                            task_ids.append(task_id)
                    # Include if sites contains the specified site (changed from exact match)
                    elif sites_filter in sites:
                        task_ids.append(task_id)
            else:
                task_ids.append(task_id)
        except (ValueError, json.JSONDecodeError):
            continue

    return sorted(task_ids)


def get_task_metadata(task_ids):
    """Get metadata for tasks including intent_template for grouping.

    Args:
        task_ids: List of task IDs to get metadata for

    Returns:
        dict: Mapping from task_id to metadata dict with 'intent_template' and 'sites'
    """
    # Try to read from webarena package first, fall back to local config_files
    try:
        import importlib.resources
        import webarena

        all_configs_str = importlib.resources.files(webarena).joinpath("test.raw.json").read_text()
        all_configs = json.loads(all_configs_str)

        metadata = {}
        for conf in all_configs:
            tid = conf.get('task_id')
            if tid in task_ids:
                metadata[tid] = {
                    'intent_template': conf.get('intent_template', ''),
                    'sites': conf.get('sites', []),
                    'intent': conf.get('intent', '')
                }

        return metadata
    except (ImportError, Exception) as e:
        # Fall back to local config files
        print(f"Warning: webarena package not available, reading from config_files")
        metadata = {}
        config_path = Path("config_files")

        for task_id in task_ids:
            config_file = config_path / f"{task_id}.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    conf = json.load(f)
                    metadata[task_id] = {
                        'intent_template': conf.get('intent_template', ''),
                        'sites': conf.get('sites', []),
                        'intent': conf.get('intent', '')
                    }
            else:
                metadata[task_id] = {'intent_template': '', 'sites': [], 'intent': ''}

        return metadata


def group_tasks_by_intent_template(task_ids, metadata):
    """Group tasks by site and consecutive task IDs for scheduling.

    Tasks are first grouped by site, then within each site they are grouped into
    consecutive ranges (e.g., 0-6, 11-15, 41-43).

    Tasks within the same group must run serially.
    Tasks in different groups can run in parallel.

    Args:
        task_ids: List of task IDs
        metadata: Task metadata from get_task_metadata()

    Returns:
        list: List of groups, where each group is a list of consecutive task IDs from the same site
    """
    from collections import defaultdict

    # Group tasks by site (using the first site if multiple)
    site_tasks = defaultdict(list)
    for tid in sorted(task_ids):
        sites = metadata.get(tid, {}).get('sites', [])
        site = sites[0] if sites else 'unknown'
        site_tasks[site].append(tid)

    # For each site, group consecutive task IDs
    groups = []
    for site in sorted(site_tasks.keys()):
        task_list = sorted(site_tasks[site])

        # Group consecutive task IDs
        if not task_list:
            continue

        current_group = [task_list[0]]
        for i in range(1, len(task_list)):
            if task_list[i] == task_list[i-1] + 1:
                # Consecutive, add to current group
                current_group.append(task_list[i])
            else:
                # Not consecutive, start a new group
                groups.append(current_group)
                current_group = [task_list[i]]

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

    return groups


def run_task(task_id, model, max_steps=15, log_file=None, llm_eval=None, llm_extract=None,
             use_screenshot_action=False, use_screenshot_eval=False, disable_memory=False,
             use_llm_success_eval=False, run_id=None, agent_type='memory', task_timeout=600,
             task_similarity_threshold=0.27, repeat_runs=1, logit_mode='verbalized',
             result_dir='results', save_html=False, no_save_memory=False):
    """Run a single task and return success status.

    Args:
        task_timeout: Maximum time (in seconds) for a task to complete. Default: 600s (10 minutes)
                     If task exceeds this time, it will be killed and restarted.
        repeat_runs: Number of times to repeat the task. If > 1, run.py will handle
                    all repeats internally, allowing agent reuse (especially for evotest).
                    Returns a list of results for all runs.
    """
    # For WebArena-Lite, we need to use webarena_lite prefix
    # But run.py might not support this yet, so we'll use webarena.{task_id}
    # and manually specify the config file path

    # Use standard task name - BrowserGym only accepts "webarena.{task_id}" format
    # run_id is only used for logging, not for the environment ID
    task_name = f"webarena.{task_id}"

    # Note: Task config is loaded from BrowserGym's webarena package (via switch_config.sh)
    # No need to check local config files - the package handles this

    cmd = [
        sys.executable, "run.py",
        "--task_name", task_name,
        "--llm_model", model,
        "--max_steps", str(max_steps)
    ]

    # Add agent_type (memory or evotest)
    if agent_type:
        cmd.extend(["--agent_type", agent_type])

    # Add repeat_runs if specified (allows agent reuse in run.py)
    if repeat_runs > 1:
        cmd.extend(["--repeat_runs", str(repeat_runs)])

    # Add llm_eval parameter if specified
    if llm_eval is not None:
        cmd.extend(["--llm_eval", llm_eval])

    # Add llm_extract parameter if specified
    if llm_extract is not None:
        cmd.extend(["--llm_extract", llm_extract])

    # Add screenshot parameters if specified
    if use_screenshot_action:
        cmd.extend(["--use_screenshot_action", "true"])
    if use_screenshot_eval:
        cmd.extend(["--use_screenshot_eval", "true"])

    # Add disable_memory parameter if specified
    if disable_memory:
        cmd.extend(["--disable_memory", "true"])

    # Add no-save_memory parameter if specified
    if no_save_memory:
        cmd.append("--no-save_memory")

    # Add task_similarity_threshold parameter
    cmd.extend(["--task_similarity_threshold", str(task_similarity_threshold)])

    # Add use_llm_success_eval parameter if specified
    if use_llm_success_eval:
        cmd.extend(["--use_llm_success_eval", "true"])

    # Add logit_mode parameter
    cmd.extend(["--logit_mode", logit_mode])

    # Add result_dir parameter
    cmd.extend(["--result_dir", result_dir])

    # Add save_html parameter if specified
    if save_html:
        cmd.extend(["--save_html", "true"])
        # Also enable use_html to ensure HTML content is captured
        cmd.extend(["--use_html", "true"])

    # Run the command with retry logic for errors and timeout detection
    max_retries = 2  # Maximum number of retries (run once, retry up to 2 times if failed/timeout)
    retry_count = 0
    retry_reason = None

    # Calculate effective timeout: task_timeout per run * number of runs
    effective_timeout = task_timeout * repeat_runs if repeat_runs > 1 else task_timeout

    while retry_count <= max_retries:
        if log_file:
            with open(log_file, 'a' if retry_count > 0 else 'w') as f:
                # Write a brief header including intent before streaming subprocess output
                if retry_count == 0:
                    try:
                        with open(log_file, 'r', encoding='utf-8') as cf:
                            cfg = json.load(cf)
                        intent_text = cfg.get('intent', '')
                    except Exception:
                        intent_text = ''

                    f.write("=" * 80 + "\n")
                    f.write(f"Task: {task_name}\n")
                    f.write(f"Model: {model}\n")
                    f.write(f"Eval Model: {llm_eval if llm_eval else model}\n")
                    f.write(f"Max Steps: {max_steps}\n")
                    f.write(f"Timeout: {task_timeout}s per run, {effective_timeout}s total ({repeat_runs} runs)\n")
                    if intent_text:
                        f.write(f"Intent: {intent_text}\n")
                    f.write("=" * 80 + "\n\n")
                else:
                    f.write("\n" + "=" * 80 + "\n")
                    f.write(f"RETRY ATTEMPT {retry_count}/{max_retries} ({retry_reason})\n")
                    f.write("=" * 80 + "\n\n")
                f.flush()
                try:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=effective_timeout)
                except subprocess.TimeoutExpired:
                    f.write(f"\n\n{'='*80}\n")
                    f.write(f"TIMEOUT: Task exceeded {effective_timeout}s limit ({task_timeout}s x {repeat_runs} runs)\n")
                    f.write(f"{'='*80}\n")
                    f.flush()
                    # Create a fake result object to trigger retry
                    result = subprocess.CompletedProcess(cmd, returncode=124)  # 124 is typical timeout exit code
        else:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                # Create a fake result object to trigger retry
                result = subprocess.CompletedProcess(cmd, returncode=124)

        # Check evaluation result from results directory
        result_dir_path = Path(result_dir) / task_name
        safe_model_name = (llm_eval if llm_eval else model).replace('/', '_')
        eval_file = result_dir_path / f"{safe_model_name}_autoeval.json"

        # Determine if we need to retry
        should_retry = False
        retry_reason = None

        # Check for timeout (exit code 124)
        if result.returncode == 124:
            should_retry = True
            retry_reason = f"Task timeout (exceeded {effective_timeout}s for {repeat_runs} runs)"
        # Check for token limit error (exit code 2)
        elif result.returncode == 2:
            should_retry = True
            retry_reason = "Token limit exceeded"
        # Check for other non-zero exit codes (process crashed or errored)
        elif result.returncode != 0:
            should_retry = True
            retry_reason = f"Process exited with code {result.returncode}"
        # Check if evaluation file exists
        elif not eval_file.exists():
            should_retry = True
            retry_reason = "No evaluation file generated (possible crash)"

        # If we should retry and haven't exhausted retries
        if should_retry and retry_count < max_retries:
            retry_count += 1
            print(f"\n[WARNING] {retry_reason} for {task_name}. Retrying ({retry_count}/{max_retries})...")
            time.sleep(2)  # Brief pause before retry
            continue
        else:
            # Either succeeded or exhausted retries
            break

    # Final check of evaluation result (eval_file is already set from the loop above)
    retry_suffix = f" (after {retry_count} {'retry' if retry_count == 1 else 'retries'})" if retry_count > 0 else ""

    # Check evaluation result from results directory (more reliable than shared autoeval/log)
    result_dir_path = Path(result_dir) / task_name

    # If repeat_runs > 1, run.py generates a cumulative results JSON file
    if repeat_runs > 1:
        # Check for cumulative results file generated by run.py
        cumulative_results_file = result_dir_path.parent / f"cumulative_results_{task_name.replace('.', '_')}_x{repeat_runs}.json"

        if cumulative_results_file.exists():
            try:
                with open(cumulative_results_file, 'r') as f:
                    cumulative_data = json.load(f)
                    detailed_results = cumulative_data.get('detailed_results', [])

                    # Return list of results for all runs
                    all_results = []
                    for result in detailed_results:
                        run_num = result.get('run', 0)
                        success = result.get('success', False)
                        status = 'success' if success else 'failure'
                        error = result.get('error', '')
                        details = error if error else f"Run {run_num}: {'Success' if success else 'Failure'}"
                        all_results.append((success, status, details))

                    # Return the first result for backward compatibility
                    # But also return all_results as a special marker
                    if all_results:
                        # For backward compatibility, return first result
                        # The caller should handle the list of results
                        return all_results, 'multiple_runs', cumulative_data
                    else:
                        return False, 'no_results', f'No results in cumulative file{retry_suffix}'
            except Exception as e:
                return False, 'error', f'{str(e)}{retry_suffix}'
        else:
            # If repeat_runs > 1 but cumulative file not found, return error for all runs
            # This indicates run.py didn't generate the expected results file
            return [(False, 'no_cumulative_file', f'Cumulative results file not found for {repeat_runs} runs{retry_suffix}')] * repeat_runs, 'multiple_runs', None

    # Standard single-run evaluation
    # The result is saved as {model_name}_autoeval.json in the results directory
    # Replace '/' with '_' in model name for safe filename (same as in evaluate_trajectory.py)
    safe_model_name = (llm_eval if llm_eval else model).replace('/', '_')
    eval_file = result_dir_path / f"{safe_model_name}_autoeval.json"

    if eval_file.exists():
        try:
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
                # eval_data is a list with one element
                if isinstance(eval_data, list) and len(eval_data) > 0:
                    result_info = eval_data[0]
                    success = result_info.get('rm', False)
                    thoughts = result_info.get('thoughts', '')
                    status = 'success' if success else 'failure'
                    return success, status, thoughts
                else:
                    return False, 'invalid_eval', f'Invalid evaluation data format{retry_suffix}'
        except Exception as e:
            return False, 'error', f'{str(e)}{retry_suffix}'
    else:
        # If we exhausted retries with this specific error
        if retry_reason and retry_count >= max_retries:
            return False, 'no_eval', f'{retry_reason}{retry_suffix}'
        else:
            return False, 'no_eval', f'Evaluation file not found: {eval_file}{retry_suffix}'


def run_task_wrapper(args_dict):
    """Wrapper function for parallel execution.

    Args:
        args_dict: Dictionary containing task parameters and success tracking

    Returns:
        List of dictionaries with task results (one per repeat run)
    """
    task_id = args_dict['task_id']
    repeat_idx = args_dict.get('repeat_idx', 0)
    model = args_dict['model']
    max_steps = args_dict['max_steps']
    log_file = args_dict['log_file']
    llm_eval = args_dict['llm_eval']
    llm_extract = args_dict.get('llm_extract')
    use_screenshot_action = args_dict.get('use_screenshot_action', False)
    use_screenshot_eval = args_dict.get('use_screenshot_eval', False)
    disable_memory = args_dict.get('disable_memory', False)
    use_llm_success_eval = args_dict.get('use_llm_success_eval', False)
    task_success_flags = args_dict.get('task_success_flags', {})
    success_lock = args_dict.get('success_lock')
    no_early_stop = args_dict.get('no_early_stop', False)
    agent_type = args_dict.get('agent_type', 'memory')
    task_timeout = args_dict.get('task_timeout', 600)
    task_similarity_threshold = args_dict.get('task_similarity_threshold', 0.27)
    repeat_runs = args_dict.get('repeat_runs', 1)
    logit_mode = args_dict.get('logit_mode', 'verbalized')
    result_dir = args_dict.get('result_dir', 'results')
    save_html = args_dict.get('save_html', False)
    no_save_memory = args_dict.get('no_save_memory', False)

    # Check if this task has already succeeded (thread-safe)
    # Only skip if early stopping is enabled (no_early_stop=False)
    if not no_early_stop and success_lock:
        with success_lock:
            if task_success_flags.get(task_id, False):
                # Return skipped results for all repeats
                return [{
                    'task_id': task_id,
                    'repeat_idx': i,
                    'success': False,
                    'status': 'skipped',
                    'details': 'Task already succeeded in previous run',
                    'duration': 0,
                    'skipped': True
                } for i in range(repeat_runs)]

    task_start = time.time()
    result = run_task(task_id, model, max_steps, log_file, llm_eval, llm_extract,
                     use_screenshot_action, use_screenshot_eval, disable_memory,
                     use_llm_success_eval, repeat_idx + 1, agent_type, task_timeout,
                     task_similarity_threshold, repeat_runs=repeat_runs, logit_mode=logit_mode,
                     result_dir=result_dir, save_html=save_html, no_save_memory=no_save_memory)
    task_duration = time.time() - task_start

    # Handle multiple results from run.py (when repeat_runs > 1)
    if repeat_runs > 1:
        # result is a tuple: (all_results, status, cumulative_data)
        if isinstance(result, tuple) and len(result) == 3:
            all_results, status, cumulative_data = result
            if status == 'multiple_runs' and isinstance(all_results, list):
                # Convert to list of result dictionaries
                result_list = []
                for i, (success, run_status, details) in enumerate(all_results):
                    # Update success flag if task succeeded (thread-safe)
                    # Only update if early stopping is enabled (no_early_stop=False)
                    if not no_early_stop and success and success_lock:
                        with success_lock:
                            task_success_flags[task_id] = True
                    
                    result_list.append({
                        'task_id': task_id,
                        'repeat_idx': i,
                        'success': success,
                        'status': run_status,
                        'details': details,
                        'duration': task_duration / repeat_runs,  # Approximate per-run duration
                        'skipped': False
                    })
                return result_list
            else:
                # Fallback: single result
                success, status, details = result[0] if isinstance(result[0], tuple) else (False, 'error', 'Invalid result format')
                if not no_early_stop and success and success_lock:
                    with success_lock:
                        task_success_flags[task_id] = True
                return [{
                    'task_id': task_id,
                    'repeat_idx': 0,
                    'success': success,
                    'status': status,
                    'details': details,
                    'duration': task_duration,
                    'skipped': False
                }]
        else:
            # Unexpected format, treat as single result
            if isinstance(result, tuple) and len(result) >= 2:
                success, status, details = result[0], result[1], result[2] if len(result) > 2 else ''
            else:
                success, status, details = False, 'error', 'Unexpected result format'
            
            if not no_early_stop and success and success_lock:
                with success_lock:
                    task_success_flags[task_id] = True
            
            return [{
                'task_id': task_id,
                'repeat_idx': 0,
                'success': success,
                'status': status,
                'details': details,
                'duration': task_duration,
                'skipped': False
            }]
    else:
        # Single run (original behavior)
        if isinstance(result, tuple) and len(result) >= 2:
            success, status, details = result[0], result[1], result[2] if len(result) > 2 else ''
        else:
            success, status, details = False, 'error', 'Unexpected result format'
        
        # Update success flag if task succeeded (thread-safe)
        # Only update if early stopping is enabled (no_early_stop=False)
        if not no_early_stop and success and success_lock:
            with success_lock:
                task_success_flags[task_id] = True
        
        return [{
            'task_id': task_id,
            'repeat_idx': repeat_idx,
            'success': success,
            'status': status,
            'details': details,
            'duration': task_duration,
            'skipped': False
        }]


def run_group_serially(group_args):
    """Run a group of tasks serially (within the same group).

    This function is at module level to be pickle-able for multiprocessing.

    Args:
        group_args: List of task argument dictionaries

    Returns:
        List of task results (flattened - each repeat run is a separate result)
    """
    group_results = []
    for task_args in group_args:
        results = run_task_wrapper(task_args)
        # run_task_wrapper now returns a list of results (one per repeat)
        # Flatten them into the group_results list
        if isinstance(results, list):
            group_results.extend(results)
        else:
            # Backward compatibility: single result
            group_results.append(results)
    return group_results


def get_completed_tasks_from_log_dir(log_dir):
    """Get completed tasks from a previous run's log directory.

    Args:
        log_dir: Path to log directory containing results_incremental.json

    Returns:
        dict: Mapping from task_id to set of completed run numbers
              e.g., {27: {1, 2, 3, 4, 5}, 28: {1, 2}}
    """
    from pathlib import Path
    import json

    log_path = Path(log_dir)
    results_file = log_path / "results_incremental.json"

    if not results_file.exists():
        print(f"Warning: No results_incremental.json found in {log_dir}")
        return {}

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        completed_runs = {}
        for result in data.get('results', []):
            task_id = result['task_id']
            run_num = result['repeat']

            if task_id not in completed_runs:
                completed_runs[task_id] = set()
            completed_runs[task_id].add(run_num)

        return completed_runs
    except Exception as e:
        print(f"Error reading results from {log_dir}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Test WebArena-Lite tasks (165 tasks) with parallel execution")
    parser.add_argument('--model', type=str,
                       default='google/gemini-2.5-flash-preview-09-2025',
                       help='LLM model to use. Can be API model name (e.g., "gpt-4o", "claude-3-opus") or local model path (e.g., "/ndata/yibo/webrl-llama-3.1-8b")')
    parser.add_argument('--llm_eval', type=str, default=None,
                       help='LLM model for evaluation (if not specified, uses --model)')
    parser.add_argument('--llm_extract', type=str, default=None,
                       help='LLM model for extracting effective action sequences (if not specified, uses --model)')
    parser.add_argument('--start', type=int, help='Start task ID')
    parser.add_argument('--end', type=int, help='End task ID')
    parser.add_argument('--tasks', type=str, help='Comma-separated task IDs (e.g., 0,1,2,25)')
    parser.add_argument('--all', action='store_true', help='Test all available tasks')
    parser.add_argument('--max_steps', type=int, default=25, help='Max steps per task')
    parser.add_argument('--log_dir', type=str, help='Directory for logs')
    parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat each task (default: 1)')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers (default: 1 for serial execution, recommended: 4-8)')
    parser.add_argument('--use_screenshot_action', action='store_true',
                       help='Use screenshots when generating actions (requires vision-capable model)')
    parser.add_argument('--use_screenshot_eval', action='store_true',
                       help='Use screenshots for visual evaluation of steps (requires vision-capable model)')
    parser.add_argument('--disable_memory', action='store_true',
                       help='Disable memory functionality (no retrieval, storage, or step evaluation). Agent relies purely on model capabilities.')
    parser.add_argument('--no-save_memory', action='store_true', dest='no_save_memory',
                       help='Disable saving memory at end of episode. Memory retrieval still works, but no new memories are saved.')
    parser.add_argument('--task_similarity_threshold', type=float, default=0.27,
                       help='Task similarity threshold for memory retrieval. Higher values require more similar tasks to share memories. Default: 0.27')
    parser.add_argument('--use_llm_success_eval', action='store_true',
                       help='Use LLM to evaluate task success instead of config-based ground truth evaluation.')
    parser.add_argument('--no_early_stop', action='store_true',
                       help='Disable early stopping. By default, when a task succeeds once, remaining repeats are skipped. Use this flag to run all repeats regardless of success.')
    parser.add_argument('--sites', type=str, default=None,
                       help='Filter tasks by site (e.g., shopping_admin, map, reddit, gitlab, wikipedia). Runs all tasks that include this site. Use "mixed" to run tasks involving 2+ sites.')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Resume from a previous run by specifying the log directory. Tasks that have completed evaluation in results_incremental.json will be skipped.')
    parser.add_argument('--agent_type', type=str, default='memory',
                       choices=['memory', 'evotest', 'reference', 'reflexion','prompt_memory'],
                       help='Agent implementation to use in run.py (memory, evotest, reference, or reflexion). Default: memory.')
    parser.add_argument('--task_timeout', type=int, default=1200,
                       help='Maximum time (in seconds) for a single task to complete. Tasks exceeding this time will be killed and retried. Default: 1200s (20 minutes)')
    parser.add_argument('--logit_mode', type=str, default='verbalized',
                       choices=['token', 'verbalized'],
                       help='Method to extract action probabilities: "token" uses token logits (GPT only), "verbalized" uses model-provided confidence scores (all models). Default: verbalized')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='Directory to save experiment results. Default: "results"')
    parser.add_argument('--save_html', action='store_true',
                       help='Save HTML content of each step to txt files. Requires --use_html to be effective.')
    args = parser.parse_args()

    # Auto-determine whether to use separate logs based on agent type
    # Agents that need to reuse state across runs (evotest, reflexion) use combined logs
    # Agents that don't need reuse (memory, reference, prompt_memory) use separate logs
    args.separate_logs = args.agent_type in ['memory', 'reference', 'prompt_memory']

    # Determine which tasks to run
    if args.tasks:
        task_ids = [int(x.strip()) for x in args.tasks.split(',')]
    elif args.all:
        task_ids = get_available_tasks(sites_filter=args.sites)
    elif args.start is not None and args.end is not None:
        task_ids = list(range(args.start, args.end + 1))
    else:
        # Default: run all 165 tasks
        task_ids = list(range(0, 165))

    # Filter to only existing config files (and by sites if specified)
    available_tasks = get_available_tasks(sites_filter=args.sites)
    task_ids = [tid for tid in task_ids if tid in available_tasks]

    if not task_ids:
        print("No valid tasks to run")
        return

    # Handle resume functionality
    completed_runs = {}
    if args.resume_from:
        print(f"Resume mode: Reading completed tasks from {args.resume_from}")
        completed_runs = get_completed_tasks_from_log_dir(args.resume_from)

        if completed_runs:
            total_completed = sum(len(runs) for runs in completed_runs.values())
            print(f"Found {len(completed_runs)} tasks with {total_completed} completed runs")
            print(f"These will be skipped during execution")
        else:
            print("No completed tasks found, will run all tasks")

    # Create log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    elif args.resume_from:
        # If resuming, use the same log directory
        log_dir = Path(args.resume_from)
        print(f"Resuming in the same log directory: {log_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Format: webarena_[site name]_[model name]_[agent name]_[time]
        site_name = args.sites if args.sites else "all"
        # Sanitize model name for folder (replace / with _)
        model_name_safe = args.model.replace('/', '_').replace(':', '_')
        log_dir = Path(f"logs/webarena_{site_name}_{model_name_safe}_{args.agent_type}_{timestamp}")

    log_dir.mkdir(parents=True, exist_ok=True)

    # Get task metadata and group by site and consecutive IDs
    print("Analyzing task dependencies...")
    metadata = get_task_metadata(task_ids)
    task_groups = group_tasks_by_intent_template(task_ids, metadata)

    # Print header
    print("=" * 80)
    print("WebArena-Lite Task Testing with Smart Scheduling")
    print("=" * 80)
    print(f"Generation Model: {args.model}")
    print(f"Evaluation Model: {args.llm_eval if args.llm_eval else args.model}")
    print(f"Agent Type: {args.agent_type}")
    print(f"Logit Mode: {args.logit_mode}")
    if args.sites:
        print(f"Filtered by site: {args.sites}")
    print(f"Tasks: {len(task_ids)} tasks ({min(task_ids)} to {max(task_ids)})")
    print(f"Task groups (by site + consecutive IDs): {len(task_groups)} groups")
    print(f"  - Tasks within same group run SERIALLY")
    print(f"  - Tasks from different groups run in PARALLEL")
    print(f"Repeats per task: {args.repeat}")
    print(f"Total runs: {len(task_ids) * args.repeat}")
    print(f"Max steps: {args.max_steps}")
    effective_total_timeout = args.task_timeout * args.repeat
    print(f"Task timeout: {args.task_timeout}s per run, {effective_total_timeout}s total for {args.repeat} runs (auto-restart on timeout)")
    print(f"Parallel workers: {args.workers}")
    print(f"Early stopping: {'Disabled' if args.no_early_stop else 'Enabled'}")
    if args.separate_logs:
        print(f"Log mode: Separate (task_X_run_Y.log) - agent type '{args.agent_type}' doesn't need reuse")
    else:
        print(f"Log mode: Combined (task_X_all_runs.log) - agent type '{args.agent_type}' reuses state across runs")
    print(f"Log directory: {log_dir}")
    print("=" * 80)

    # Print all tasks that will be executed
    print("\n" + "=" * 80)
    print("TASKS TO BE EXECUTED:")
    print("=" * 80)
    print(f"Total tasks: {len(task_ids)}")
    print(f"Task IDs: {sorted(task_ids)}")
    print()

    # Show task grouping details
    if len(task_groups) <= 30:  # Only show details if not too many groups
        print("\nTask Grouping:")
        for i, group in enumerate(task_groups, 1):
            site = metadata[group[0]]['sites'][0] if group and metadata[group[0]].get('sites') else 'unknown'
            task_range = f"{group[0]}-{group[-1]}" if len(group) > 1 else str(group[0])
            print(f"  Group {i} ({len(group)} tasks, Site: {site}): Tasks {task_range}")
    else:
        print(f"\nTask Grouping: {len(task_groups)} groups (too many to display)")
        # Still show summary by site
        from collections import defaultdict
        site_summary = defaultdict(list)
        for group in task_groups:
            site = metadata[group[0]]['sites'][0] if group and metadata[group[0]].get('sites') else 'unknown'
            site_summary[site].extend(group)

        print("\nTask Summary by Site:")
        for site in sorted(site_summary.keys()):
            tasks = sorted(site_summary[site])
            print(f"  {site}: {len(tasks)} tasks - {tasks}")
    print()

    # Track which tasks have succeeded (thread-safe)
    task_success_flags = {task_id: False for task_id in task_ids}
    success_lock = threading.Lock() if args.workers == 1 else None

    # Prepare all task arguments, organized by groups
    # Within each group, tasks run serially. Across groups, tasks run in parallel.
    all_task_args_by_group = []
    skipped_count = 0

    for group in task_groups:
        group_args = []
        for task_id in group:
            # Check if this task was already completed (when resuming)
            # For repeat_runs, we check if all runs are completed
            if completed_runs and task_id in completed_runs:
                completed_run_nums = completed_runs[task_id]
                if len(completed_run_nums) >= args.repeat:
                    # All runs completed, skip this task
                    skipped_count += args.repeat
                    continue

            if args.separate_logs:
                # Create separate task entries for each run (separate log files)
                for run_idx in range(args.repeat):
                    log_file = log_dir / f"task_{task_id}_run_{run_idx + 1}.log"
                    group_args.append({
                        'task_id': task_id,
                        'repeat_idx': run_idx,
                        'model': args.model,
                        'max_steps': args.max_steps,
                        'log_file': log_file,
                        'llm_eval': args.llm_eval,
                        'llm_extract': args.llm_extract,
                        'use_screenshot_action': args.use_screenshot_action,
                        'use_screenshot_eval': args.use_screenshot_eval,
                        'disable_memory': args.disable_memory,
                        'use_llm_success_eval': args.use_llm_success_eval,
                        'agent_type': args.agent_type,
                        'no_early_stop': args.no_early_stop,
                        'task_timeout': args.task_timeout,
                        'task_similarity_threshold': args.task_similarity_threshold,
                        'repeat_runs': 1,  # Single run per entry
                        'logit_mode': args.logit_mode,
                        'result_dir': args.result_dir,
                        'save_html': args.save_html,
                        'no_save_memory': args.no_save_memory,
                        'task_success_flags': {} if args.workers > 1 else task_success_flags,
                        'success_lock': None if args.workers > 1 else success_lock
                    })
            else:
                # Create a single task entry with repeat_runs parameter
                # run.py will handle all repeats internally
                log_file = log_dir / f"task_{task_id}_all_runs.log"
                group_args.append({
                    'task_id': task_id,
                    'repeat_idx': 0,  # Not used when repeat_runs > 1, but kept for compatibility
                    'model': args.model,
                    'max_steps': args.max_steps,
                    'log_file': log_file,
                    'llm_eval': args.llm_eval,
                    'llm_extract': args.llm_extract,
                    'use_screenshot_action': args.use_screenshot_action,
                    'use_screenshot_eval': args.use_screenshot_eval,
                    'disable_memory': args.disable_memory,
                    'use_llm_success_eval': args.use_llm_success_eval,
                    'agent_type': args.agent_type,
                    'no_early_stop': args.no_early_stop,
                    'task_timeout': args.task_timeout,
                    'task_similarity_threshold': args.task_similarity_threshold,
                    'repeat_runs': args.repeat,
                    'logit_mode': args.logit_mode,
                    'result_dir': args.result_dir,
                    'save_html': args.save_html,
                    'no_save_memory': args.no_save_memory,
                    'task_success_flags': {} if args.workers > 1 else task_success_flags,
                    'success_lock': None if args.workers > 1 else success_lock
                })
        # Only add group if it has tasks to run
        if group_args:
            all_task_args_by_group.append(group_args)

    # Flatten for total count
    all_task_args = [arg for group_args in all_task_args_by_group for arg in group_args]

    # Calculate total expected runs (tasks * repeats per task)
    total_expected_runs = len(all_task_args) * args.repeat

    # Print resume information if applicable
    if args.resume_from and skipped_count > 0:
        print("\n" + "=" * 80)
        print("RESUME MODE SUMMARY")
        print("=" * 80)
        print(f"Skipped {skipped_count} already-completed runs")
        print(f"Remaining tasks to execute: {len(all_task_args)}")
        print(f"Remaining runs to execute: {total_expected_runs}")
        print("=" * 80 + "\n")

    # Print detailed task execution plan (especially useful for resume mode)
    if len(all_task_args) > 0:
        print("\n" + "=" * 80)
        print("EXECUTION PLAN - Tasks and Runs to Execute")
        print("=" * 80)
        print(f"Total groups: {len(all_task_args_by_group)}")
        print(f"Total runs to execute: {len(all_task_args)}")
        print()

        # Collect all tasks that will be run with their run numbers
        from collections import defaultdict
        tasks_to_run = defaultdict(list)  # task_id -> [run_nums]
        for task_arg in all_task_args:
            task_id = task_arg['task_id']
            run_num = task_arg['repeat_idx'] + 1
            tasks_to_run[task_id].append(run_num)

        # Print by group
        for group_idx, group_args in enumerate(all_task_args_by_group, 1):
            if not group_args:
                continue

            # Get unique task IDs in this group
            group_task_ids = sorted(set(arg['task_id'] for arg in group_args))

            # Get site for this group
            first_task = group_task_ids[0]
            site = metadata.get(first_task, {}).get('sites', ['unknown'])[0]

            print(f"Group {group_idx} (Site: {site}, {len(group_task_ids)} tasks, {len(group_args)} runs):")
            print(f"  Tasks: {group_task_ids}")

            # Show detailed run information for each task in this group
            for task_id in group_task_ids:
                runs_to_execute = sorted(tasks_to_run[task_id])
                all_runs = set(range(1, args.repeat + 1))
                skipped_runs = sorted(all_runs - set(runs_to_execute))

                if skipped_runs:
                    print(f"    Task {task_id}: runs {runs_to_execute} (skipped: {skipped_runs})")
                else:
                    print(f"    Task {task_id}: runs {runs_to_execute}")
            print()

        print("=" * 80 + "\n")

    if len(all_task_args) == 0:
        print("\n" + "=" * 80)
        print("All tasks already completed! Nothing to run.")
        print("=" * 80)
        return

    # Run tasks (parallel or serial)
    start_time = time.time()
    total_runs = 0
    total_success = 0
    total_failure = 0
    total_skipped = 0

    # Dictionary to store results by task_id
    task_results_dict = {task_id: [] for task_id in task_ids}

    # Thread-safe printing lock
    print_lock = threading.Lock()

    # Create incremental results file
    results_file = log_dir / "results_incremental.json"
    results_summary_file = log_dir / "results_summary.txt"

    # Initialize or load results file
    # In resume mode, preserve existing results if file exists
    if args.resume_from and results_file.exists():
        print(f"Loading existing results from {results_file}")
        try:
            with open(results_file, 'r') as f:
                existing_data = json.load(f)
            # Preserve existing results and statistics
            existing_results = existing_data.get('results', [])
            print(f"Found {len(existing_results)} existing results to preserve")

            initial_data = {
                'metadata': {
                    'model': args.model,
                    'eval_model': args.llm_eval if args.llm_eval else args.model,
                    'total_tasks': len(task_ids),
                    'task_range': f"{min(task_ids)}-{max(task_ids)}",
                    'repeats_per_task': args.repeat,
                    'max_steps': args.max_steps,
                    'workers': args.workers,
                    'early_stopping': not args.no_early_stop,
                    'start_time': existing_data.get('metadata', {}).get('start_time', datetime.now().isoformat()),
                    'resumed_at': datetime.now().isoformat(),
                    'log_dir': str(log_dir)
                },
                'results': existing_results,  # Preserve existing results
                'statistics': existing_data.get('statistics', {
                    'completed_tasks': 0,
                    'total_runs': 0,
                    'total_success': 0,
                    'total_failure': 0,
                    'total_skipped': 0,
                    'overall_accuracy': 0.0,
                    'average_auc': 0.0
                })
            }
        except Exception as e:
            print(f"Warning: Failed to load existing results: {e}")
            print("Starting with empty results")
            initial_data = {
                'metadata': {
                    'model': args.model,
                    'eval_model': args.llm_eval if args.llm_eval else args.model,
                    'total_tasks': len(task_ids),
                    'task_range': f"{min(task_ids)}-{max(task_ids)}",
                    'repeats_per_task': args.repeat,
                    'max_steps': args.max_steps,
                    'workers': args.workers,
                    'early_stopping': not args.no_early_stop,
                    'start_time': datetime.now().isoformat(),
                    'log_dir': str(log_dir)
                },
                'results': [],
                'statistics': {
                    'completed_tasks': 0,
                    'total_runs': 0,
                    'total_success': 0,
                    'total_failure': 0,
                    'total_skipped': 0,
                    'overall_accuracy': 0.0,
                    'average_auc': 0.0
                }
            }
    else:
        # New run, start with empty results
        initial_data = {
            'metadata': {
                'model': args.model,
                'eval_model': args.llm_eval if args.llm_eval else args.model,
                'total_tasks': len(task_ids),
                'task_range': f"{min(task_ids)}-{max(task_ids)}",
                'repeats_per_task': args.repeat,
                'max_steps': args.max_steps,
                'workers': args.workers,
                'early_stopping': not args.no_early_stop,
                'start_time': datetime.now().isoformat(),
                'log_dir': str(log_dir)
            },
            'results': [],
            'statistics': {
                'completed_tasks': 0,
                'total_runs': 0,
                'total_success': 0,
                'total_failure': 0,
                'total_skipped': 0,
                'overall_accuracy': 0.0,
                'average_auc': 0.0
            }
        }

    with open(results_file, 'w') as f:
        json.dump(initial_data, f, indent=2)

    def save_incremental_result(task_id, repeat_idx, success, status, details, duration, skipped):
        """Save result incrementally to avoid data loss on interruption"""
        # Read current results
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Add new result
        result_entry = {
            'task_id': task_id,
            'repeat': repeat_idx + 1,
            'success': success,
            'status': status,
            'details': details,
            'duration': duration,
            'skipped': skipped,
            'timestamp': datetime.now().isoformat()
        }
        data['results'].append(result_entry)

        # Calculate average AUC from all completed tasks
        def calc_task_auc(task_results):
            """Calculate AUC for a single task"""
            if not task_results:
                return 0.0
            n = len(task_results)
            if n == 1:
                return 1.0 if task_results[0]['success'] else 0.0
            cumulative_success_rates = []
            success_count = 0
            for i, result in enumerate(task_results):
                if result['success']:
                    success_count += 1
                cumulative_rate = success_count / (i + 1)
                cumulative_success_rates.append(cumulative_rate)
            auc = 0.0
            for i in range(n - 1):
                auc += (cumulative_success_rates[i] + cumulative_success_rates[i + 1]) / 2.0
            max_auc = n - 1
            return auc / max_auc if max_auc > 0 else 0.0

        # Group results by task and calculate average AUC
        task_groups_for_auc = {}
        for r in data['results']:
            tid = r['task_id']
            if tid not in task_groups_for_auc:
                task_groups_for_auc[tid] = []
            task_groups_for_auc[tid].append(r)

        task_aucs = []
        for tid in task_groups_for_auc:
            task_res = sorted(task_groups_for_auc[tid], key=lambda x: x['repeat'])
            task_auc = calc_task_auc(task_res)
            task_aucs.append(task_auc)

        avg_auc = sum(task_aucs) / len(task_aucs) if task_aucs else 0.0

        # Update statistics
        data['statistics']['completed_tasks'] = len(set(r['task_id'] for r in data['results']))
        data['statistics']['total_runs'] = total_runs
        data['statistics']['total_success'] = total_success
        data['statistics']['total_failure'] = total_failure
        data['statistics']['total_skipped'] = total_skipped
        data['statistics']['overall_accuracy'] = 100.0 * total_success / total_runs if total_runs > 0 else 0.0
        data['statistics']['average_auc'] = avg_auc
        data['statistics']['last_update'] = datetime.now().isoformat()

        # Write back
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Also update summary text file
        with open(results_summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WebArena-Lite Testing - Real-time Results\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model: {data['metadata']['model']}\n")
            f.write(f"Eval Model: {data['metadata']['eval_model']}\n")
            f.write(f"Started: {data['metadata']['start_time']}\n")
            f.write(f"Last Update: {data['statistics']['last_update']}\n\n")
            f.write(f"Progress: {len(data['results'])}/{len(task_ids) * args.repeat} runs completed\n")
            f.write(f"Tasks completed: {data['statistics']['completed_tasks']}/{len(task_ids)}\n")
            f.write(f"Success: {data['statistics']['total_success']}\n")
            f.write(f"Failure: {data['statistics']['total_failure']}\n")
            f.write(f"Skipped: {data['statistics']['total_skipped']}\n")
            f.write(f"Overall Accuracy: {data['statistics']['overall_accuracy']:.2f}%\n")
            f.write(f"Average AUC: {data['statistics'].get('average_auc', 0.0):.4f}\n")
            f.write("\n" + "=" * 80 + "\n\n")

            # Group results by task
            task_groups = {}
            for r in data['results']:
                tid = r['task_id']
                if tid not in task_groups:
                    task_groups[tid] = []
                task_groups[tid].append(r)

            f.write("Results by Task:\n")
            f.write("-" * 80 + "\n")
            for tid in sorted(task_groups.keys()):
                task_res = task_groups[tid]
                success_count = sum(1 for r in task_res if r['success'])
                f.write(f"Task {tid}: {success_count}/{len(task_res)} successful\n")
                for r in task_res:
                    status_icon = "✓" if r['success'] else "✗"
                    f.write(f"  Run {r['repeat']}: {status_icon} {r['status']} ({r['duration']:.1f}s)\n")

            # Add detailed task results with success sequence and AUC
            f.write("\n" + "=" * 80 + "\n")
            f.write("Detailed Task Results (Success Sequence, AUC, and Accuracy)\n")
            f.write("=" * 80 + "\n")
            f.write("Format: Task ID | Success Sequence | AUC | ACC\n")
            f.write("-" * 80 + "\n")

            # Function to calculate cumulative success rate AUC
            def calculate_cumulative_auc(task_results):
                """Calculate the area under the cumulative success rate curve."""
                if not task_results:
                    return 0.0

                n = len(task_results)
                cumulative_success_rates = []

                # Calculate cumulative success rate at each run
                success_count = 0
                for i, result in enumerate(task_results):
                    if result['success']:
                        success_count += 1
                    cumulative_rate = success_count / (i + 1)
                    cumulative_success_rates.append(cumulative_rate)

                # Calculate AUC using trapezoidal rule
                auc = 0.0
                for i in range(n - 1):
                    auc += (cumulative_success_rates[i] + cumulative_success_rates[i + 1]) / 2.0

                # Normalize: max AUC is (n-1) when all runs succeed
                max_auc = n - 1
                normalized_auc = auc / max_auc if max_auc > 0 else 0.0

                return normalized_auc

            for tid in sorted(task_groups.keys()):
                task_res = sorted(task_groups[tid], key=lambda x: x['repeat'])

                # Build success sequence (1 for success, 0 for failure, S for skipped)
                success_sequence = []
                for r in task_res:
                    if r.get('skipped', False):
                        success_sequence.append('S')
                    elif r['success']:
                        success_sequence.append('1')
                    else:
                        success_sequence.append('0')

                # Calculate AUC
                if len(task_res) > 1:
                    task_auc = calculate_cumulative_auc(task_res)
                else:
                    task_auc = 1.0 if task_res[0]['success'] else 0.0

                # Calculate ACC (accuracy)
                success_count = sum(1 for r in task_res if r['success'])
                task_acc = 100.0 * success_count / len(task_res) if len(task_res) > 0 else 0.0

                sequence_str = ','.join(success_sequence)
                f.write(f"Task {tid:3d} | {sequence_str:20s} | AUC: {task_auc:.4f} | ACC: {task_acc:6.2f}%\n")

    if args.workers > 1:
        # Smart parallel execution: groups run in parallel, tasks within groups run serially
        print(f"Starting smart parallel execution with {args.workers} workers...")
        print(f"  {len(task_groups)} groups will run in parallel")
        print(f"  Tasks within each group run serially\n")

        with ProcessPoolExecutor(max_workers=min(args.workers, len(task_groups))) as executor:
            # Submit each group as a unit
            future_to_group = {executor.submit(run_group_serially, group_args): i
                              for i, group_args in enumerate(all_task_args_by_group)}

            # Process completed groups
            for future in as_completed(future_to_group):
                group_results = future.result()

                # Process all results from the completed group
                for result in group_results:
                    task_id = result['task_id']
                    repeat_idx = result['repeat_idx']
                    success = result['success']
                    status = result['status']
                    details = result['details']
                    duration = result['duration']
                    skipped = result.get('skipped', False)

                    if skipped:
                        total_skipped += 1
                        result_str = "⊘ SKIPPED (already succeeded)"
                    else:
                        total_runs += 1
                        if success:
                            total_success += 1
                            result_str = "✓ SUCCESS"
                        else:
                            total_failure += 1
                            result_str = f"✗ FAILURE ({status})"

                    # Store result
                    task_results_dict[task_id].append({
                        'repeat': repeat_idx + 1,
                        'success': success,
                        'status': status,
                        'details': details,
                        'duration': duration,
                        'skipped': skipped
                    })

                    # Print progress (thread-safe)
                    with print_lock:
                        overall_accuracy = 100.0 * total_success / total_runs if total_runs > 0 else 0
                        completed = total_runs + total_skipped
                        print(f"[{completed}/{total_expected_runs}] Task {task_id} (Run {repeat_idx + 1}): {result_str} | "
                              f"Duration: {duration:.1f}s | Overall: {total_success}/{total_runs} "
                              f"({overall_accuracy:.2f}%)")

                        # Save result incrementally (thread-safe)
                        save_incremental_result(task_id, repeat_idx, success, status, details, duration, skipped)
    else:
        # Serial execution
        print("Starting serial execution...\n")

        for idx, task_args in enumerate(all_task_args, 1):
            results = run_task_wrapper(task_args)
            # run_task_wrapper now returns a list of results (one per repeat)
            
            # Process each result in the list
            for result in results:
                task_id = result['task_id']
                repeat_idx = result['repeat_idx']
                success = result['success']
                status = result['status']
                details = result['details']
                duration = result['duration']
                skipped = result.get('skipped', False)

                if skipped:
                    total_skipped += 1
                    result_str = "⊘ SKIPPED (already succeeded)"
                else:
                    total_runs += 1
                    if success:
                        total_success += 1
                        result_str = "✓ SUCCESS"
                    else:
                        total_failure += 1
                        result_str = f"✗ FAILURE ({status})"

                # Store result
                task_results_dict[task_id].append({
                    'repeat': repeat_idx + 1,
                    'success': success,
                    'status': status,
                    'details': details,
                    'duration': duration,
                    'skipped': skipped
                })

                # Print progress
                overall_accuracy = 100.0 * total_success / total_runs if total_runs > 0 else 0
                completed = total_runs + total_skipped
                print(f"[{completed}/{total_expected_runs}] Task {task_id} (Run {repeat_idx + 1}): {result_str}")
                print(f"  Duration: {duration:.1f}s")
                print(f"  Overall progress: {completed}/{total_expected_runs} | Success: {total_success} | "
                      f"Failure: {total_failure} | Skipped: {total_skipped} | Accuracy: {overall_accuracy:.2f}%")
                print()

                # Save result incrementally
                save_incremental_result(task_id, repeat_idx, success, status, details, duration, skipped)

    total_duration = time.time() - start_time

    # Function to calculate cumulative success rate AUC
    def calculate_cumulative_auc(task_results):
        """
        Calculate the area under the cumulative success rate curve.

        Args:
            task_results: List of task run results (sorted by repeat number)

        Returns:
            Normalized AUC in [0, 1] range
        """
        if not task_results:
            return 0.0

        n = len(task_results)
        cumulative_success_rates = []

        # Calculate cumulative success rate at each run
        success_count = 0
        for i, result in enumerate(task_results):
            if result['success']:
                success_count += 1
            cumulative_rate = success_count / (i + 1)
            cumulative_success_rates.append(cumulative_rate)

        # Calculate AUC using trapezoidal rule
        # For points at x = 1, 2, 3, ..., n with uniform spacing of 1
        auc = 0.0
        for i in range(n - 1):
            # Area of trapezoid: (y[i] + y[i+1]) / 2 * delta_x
            # delta_x = 1 for uniform spacing
            auc += (cumulative_success_rates[i] + cumulative_success_rates[i + 1]) / 2.0

        # Normalize: max AUC is (n-1) when all runs succeed (all rates = 1.0)
        # AUC = (1 + 1)/2 * (n-1) = n-1
        max_auc = n - 1
        normalized_auc = auc / max_auc if max_auc > 0 else 0.0

        return normalized_auc

    # Reorganize results by task
    results = []
    for task_id in task_ids:
        task_results = sorted(task_results_dict[task_id], key=lambda x: x['repeat'])
        task_success_count = sum(1 for r in task_results if r['success'])
        # Compute attempts excluding skipped repeats for fair accuracy
        task_attempts = sum(1 for r in task_results if not r.get('skipped', False))
        task_accuracy = 100.0 * task_success_count / task_attempts if task_attempts > 0 else 0

        # Calculate cumulative success rate AUC
        if len(task_results) > 1:
            task_auc = calculate_cumulative_auc(task_results)
        else:
            task_auc = 1.0 if task_success_count > 0 else 0.0

        results.append({
            'task_id': task_id,
            'repeats': task_results,
            'success_count': task_success_count,
            'attempted_repeats': task_attempts,
            'total_repeats': args.repeat,
            'task_accuracy': task_accuracy,
            'cumulative_auc': task_auc
        })

    # Print final summary
    print("=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"Total tasks tested: {len(task_ids)}")
    print(f"Repeats per task: {args.repeat}")
    print(f"Total runs attempted: {total_expected_runs}")
    print(f"Actual runs: {total_runs}")
    print(f"Successful runs: {total_success}")
    print(f"Failed runs: {total_failure}")
    print(f"Skipped runs: {total_skipped} (due to early success)")

    if total_runs > 0:
        overall_accuracy = 100.0 * total_success / total_runs
        print(f"Overall accuracy: {overall_accuracy:.2f}%")

    # Task-level success rate (any success across repeats)
    tasks_with_any_success = sum(1 for r in results if r['success_count'] > 0)
    task_any_success_rate = 100.0 * tasks_with_any_success / len(task_ids) if len(task_ids) > 0 else 0
    print(f"Task success rate (any success): {tasks_with_any_success}/{len(task_ids)} ({task_any_success_rate:.2f}%)")

    print(f"Total duration: {total_duration:.1f}s")
    if total_runs > 0:
        print(f"Average per run: {total_duration/total_runs:.1f}s")

    if args.repeat > 1:
        # Calculate task-level accuracy
        tasks_with_any_success = sum(1 for r in results if r['success_count'] > 0)
        tasks_with_all_success = sum(1 for r in results if r['success_count'] == args.repeat)

        # Calculate average cumulative AUC
        average_auc = sum(r['cumulative_auc'] for r in results) / len(results) if results else 0.0

        print(f"\nTask-level statistics:")
        print(f"  Tasks with at least 1 success: {tasks_with_any_success}/{len(task_ids)} ({100.0*tasks_with_any_success/len(task_ids):.1f}%)")
        print(f"  Tasks with all successes: {tasks_with_all_success}/{len(task_ids)} ({100.0*tasks_with_all_success/len(task_ids):.1f}%)")
        print(f"  Average Cumulative Success Rate AUC: {average_auc:.4f}")

    print()

    # Calculate average AUC for JSON
    average_auc = sum(r['cumulative_auc'] for r in results) / len(results) if results else 0.0

    # Save detailed results
    results_file = log_dir / "results.json"
    result_data = {
        'benchmark': 'WebArena-Lite',
        'model': args.model,
        'timestamp': datetime.now().isoformat(),
        'total_tasks': len(task_ids),
        'repeats_per_task': args.repeat,
        'total_runs_attempted': total_expected_runs,
        'actual_runs': total_runs,
        'successful_runs': total_success,
        'failed_runs': total_failure,
        'skipped_runs': total_skipped,
        'overall_accuracy': overall_accuracy if total_runs > 0 else 0,
        'average_cumulative_auc': average_auc,
        'total_duration': total_duration,
        'results': results
    }

    with open(results_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  Final results (JSON): {results_file}")
    print(f"  Incremental results (JSON): {log_dir / 'results_incremental.json'}")
    print(f"  Real-time summary (TXT): {log_dir / 'results_summary.txt'}")

    # Save summary
    summary_file = log_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("WebArena-Lite Test Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Task Range: {min(task_ids)} to {max(task_ids)}\n")
        f.write(f"Repeats per task: {args.repeat}\n")
        f.write(f"Early stopping: {'Disabled (run all repeats)' if args.no_early_stop else 'Enabled (skip remaining repeats after first success)'}\n\n")
        f.write("Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total tasks tested: {len(task_ids)}\n")
        f.write(f"Total runs attempted: {total_expected_runs}\n")
        f.write(f"Actual runs: {total_runs}\n")
        f.write(f"Successful runs: {total_success}\n")
        f.write(f"Failed runs: {total_failure}\n")
        f.write(f"Skipped runs: {total_skipped} (due to early success)\n")
        f.write(f"Overall accuracy: {overall_accuracy:.2f}%\n" if total_runs > 0 else "Overall accuracy: N/A\n")
        f.write(f"Task success rate (any success): {task_any_success_rate:.2f}%\n")
        f.write(f"Average Cumulative Success Rate AUC: {average_auc:.4f}\n")
        f.write(f"Total duration: {total_duration:.1f}s\n")
        f.write(f"Average per run: {total_duration/total_runs:.1f}s\n" if total_runs > 0 else "")

        if args.repeat > 1:
            f.write(f"\nTask-level statistics:\n")
            f.write(f"  Tasks with at least 1 success: {tasks_with_any_success}/{len(task_ids)} ({100.0*tasks_with_any_success/len(task_ids):.1f}%)\n")
            f.write(f"  Tasks with all successes: {tasks_with_all_success}/{len(task_ids)} ({100.0*tasks_with_all_success/len(task_ids):.1f}%)\n")

        # List all tasks with their status
        f.write("\nAll Tasks Status:\n")
        f.write("-" * 80 + "\n")
        for result in results:
            task_id = result['task_id']
            success_count = result['success_count']
            attempted_repeats = result['attempted_repeats']
            total_repeats = result['total_repeats']
            task_auc = result.get('cumulative_auc', 0.0)

            if attempted_repeats > 0 and success_count == attempted_repeats:
                # All attempted runs succeeded
                if args.repeat > 1:
                    status_str = f"✓ SUCCESS ({success_count}/{total_repeats}, AUC: {task_auc:.4f})"
                else:
                    # Get the status from the first (only) repeat
                    status_str = "✓ SUCCESS"
                f.write(f"  Task {task_id}: {status_str}\n")
            elif success_count > 0:
                # Partial success
                if args.repeat > 1:
                    status_str = f"⚠ PARTIAL ({success_count}/{total_repeats} success, AUC: {task_auc:.4f})"
                else:
                    status_str = f"⚠ PARTIAL ({success_count}/{total_repeats} success)"
                f.write(f"  Task {task_id}: {status_str}\n")
            else:
                # All runs failed
                # Get the failure reason from the first repeat
                failure_status = result['repeats'][0]['status']
                if args.repeat > 1:
                    status_str = f"✗ FAILED ({failure_status}) - 0/{total_repeats} (AUC: {task_auc:.4f})"
                else:
                    status_str = f"✗ FAILED ({failure_status})"
                f.write(f"  Task {task_id}: {status_str}\n")

            # If repeat > 1, show details for each run
            if args.repeat > 1:
                for repeat in result['repeats']:
                    if repeat.get('skipped', False):
                        status_icon = "⊘"
                        status_text = "SKIPPED (already succeeded)"
                    else:
                        status_icon = "✓" if repeat['success'] else "✗"
                        status_text = repeat['status']
                    f.write(f"    Run {repeat['repeat']}: {status_icon} {status_text}\n")

        # Add detailed task results with success sequence and AUC
        f.write("\n" + "=" * 80 + "\n")
        f.write("Detailed Task Results (Success Sequence, AUC, and Accuracy)\n")
        f.write("=" * 80 + "\n")
        f.write("Format: Task ID | Success Sequence | AUC | ACC\n")
        f.write("-" * 80 + "\n")

        for result in results:
            task_id = result['task_id']
            task_auc = result.get('cumulative_auc', 0.0)
            task_acc = result.get('task_accuracy', 0.0)

            # Build success sequence (1 for success, 0 for failure/skipped)
            success_sequence = []
            for repeat in result['repeats']:
                if repeat.get('skipped', False):
                    # Skipped runs are not counted in the sequence
                    success_sequence.append('S')
                elif repeat['success']:
                    success_sequence.append('1')
                else:
                    success_sequence.append('0')

            sequence_str = ','.join(success_sequence)
            f.write(f"Task {task_id:3d} | {sequence_str:20s} | AUC: {task_auc:.4f} | ACC: {task_acc:6.2f}%\n")

        # List tasks with failures (separate section for easier analysis)
        tasks_with_failures = [r for r in results if r['success_count'] < args.repeat]
        if tasks_with_failures:
            f.write("\nFailed Tasks Summary:\n")
            f.write("-" * 80 + "\n")
            for result in tasks_with_failures:
                task_id = result['task_id']
                success_count = result['success_count']
                failure_status = result['repeats'][0]['status']
                task_auc = result.get('cumulative_auc', 0.0)
                if args.repeat > 1:
                    f.write(f"  Task {task_id}: {success_count}/{args.repeat} successes ({result['task_accuracy']:.1f}%, AUC: {task_auc:.4f}) - Status: {failure_status}\n")
                else:
                    f.write(f"  Task {task_id}: {success_count}/{args.repeat} successes ({result['task_accuracy']:.1f}%) - Status: {failure_status}\n")

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
