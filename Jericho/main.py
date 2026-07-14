import os
import argparse
from src.evaluation import GameEvaluator

try:
    from debug_fix import setup_multiprocessing_for_debug
except ImportError:
    def setup_multiprocessing_for_debug():
        pass  # No-op if debug_fix module is not available


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Game
    parser.add_argument('--rom_path', default='jericho-games/', type=str, help="Path to the directory containing game ROMs.")
    parser.add_argument('--game_name', default='library', type=str, help="Name of the game to play (e.g., 'zork1', 'library').")

    parser.add_argument('--output_path', default='output', type=str, help="Base directory for all output, logs, and outputs.") 
    parser.add_argument('--env_step_limit', default=50, type=int, help="Maximum number of steps per game episode.")
    parser.add_argument('--seed', default=0, type=int, help="Random seed for reproducibility. If None, a random seed is used.")

    # LLM
    parser.add_argument('--llm_model', default='google/gemini-2.5-flash', type=str, help="LLM model for the game-playing agent.")

    parser.add_argument('--top_actions', default=3, type=int, help="Number of potential action.")
    parser.add_argument('--llm_temperature', default=0.8, type=float, help="Temperature for the agent's LLM.")
    parser.add_argument('--max_memory', default=30, type=int, help="Maximum number of past states to keep in memory for the agent.")
    parser.add_argument('--gamma', default=0.5, type=float, help="Discount factor for computing returns in cross-episode memory.")
    parser.add_argument('--max_trajectory_window', default=5, type=int, help="Maximum window size for trajectory comparison (uses sliding window for longer trajectories).")
    parser.add_argument('--exploration_rate', default=0.65, type=float, help="Exploration rate parameter for adjusting confidence calculation in action selection.")
    parser.add_argument('--use_history_prompt', default=True, action=argparse.BooleanOptionalAction, help="Use history-based prompt generation that compares with top-scoring episodes (recommended).")

    # Debug options
    parser.add_argument('--debug_info', default=False, action=argparse.BooleanOptionalAction, help='Print detailed info updates during game episodes.')
    parser.add_argument('--track_valid_changes', default=False, action=argparse.BooleanOptionalAction, help='Track valid action changes (if applicable).')

    # Evaluation parameters
    parser.add_argument('--agent_type', type=str, default='jitrl', choices=['jitrl', 'naive', 'awm'], help='Method to evaluate.')

    parser.add_argument('--eval_runs', type=int, default=50, help='Number of episodes to run for statistical evaluation.')
    parser.add_argument('--evol_temperature', default=0.8, type=float, help="Temperature for the evolutionary's LLM.")

    # Memory agent parameters
    parser.add_argument('--retrieval_top_k', type=int, default=10, help='Number of top-k most similar trajectories to retrieve from cross-episode memory.')
    parser.add_argument('--retrieval_threshold', type=float, default=0.95, help='Similarity threshold for retrieving relevant history entries.')

    # Evolutionary parameters (used by AWM induction model)
    parser.add_argument('--evolution_llm_model', default='google/gemini-2.5-flash', type=str, help='LLM model for the evolutionary operator.')

    # Cross-episode memory toggle (few-shot positives + negative contrast during evolution)
    parser.add_argument('--enable_cross_mem', default=True, action=argparse.BooleanOptionalAction,
                    help='Enable cross-episode memory: store successful/failed snippets across episodes, few-shot retrieval, and negative-contrast evolution.')

    # Guiding prompt update control
    parser.add_argument('--update_guiding_prompt', default=False, action=argparse.BooleanOptionalAction,
                    help='Enable automatic guiding prompt updates at the end of each episode based on performance.')

    # Valid actions control
    parser.add_argument('--use_valid_actions', default=True, action=argparse.BooleanOptionalAction,
                    help='Provide the list of valid actions from the game environment to the agent.')

    # Confidence mode control
    parser.add_argument('--confidence_mode', type=str, default='verbalized', choices=['logit', 'verbalized'],
                    help='Mode for confidence calculation: "logit" extracts logprobs (OpenAI only), "verbalized" asks model for explicit confidence percentages (works with all models).')

    # Evaluation LLM model (for step scoring)
    parser.add_argument('--eval_llm_model', type=str, default='google/gemini-2.5-flash',
                    help='LLM model for evaluating step scores in cross-episode memory.')

    return parser.parse_args()


if __name__ == "__main__":
    setup_multiprocessing_for_debug()
    
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    evaluator = GameEvaluator(args)
    results = evaluator.run_evaluation()
    
    # Exit with appropriate code
    if results.get("success", False):
        print("Evaluation completed successfully!")
        exit(0)
    else:
        print("Evaluation failed!")
        exit(1)