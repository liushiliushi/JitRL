import traceback
import json
import math
import os
import re
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from browsergym.experiments import Agent, AbstractAgentArgs
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from langchain.schema import HumanMessage, SystemMessage
from warnings import warn

from .utils.openai_helpers import chat_completion_with_retries, extract_json_from_response, TokenLimitExceededError
from .utils.cross_episode_memory import CrossEpisodeMemory
from .utils.utils import generate_trajectory_summary, generate_history_summary, dump_obs
from .utils.chat_api import ChatModelArgs
from .utils.llm_utils import ParseError, retry
from .dynamic_prompting import ActionSpace


from . import dynamic_prompting


class StateNode:
    def __init__(self, state, instruction, reward=0.0):
        self.state = state
        self.instruction = instruction
        self.reward = reward
        self.response = ""
        

@dataclass
class MemoryAgentArgs(AbstractAgentArgs):    
    chat_model_args: ChatModelArgs = None
    flags: dynamic_prompting.Flags = field(default_factory=lambda: dynamic_prompting.Flags())
    args: any = None  # To hold the parsed arguments

    def make_agent(self):
        return BrowserGymMemoryAgent(
            args=self.args, 
            chat_model_args=self.chat_model_args, 
            flags=self.flags,
            guiding_prompt=None
        )


class BrowserGymMemoryAgent(Agent):
    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Augment observations with text HTML and AXTree representations, which will be stored in
        the experiment traces.
        """

        obs = obs.copy()
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])
        return obs


    def __init__(
        self,
        args,
        chat_model_args: ChatModelArgs = None,
        flags: dynamic_prompting.Flags = None,
        guiding_prompt: str = None,
    ):
        self.args = args
        self.chat_model_args = chat_model_args
        self.flags = flags
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."

        self.memory = [] # Used by agent
        self.web_history = [] # Used by evolutionary LLM
        self.best_score = float("-inf")
        # Check both enable_cross_mem and disable_memory flags
        disable_memory = getattr(self.args, 'disable_memory', False)
        self.enable_cross_mem = getattr(self.args, 'enable_cross_mem', True) and not disable_memory
        self.logit_mode = getattr(self.args, 'logit_mode', 'token')  # 'token' or 'verbalized'

        self.action_space = ActionSpace(self.flags)

        # Cross-episode memory
        if self.enable_cross_mem:
            # Build output dir same structure as evaluation does: output/<game>/<agent_type>/<model_slug>/<timestamp>
            # For cross-episode we use base path output/<game>/<agent_type>/<model_slug>
            game_dir = getattr(self.args, 'output_path', 'output')
            game_dir = os.path.join(game_dir, getattr(self.args, 'game_name', 'game'))
            model_slug = getattr(self.args, 'llm_model', 'model').replace('/', '_').replace('\\', '_')
            agent_type = getattr(self.args, 'agent_type', 'our')
            self.cross_mem_dir = os.path.join(game_dir, agent_type, model_slug)
            gamma = getattr(self.args, 'gamma', 0.95)
            llm_model = getattr(self.args, 'llm_model', 'gpt-4')
            llm_extract = getattr(self.args, 'llm_extract', None) or llm_model
            self.llm_extract = llm_extract  # Save for use in generate_single_step_summary
            task_similarity_threshold = getattr(self.args, 'task_similarity_threshold', 0.27)
            self.cross_mem = CrossEpisodeMemory(
                self.cross_mem_dir,
                gamma=gamma,
                llm_model=llm_model,
                llm_extract=llm_extract,
                task_similarity_threshold=task_similarity_threshold
            )
            print(f"[INFO] Cross-episode memory enabled. Memory directory: {self.cross_mem_dir}")
        else:
            self.cross_mem_dir = None
            self.cross_mem = None
            if disable_memory:
                print(f"[INFO] Memory functionality disabled (--disable_memory=true). Agent will rely purely on model capabilities.")
                
        # Simple loop detection buffer for scores 
        self._recent_scores = []
        
        print(chat_model_args)
        print(f"\n--- STARTING EVALUATION ---")
        print(f"Task: WebArena, Agent LLM Model: {self.args.llm_model}")
        print(f"Agent Type: {self.args.agent_type}, Runs for statistics: {self.args.eval_runs}")
        print(f"Base Seed for evaluation session: {self.args.seed}")
        self.start_episode()
        
        
    def add_to_memory(self, state, response):
        memory_entry = {"state": state, "response": response}
        self.memory.append(memory_entry)
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)


    def start_episode(self):
        """
        """
        self.memory = []
        self.game_history = []
        self._recent_scores = []
        self.step_summaries = []
        print(f"Using initial prompt: '{self.guiding_prompt}'")


    def end_episode(self, state, score, success, llm_analysis, task_goal=None, user_instruction=None, screenshots_dir=None):
        """
        End an episode: update the current node's score and game history.

        Args:
            state: Current state
            score: Final score
            success: Whether the episode was successful
            llm_analysis: LLM analysis of the episode
            task_goal: Task goal/instruction for this episode
            user_instruction: Custom user instruction for scoring actions (optional)
        """
        print(f"Ending episode with score: {score}.")

        # Skip memory storage if memory is disabled
        if not self.enable_cross_mem or self.cross_mem is None:
            print(f"[INFO] Memory disabled - skipping episode storage and step evaluation")
            return

        # Check if save_memory is disabled
        save_memory = getattr(self.args, 'save_memory', True)
        if not save_memory:
            print(f"[INFO] save_memory=False - skipping episode storage (memory retrieval still enabled)")
            return

        # Store complete episode in cross-episode memory
        if self.game_history:
            # Determine if episode was successful (score > 0 or won)
            # success = self._detect_victory_from_observation(state)
            print(f"Episode success: {success}")
            # Use saved task goal if not provided
            if task_goal is None and hasattr(self, 'current_task_goal'):
                task_goal = self.current_task_goal

            # Auto-detect screenshots directory from task_name if use_screenshot_eval is enabled
            if screenshots_dir is None and hasattr(self.args, 'use_screenshot_eval') and self.args.use_screenshot_eval:
                if hasattr(self.args, 'task_name'):
                    task_name = self.args.task_name
                    result_dir = getattr(self.args, 'result_dir', 'results')
                    screenshots_dir = f"./{result_dir}/{task_name}"
                    print(f"[INFO] Screenshot evaluation enabled. Using screenshots from: {screenshots_dir}")
                else:
                    print(f"[WARNING] Screenshot evaluation enabled but task_name not found in args")
            elif screenshots_dir is None:
                print(f"[INFO] Screenshot evaluation disabled (use --use_screenshot_eval to enable)")

            self.cross_mem.add_episode(
                    game_history=self.game_history,
                    state=state,
                    final_score=score,
                    success=success,
                    llm_analysis=llm_analysis,
                    task_goal=task_goal,
                    user_instruction=user_instruction,
                    screenshots_dir=screenshots_dir
                )

        victory = success

    
    def update_scores(self, state_node, options_with_logits, k, r, memory_text, current_url=None, screenshots_dir=None):
        # Skip memory retrieval if memory is disabled
        if not self.enable_cross_mem or self.cross_mem is None:
            return {}

        from .utils.utils import normalize_action, calculate_action_similarity

        nearest_trajectories = self.cross_mem.retrieve_similar(
                game_history=self.game_history,
                current_state=state_node.state,
                current_summary=memory_text,
                task_goal=state_node.instruction,
                current_url=current_url,
                k=k,
                r=r,
                screenshots_dir=screenshots_dir,
            )

        if not nearest_trajectories:
            return {}

        # Add actions from nearest_trajectories to options_with_logits
        if not options_with_logits:
            options_with_logits = {}

        # Note: options_with_logits already contains normalized_action from generate_action
        # We don't need to normalize again here

        # Get existing normalized actions
        existing_normalized_actions = set()
        for option_data in options_with_logits.values():
            if isinstance(option_data, dict) and 'normalized_action' in option_data:
                existing_normalized_actions.add(option_data['normalized_action'])

        # Step 1: Aggregate action rewards based on NORMALIZED actions
        action_rewards = {}  # Key: normalized_action, Value: list of rewards
        action_to_raw = {}   # Map normalized_action back to raw action for later use

        for sim, discounted_return, result_dict in nearest_trajectories:
            # Get normalized action from history
            normalized_action = result_dict.get('normalized_action', '').strip()
            if not normalized_action:
                # Fallback: use raw action if normalized not available
                normalized_action = result_dict.get('action', '').strip()

            discounted_reward = result_dict.get('discounted_reward', 0)

            if normalized_action not in action_rewards:
                action_rewards[normalized_action] = []
                action_to_raw[normalized_action] = result_dict.get('action', normalized_action)

            action_rewards[normalized_action].append(discounted_reward)
        
        # Add actions from memory that are not in options_with_logits
        for normalized_action in action_rewards:
            found = False
            for option_data in options_with_logits.values():
                if isinstance(option_data, dict) and option_data.get('normalized_action', '') == normalized_action:
                    found = True
                    break

            if not found:
                if options_with_logits:
                    new_option_num = max(options_with_logits.keys()) + 1
                else:
                    new_option_num = 1

                from .utils.utils import denormalize_action
                current_state = state_node.state
                raw_action = denormalize_action(normalized_action, current_state)

                if raw_action is None:
                    print(f"[Memory] Skipped historical action (element not found in current state): {normalized_action}")
                    continue

                options_with_logits[new_option_num] = {
                    'action': raw_action,
                    'normalized_action': normalized_action,
                    'normalized_prob': 0
                }
                print(f"[Memory] Added historical action from memory: {normalized_action} -> {raw_action}")
        # Step 4: Calculate average discounted reward for each NORMALIZED action
        action_avg_rewards = {}  # Key: normalized_action
        for normalized_action, rewards in action_rewards.items():
            if rewards:  # Ensure not empty
                action_avg_rewards[normalized_action] = sum(rewards) / len(rewards)
            else:
                action_avg_rewards[normalized_action] = 0

        # Calculate overall average discounted reward (baseline)
        count = 0
        if action_rewards:
            all_rewards = []
            for rewards_list in action_rewards.values():
                all_rewards.extend(rewards_list)
            count = len(all_rewards)
            overall_avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        else:
            overall_avg_reward = 0

        # Calculate adaptive exploration probability based on trajectory count and quality
        # exploration_prob = self.calculate_exploration_probability(nearest_trajectories, action_rewards)
        exploration_prob = 0.05
        
        alpha = 1
        new_actions_added = False

        # Step 5: For candidate actions not in history, assign exploration reward
        for option_num, option_data in options_with_logits.items():
            normalized_action = option_data.get('normalized_action', option_data.get('action'))

            if normalized_action not in action_avg_rewards:
                if random.random() < exploration_prob:
                    action_avg_rewards[normalized_action] = overall_avg_reward + alpha / max(count, 1)
                else:
                    action_avg_rewards[normalized_action] = 0
                new_actions_added = True

        # If new actions were added, recalculate overall_avg_reward
        if new_actions_added:
            all_avg_rewards = list(action_avg_rewards.values())
            overall_avg_reward = sum(all_avg_rewards) / len(all_avg_rewards) if all_avg_rewards else 0

        # Step 6: Calculate advantage value for each NORMALIZED action
        action_advantages = {}  # Key: normalized_action
        for normalized_action, avg_reward in action_avg_rewards.items():
            action_advantages[normalized_action] = avg_reward - overall_avg_reward
        
        # Normalize advantage values
        if action_advantages:
            import numpy as np
            adv_values = list(action_advantages.values())
            
            positive_advs = [adv for adv in adv_values if adv > 0]

            if positive_advs:
                max_positive = max(positive_advs)
                normalized_advantages = {action: adv / max_positive for action, adv in action_advantages.items()}
            else:
                max_negative_abs = abs(min(adv_values))
                if max_negative_abs > 0:
                    normalized_advantages = {action: adv / max_negative_abs for action, adv in action_advantages.items()}
                else:
                    normalized_advantages = {action: 0 for action in action_advantages.keys()}
        else:
            normalized_advantages = {}
        
        # Step 7: Correct logit values in options_with_logits based on NORMALIZED actions
        updated_options = {}
        if options_with_logits and normalized_advantages:
            print("\n" + "="*100)
            print("MEMORY-BASED SCORE ADJUSTMENT")
            print("="*100)
            print(f"Baseline (average reward): {overall_avg_reward:.4f}")
            print("-"*100)

            print("Action Rewards from Memory:")
            for normalized_action, rewards in action_rewards.items():
                print(f"  {normalized_action[:80]}...")
                print(f"    Rewards: {rewards}")
            print("-"*100)

            for option_num, option_data in options_with_logits.items():
                if isinstance(option_data, dict) and 'action' in option_data:
                    raw_action = option_data['action']
                    normalized_action = option_data.get('normalized_action', raw_action)

                    # Get normalized advantage value based on NORMALIZED action
                    normalized_advantage = normalized_advantages.get(normalized_action, 0)
                    raw_advantage = action_advantages.get(normalized_action, 0)
                    avg_reward = action_avg_rewards.get(normalized_action, 0)

                    # Correct logit (add normalized advantage value)
                    normalized_prob = option_data.get('normalized_prob', 0)
                    corrected_logprob = normalized_prob + 1 * normalized_advantage

                    # Create corrected option data
                    updated_options[option_num] = {
                        'action': raw_action,  # Keep raw action for execution
                        'normalized_action': normalized_action,
                        'normalized_advantage': normalized_advantage,
                        'corrected_logprob': corrected_logprob,
                        'avg_reward': avg_reward,
                        'raw_advantage': raw_advantage,
                        'token': option_data.get('token', ''),
                        'normalized_prob': option_data.get('normalized_prob', 0)
                    }

                    print(f"[Option {option_num}] {normalized_action}")
                    print(f"  Avg Reward: {avg_reward:.4f} | Advantage: {raw_advantage:+.4f} (normalized: {normalized_advantage:+.4f})")
                    print(f"  Score: {normalized_prob:.4f} → {corrected_logprob:.4f}")
                else:
                    # If no matching advantage value, keep original value
                    updated_options[option_num] = option_data.copy()
                    updated_options[option_num]['corrected_logprob'] = option_data.get('logprob', 0)
                    updated_options[option_num]['normalized_advantage'] = 0
            print("="*100)
            return updated_options
        else:
            # If no advantage value data, return original options_with_logits
            if options_with_logits:
                updated_options = {}
                for option_num, option_data in options_with_logits.items():
                    updated_options[option_num] = option_data.copy()
                    updated_options[option_num]['corrected_logprob'] = option_data.get('logprob', 0)
                    updated_options[option_num]['normalized_advantage'] = 0
                return updated_options
            else:
                return {}
            
    def _extract_action_reasoning(self, full_response, action):
        """
        Extract the reasoning for a specific action from the full LLM response.

        Args:
            full_response: The full JSON response from LLM (string or dict)
            action: The action that was executed

        Returns:
            str: The reasoning for this action, or None if not found
        """
        if not full_response or not action:
            return None

        try:
            # Parse response if it's a string
            if isinstance(full_response, str):
                response_dict = json.loads(full_response)
            else:
                response_dict = full_response

            # Find which option matches the action
            for i in range(1, 10):  # Check up to 10 options
                option_key = f"option{i}"
                reasoning_key = f"option{i}_reasoning"

                if option_key in response_dict and reasoning_key in response_dict:
                    # Compare actions (normalize whitespace)
                    if response_dict[option_key].strip() == action.strip():
                        return response_dict[reasoning_key]

            # If no exact match found, return general reasoning
            return response_dict.get('reasoning', None)

        except (json.JSONDecodeError, KeyError, AttributeError):
            return None

    def get_prompts(self, state_node):
        # Enhance send_msg_to_user description in action space prompt
        original_prompt = self.action_space.prompt
        if "send_msg_to_user(text: str)" in original_prompt:
            self.action_space._prompt = original_prompt.replace(
                "send_msg_to_user(text: str)\n    Description: Sends a message to the user.",
                "send_msg_to_user(text: str)\n    Description: Sends final answer to user and terminates the task. ⚠️ WARNING: This immediately ends the task - you cannot take any more actions afterward. ONLY use when you have a definitive final answer. DO NOT use to ask questions or explain your thinking."
            )

        # Build memory_text directly from game_history
        # Use sliding window: detailed info for recent steps, condensed info for older steps
        memory_text = ""
        if self.game_history:
            memory_parts = []

            # Configure window size: only keep detailed state for recent N steps
            recent_window_size = 0 
            history_length = len(self.game_history)

            # If history is very long, use cached summaries for early steps
            if history_length > recent_window_size:
                old_steps_count = history_length - recent_window_size

                # Use cached summaries for early steps (no need to regenerate!)
                if hasattr(self, 'step_summaries') and len(self.step_summaries) >= old_steps_count:
                    memory_parts.append(f"=== Earlier steps summary (0-{old_steps_count-1}) ===")

                    # Add reasoning to each cached summary
                    for idx in range(old_steps_count):
                        summary_text = self.step_summaries[idx]
                        entry = self.game_history[idx]

                        # Ensure step number is included (add prefix if not already present)
                        if not summary_text.strip().startswith(f"Step {idx}"):
                            summary_text = f"Step {idx}: {summary_text}"

                        # Extract and append reasoning
                        reasoning = self._extract_action_reasoning(entry.get('full_response'), entry.get('action'))
                        if reasoning:
                            summary_text += f"\nAction Reasoning: {reasoning}"

                        memory_parts.append(summary_text)

                    memory_parts.append("")  # Add blank line for separation
                else:
                    # Fallback: if no cached summaries, use simple format
                    memory_parts.append(f"=== Earlier steps (0-{old_steps_count-1}) ===")
                    for idx in range(old_steps_count):
                        entry = self.game_history[idx]

                        # Use normalized action if available
                        raw_action = entry.get('action', 'N/A')
                        state_text = entry.get('state', '')
                        try:
                            from .utils.utils import normalize_action
                            normalized_data = normalize_action(raw_action, state_text)
                            action_display = normalized_data.get('normalized_action', raw_action)
                        except:
                            action_display = raw_action

                        step_text = f"Step {idx}: {action_display}"

                        # Extract and display reasoning
                        reasoning = self._extract_action_reasoning(entry.get('full_response'), entry.get('action'))
                        if reasoning:
                            step_text += f"\nAction Reasoning: {reasoning}"

                        if entry.get('url'):
                            step_text += f" | URL: {entry.get('url')}"
                        if entry.get('reward') is not None:
                            step_text += f" | Reward: {entry.get('reward')}"
                        memory_parts.append(step_text)
                    memory_parts.append("")

                # Add detailed info for recent steps (only if recent_window_size > 0)
                if recent_window_size > 0:
                    memory_parts.append(f"=== Recent steps ({history_length - recent_window_size}-{history_length-1}) ===")
                    for idx in range(history_length - recent_window_size, history_length):
                        entry = self.game_history[idx]
                        step_text = f"\nStep {idx}:\n"
                        step_text += f"State:\n{entry.get('state', '')}\n"

                        # Use normalized action if available
                        raw_action = entry.get('action', 'N/A')
                        state_text = entry.get('state', '')
                        try:
                            from .utils.utils import normalize_action
                            normalized_data = normalize_action(raw_action, state_text)
                            action_display = normalized_data.get('normalized_action', raw_action)
                        except:
                            action_display = raw_action

                        step_text += f"Action: {action_display}\n"

                        # Extract and display reasoning for this action
                        reasoning = self._extract_action_reasoning(entry.get('full_response'), entry.get('action'))
                        if reasoning:
                            step_text += f"Action Reasoning: {reasoning}\n"

                        if entry.get('reward') is not None:
                            step_text += f"Reward: {entry.get('reward')}\n"
                        if entry.get('score') is not None:
                            step_text += f"Score: {entry.get('score')}\n"
                        memory_parts.append(step_text)
            else:
                # History is short enough, use sliding window approach
                for idx, entry in enumerate(self.game_history):
                    step_text = f"Step {idx}:\n"

                    # Recent steps: include full state
                    if idx >= history_length - recent_window_size:
                        step_text += f"State:\n{entry.get('state', '')}\n"

                        # Use normalized action if available
                        raw_action = entry.get('action', 'N/A')
                        state_text = entry.get('state', '')
                        try:
                            from .utils.utils import normalize_action
                            normalized_data = normalize_action(raw_action, state_text)
                            action_display = normalized_data.get('normalized_action', raw_action)
                        except:
                            action_display = raw_action

                        step_text += f"Action: {action_display}\n"

                        # Extract and display reasoning
                        reasoning = self._extract_action_reasoning(entry.get('full_response'), entry.get('action'))
                        if reasoning:
                            step_text += f"Action Reasoning: {reasoning}\n"
                    else:
                        # Older steps: only include key information
                        # Use normalized action if available
                        raw_action = entry.get('action', 'N/A')
                        state_text = entry.get('state', '')
                        try:
                            from .utils.utils import normalize_action
                            normalized_data = normalize_action(raw_action, state_text)
                            action_display = normalized_data.get('normalized_action', raw_action)
                        except:
                            action_display = raw_action

                        step_text += f"Action: {action_display}\n"

                        # Extract and display reasoning even for older steps
                        reasoning = self._extract_action_reasoning(entry.get('full_response'), entry.get('action'))
                        if reasoning:
                            step_text += f"Action Reasoning: {reasoning}\n"

                        if entry.get('url'):
                            step_text += f"URL: {entry.get('url')}\n"

                    if entry.get('reward') is not None:
                        step_text += f"Reward: {entry.get('reward')}\n"
                    if entry.get('score') is not None:
                        step_text += f"Score: {entry.get('score')}\n"
                    memory_parts.append(step_text)

            memory_text = "\n".join(memory_parts)

        # Add current step to memory
        if memory_text:
            memory_text += "\n\n"
        memory_text += f"=== Current step ({len(self.game_history)}) ===\n"
        memory_text += f"Step {len(self.game_history)}: State:\n{state_node.state}\n"

        best_action_range = f"1, 2, or {self.args.top_actions}" if self.args.top_actions > 2 else "1 or 2" if self.args.top_actions == 2 else "1"

        # System prompt: Define role, task, and output format
        sys_prompt = f"""You are an intelligent web agent that interacts with real web pages on behalf of the user. Your goal is to accurately follow the user's natural language instructions by selecting and executing appropriate web actions. Select promising actions based on the web state and memory of past interactions.

User's instructions: {state_node.instruction}"""

        print("Current instruction:", state_node.instruction)
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        sys_prompt += f"""

## Available Actions
The available actions you can take are:
{self.action_space.prompt}

## Response Format
You need to think step-by-step, then provide {self.args.top_actions} potential actions you can take as a web agent.

IMPORTANT: When analyzing the state and selecting actions, carefully avoid repeating ineffective patterns:
- If an action failed or gave no progress before, don't try it again in the same context
- Send message to the user once the instruction is fulfilled

⚠️ CRITICAL: send_msg_to_user() is a TASK-ENDING action - once called, the task IMMEDIATELY TERMINATES.
Before using send_msg_to_user():
1. Verify you have gathered ALL information to COMPLETELY answer: "{state_node.instruction}"
2. Ensure your message is CONCISE, ON-TOPIC, and answers EVERY part of the instruction (not just partial info)
3. Do NOT use it to explain your process, ask questions, or report partial progress

If your answer is incomplete or off-topic, continue gathering information instead.
"""

        # Different prompt formats based on logit_mode
        if self.logit_mode == 'verbalized':
            # Verbalized mode: request confidence scores
            sys_prompt += f"""
You must respond with a JSON object containing:

- option1, option2, ..., option{self.args.top_actions}: Each option should be a single action command (string)
- option1_reasoning, option2_reasoning, ..., option{self.args.top_actions}_reasoning: For each option, explain WHY you chose this specific action and what you expect it to accomplish
- option1_confidence, option2_confidence, ..., option{self.args.top_actions}_confidence: For each option, provide your confidence level (0-100) in this action's success
- reasoning: Your overall reasoning about the current state and your thought process
- best_action: The number ({best_action_range}) of the option you think is best

Example JSON response format:
{{
    "reasoning": "I need to... because...",
    "option1": "click('a51')",
    "option1_reasoning": "This button appears to be the 'Continue' button which should advance the checkout process to the next step",
    "option1_confidence": 85,
    "option2": "fill('b534', '06/24/2002')",
    "option2_reasoning": "This date field might be required before proceeding, filling it could unlock the next step",
    "option2_confidence": 70,"""

            if self.args.top_actions >= 3:
                sys_prompt += """
    "option3": "scroll('down')",
    "option3_reasoning": "Scrolling might reveal additional required fields or a submit button",
    "option3_confidence": 60,"""

            sys_prompt += """
    "best_action": 1
}}"""
        else:
            # Token mode: no confidence scores needed
            sys_prompt += f"""
You must respond with a JSON object containing:

- option1, option2, ..., option{self.args.top_actions}: Each option should be a single action command (string)
- option1_reasoning, option2_reasoning, ..., option{self.args.top_actions}_reasoning: For each option, explain WHY you chose this specific action and what you expect it to accomplish
- reasoning: Your overall reasoning about the current state and your thought process
- best_action: The number ({best_action_range}) of the option you think is best

Example JSON response format:
{{
    "reasoning": "I need to... because...",
    "option1": "click('a51')",
    "option1_reasoning": "This button appears to be the 'Continue' button which should advance the checkout process to the next step",
    "option2": "fill('b534', '06/24/2002')",
    "option2_reasoning": "This date field might be required before proceeding, filling it could unlock the next step","""

            if self.args.top_actions >= 3:
                sys_prompt += """
    "option3": "scroll('down')",
    "option3_reasoning": "Scrolling might reveal additional required fields or a submit button","""

            sys_prompt += """
    "best_action": 1
}}"""

        # Import the model detection function
        from .utils.openai_helpers import _is_openai_model

        # Only add JSON format instructions for non-OpenAI models
        # OpenAI models use response_format parameter instead
        if not _is_openai_model(self.args.llm_model):
            sys_prompt += """

**CRITICAL: YOU MUST OUTPUT ONLY VALID JSON**

YOUR RESPONSE MUST BE PURE JSON ONLY. Follow these rules strictly:
- Do NOT write any explanatory text, thoughts, or natural language before or after the JSON
- Do NOT use markdown code blocks like ```json ... ``` or ``` ... ```
- Do NOT add any commentary or reasoning outside the JSON structure
- Output ONLY the raw JSON object starting with { and ending with }
- The JSON must be valid and parseable by a JSON parser

CORRECT output example (THIS IS WHAT YOU MUST DO):
{"reasoning": "I need to click the submit button", "option1": "click('123')", "option1_reasoning": "This will submit the form", "option2": "fill('456', 'text')", "option2_reasoning": "This fills the required field", "best_action": 1}

INCORRECT output examples (NEVER DO THIS):
❌ Let me think about this... The user wants me to click the button.
❌ ```json
{"reasoning": "..."}
```
❌ I will analyze the state first, then provide my response: {"reasoning": "..."}

Remember: Your ENTIRE response must be ONLY the JSON object. Nothing else."""
        else:
            # For OpenAI models, just a brief reminder (response_format handles the rest)
            sys_prompt += """

Output your response as a JSON object with the specified fields."""

        # User prompt: Provide current context and state
        user_prompt = f"""Your web browsing history and current state:
{memory_text}

Analyze the current state, consider your recent history, and provide {self.args.top_actions} different and useful actions with reasoning for each, then choose the best one."""

        return sys_prompt, user_prompt, memory_text
    
    
    def _validate_json_response(self, json_response, top_actions):
        """
        Validate if the JSON response contains all required fields.

        Returns:
            tuple: (is_valid: bool, error_message: str)
        """
        if not json_response or not isinstance(json_response, dict):
            return False, "Response is not a valid dictionary"

        # Check if all required option fields exist
        for i in range(1, top_actions + 1):
            option_key = f"option{i}"
            option_reasoning_key = f"option{i}_reasoning"
            option_confidence_key = f"option{i}_confidence"

            if option_key not in json_response:
                return False, f"Missing required field: {option_key}"
            if option_reasoning_key not in json_response:
                return False, f"Missing required field: {option_reasoning_key}"

            # Check if option is not None or empty
            if not json_response.get(option_key):
                return False, f"{option_key} is empty or None"

            # In verbalized mode, check confidence field
            if self.logit_mode == 'verbalized':
                if option_confidence_key not in json_response:
                    return False, f"Missing required field: {option_confidence_key}"

                confidence = json_response.get(option_confidence_key)
                if not isinstance(confidence, (int, float)):
                    return False, f"{option_confidence_key} must be a number, got: {type(confidence)}"

                if not (0 <= confidence <= 100):
                    return False, f"{option_confidence_key} must be between 0 and 100, got: {confidence}"

        # Check reasoning field
        if 'reasoning' not in json_response:
            return False, "Missing required field: reasoning"

        # Check best_action field
        if 'best_action' not in json_response:
            return False, "Missing required field: best_action"

        best_action = json_response.get('best_action')
        if not isinstance(best_action, (int, float)):
            return False, f"best_action must be a number, got: {type(best_action)}"

        if not (1 <= int(best_action) <= top_actions):
            return False, f"best_action must be between 1 and {top_actions}, got: {best_action}"

        return True, ""

    # Generates the next action from the LLM based on its memory and the current state node.
    def generate_action(self, state_node, url=None, screenshot=None):
        sys_prompt, user_prompt, memory_text = self.get_prompts(state_node)

        # Prepare screenshot as base64 if provided
        image_content = None
        if screenshot is not None:
            import base64
            from io import BytesIO
            from PIL import Image
            import numpy as np

            # Convert numpy array to PIL Image if needed
            if isinstance(screenshot, np.ndarray):
                screenshot = Image.fromarray(screenshot)

            # Convert PIL Image to base64
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_content = image_base64
            print(f"[INFO] Screenshot encoded for action generation (base64 length: {len(image_base64)})")
        # Dynamically create JSON Schema to force final choice to be valid option numbers only
        properties = {}
        required_fields = []
        valid_choices = []
        
        properties["reasoning"] = {
            "type": "string",
            "description": "The overall reasoning about the current state and action selection"
        }
        required_fields.extend(["reasoning"])

        # Dynamically generate option fields and their reasoning fields
        for i in range(1, self.args.top_actions + 1):
            option_key = f"option{i}"
            option_reasoning_key = f"option{i}_reasoning"
            option_confidence_key = f"option{i}_confidence"

            properties[option_key] = {
                "type": "string",
                "description": f"Option {i} possible action"
            }
            properties[option_reasoning_key] = {
                "type": "string",
                "description": f"Reasoning explaining why option {i} was chosen and what it is expected to accomplish"
            }

            required_fields.append(option_key)
            required_fields.append(option_reasoning_key)

            # Add confidence field for verbalized mode
            if self.logit_mode == 'verbalized':
                properties[option_confidence_key] = {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": f"Confidence level (0-100) in the success of option {i}"
                }
                required_fields.append(option_confidence_key)

            valid_choices.append(i)
        
        # Add best_action field

        properties["best_action"] = {
            "type": "number",
            "minimum": 1,
            "maximum": self.args.top_actions,
            "description": f"The number of the best option (must be one of: {valid_choices})"
        }
        required_fields.extend(["best_action"])
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "web_action_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required_fields,
                    "additionalProperties": False
                }
            }
        }

        # Retry logic: try up to 5 times until we get a valid response
        max_retries = 5
        json_response = None
        full_response = None
        res_obj = None

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(f"[Retry {attempt}/{max_retries}]")

            # Adaptive retry: modify prompt based on attempt number
            current_user_prompt = user_prompt
            if attempt >= 4:
                # For attempts 4-5, add urgent JSON-only reminder
                current_user_prompt = user_prompt + f"""\n
⚠️ CRITICAL REMINDER (Attempt {attempt}/{max_retries}):
Your previous {attempt-1} attempts failed because you did not output valid JSON.
You MUST respond with ONLY a JSON object. No explanations, no markdown, no text.
Start your response with {{ and end with }}. Nothing else."""

            try:
                # Only request logprobs in token mode, not in verbalized mode
                top_logprobs = self.args.top_actions if self.logit_mode == 'token' else 0
                res_obj = chat_completion_with_retries(
                    model=self.args.llm_model,
                    sys_prompt=sys_prompt,
                    prompt=current_user_prompt,
                    max_tokens=4000,
                    temperature=self.args.llm_temperature,
                    top_logprobs=top_logprobs,
                    response_format=response_format,
                    image_content=image_content  # Pass screenshot if available
                )
            except TokenLimitExceededError as e:
                # Token limit exceeded - terminate this task immediately
                print(f"\n{'='*80}")
                print(f"FATAL ERROR: Context length exceeded for this task!")
                print(f"Task will be terminated. Error: {e}")
                print(f"{'='*80}\n")
                # Re-raise to propagate to run.py level
                raise

            if not res_obj or not hasattr(res_obj, 'choices') or not res_obj.choices:
                if attempt > 1:
                    print(f"  Failed to get API response")
                continue

            full_response = res_obj.choices[0].message.content

            # Parse JSON response using robust extraction
            json_response = extract_json_from_response(full_response)

            # Validate the response
            is_valid, error_msg = self._validate_json_response(json_response, self.args.top_actions)

            if is_valid:
                break
            else:
                if attempt > 1:
                    print(f"  Invalid response: {error_msg}")
                if attempt < max_retries:
                    json_response = None

        # If all attempts failed, try to extract action from natural language as fallback
        if not json_response:
            print(f"\n!!! Failed to get valid response after {max_retries} attempts !!!")
            print("Attempting to extract action from natural language response...")

            if full_response:
                # Try to find action patterns in the response
                import re

                # Pattern 1: Look for action function calls like click('123'), fill('145', 'text')
                action_patterns = [
                    r"(click\(['\"][\w-]+['\"]\))",
                    r"(fill\(['\"][\w-]+['\"]\s*,\s*['\"].*?['\"]\))",
                    r"(send_msg_to_user\(['\"].*?['\"]\))",
                    r"(scroll\(['\"][\w-]+['\"]\))",
                    r"(select_option\(['\"][\w-]+['\"]\s*,\s*['\"].*?['\"]\))",
                ]

                for pattern in action_patterns:
                    match = re.search(pattern, full_response, re.DOTALL)
                    if match:
                        extracted_action = match.group(1)
                        print(f"Extracted action from natural language: {extracted_action}")
                        return extracted_action, full_response

                # Pattern 2: Look for descriptive text like "fill element 145 with 'text'"
                fill_match = re.search(r"fill.*?['\"]?(\d+)['\"]?.*?with['\"]?\s*['\"]([^'\"]+)['\"]", full_response, re.IGNORECASE)
                if fill_match:
                    element_id = fill_match.group(1)
                    text_value = fill_match.group(2)
                    extracted_action = f"fill('{element_id}', '{text_value}')"
                    print(f"Extracted action from description: {extracted_action}")
                    return extracted_action, full_response

                click_match = re.search(r"click.*?['\"]?(\d+)['\"]?", full_response, re.IGNORECASE)
                if click_match:
                    element_id = click_match.group(1)
                    extracted_action = f"click('{element_id}')"
                    print(f"Extracted action from description: {extracted_action}")
                    return extracted_action, full_response

            print("Could not extract valid action. Using default action 'look'\n")
            return "look", full_response or ""

        all_options = {}
        all_options_reasoning = {}

        # First, collect all options and reasoning (but don't print yet)
        for i in range(1, self.args.top_actions + 1):
            option_key = f"option{i}"
            option_reasoning_key = f"option{i}_reasoning"

            action = json_response.get(option_key)
            reasoning = json_response.get(option_reasoning_key, "No reasoning provided")

            all_options[i] = action
            all_options_reasoning[i] = reasoning

        best_choice = json_response.get("best_action", 1)
        action_text = all_options.get(best_choice)



        # Extract logprobs information based on mode
        options_with_logits = {}

        if self.logit_mode == 'token':
            # Token mode: extract from API response logprobs
            if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and \
               hasattr(res_obj.choices[0], 'logprobs') and res_obj.choices[0].logprobs and \
               res_obj.choices[0].logprobs.content:

                last_token_logprob = res_obj.choices[0].logprobs.content[-2]
                if all_options:
                    for i, candidate in enumerate(last_token_logprob.top_logprobs[:self.args.top_actions]):
                        try:
                            option_num = int(candidate.token)
                            if option_num in all_options:
                                normalized_prob = math.exp(candidate.logprob)
                                options_with_logits[option_num] = {
                                        'action': all_options[option_num],
                                        'token': option_num,
                                        'logprob': candidate.logprob,
                                        'normalized_prob': normalized_prob
                                    }
                        except (ValueError, TypeError):
                                # Skip tokens that cannot be converted to integers
                            continue

                    selected_prob = math.exp(last_token_logprob.logprob)
                    print(f"\n[Token Mode] Extracted logprobs from {len(options_with_logits)} options")

        elif self.logit_mode == 'verbalized':
            # Verbalized mode: extract confidence from JSON response
            # print("\n[Verbalized Mode] Extracting confidence scores from response")
            for i in range(1, self.args.top_actions + 1):
                option_confidence_key = f"option{i}_confidence"
                if i in all_options and option_confidence_key in json_response:
                    confidence = json_response.get(option_confidence_key, 0)
                    # Convert confidence (0-100) to normalized probability (0-1)
                    normalized_prob = confidence / 100.0
                    logprob=0
                    # Convert to log probability for consistency
                    options_with_logits[i] = {
                        'action': all_options[i],
                        'token': i,
                        'logprob': logprob,
                        'normalized_prob': normalized_prob,
                        'confidence': confidence
                    }
                    # print(f"  Option {i}: confidence={confidence}%, normalized_prob={normalized_prob:.4f}, logprob={logprob:.4f}")

            selected_prob = json_response.get(f"option{best_choice}_confidence", 0) / 100.0

        # Normalize all actions in options_with_logits
        from .utils.utils import normalize_action
        current_state = state_node.state
        for option_num, option_data in options_with_logits.items():
            if isinstance(option_data, dict) and 'action' in option_data:
                raw_action = option_data['action']
                try:
                    normalized_data = normalize_action(raw_action, current_state)
                    option_data['normalized_action'] = normalized_data.get('normalized_action', raw_action)
                    option_data['action_metadata'] = normalized_data
                except Exception as e:
                    print(f"Warning: Failed to normalize action '{raw_action}': {e}")
                    option_data['normalized_action'] = raw_action
                    option_data['action_metadata'] = None

        # Display normalized actions with reasoning (simplified)
        if options_with_logits:
            # print(f"\n{'-'*100}")
            # print("LLM GENERATED OPTIONS:")
            # print(f"{'-'*100}")
            for option_num in sorted(options_with_logits.keys()):
                option_data = options_with_logits[option_num]
                raw_action = option_data.get('action', '')
                normalized_action = option_data.get('normalized_action', raw_action)

                # print(f"  [{option_num}] {normalized_action}")

            # print(f"\n  Initial choice: Option {best_choice}")
            # print(f"{'-'*100}")

        # Generate memory_text2 from step summaries with action reasoning
        memory_parts = []
        for i, summary in enumerate(self.step_summaries):
            summary_text = summary
            # Add reasoning from corresponding game history entry
            if i < len(self.game_history):
                entry = self.game_history[i]
                reasoning = self._extract_action_reasoning(entry.get('full_response'), entry.get('action'))
                if reasoning:
                    summary_text += f"\nAction Reasoning: {reasoning}"
            memory_parts.append(f"{summary_text}")

        memory_text2 = "\n\n".join(memory_parts)
        if memory_text2 == "":
            memory_text2 = " "

        # Get screenshots directory - use the current run's exp_dir
        # Screenshot for current step has been pre-saved in get_action()
        screenshots_dir = None
        if hasattr(self, '_exp_args') and hasattr(self._exp_args, 'exp_dir'):
            from pathlib import Path
            screenshots_dir = str(Path(self._exp_args.exp_dir))
            print(f"[INFO] Using screenshots from: {screenshots_dir}")

        updated_options_with_logits = self.update_scores(state_node, options_with_logits, k=10, r=0.8, memory_text=memory_text2, current_url=url, screenshots_dir=screenshots_dir)

        # Get normalized action for best option
        best_normalized_action = options_with_logits.get(best_choice, {}).get('normalized_action', action_text)

        print("\n" + "="*100)
        print("ACTION SELECTION")
        print("="*100)
        print(f"LLM Initial Choice: {best_normalized_action}")

        if updated_options_with_logits.items():
            best_corrected_option = max(updated_options_with_logits.items(), key=lambda x: x[1].get('corrected_logprob', float('-inf')))
            best_option_num, best_option_data = best_corrected_option
            action_text = best_option_data['action']
            best_corrected_normalized = best_option_data.get('normalized_action', action_text)

            if best_corrected_normalized != best_normalized_action:
                print(f"Memory-Adjusted Choice: {best_corrected_normalized} ← Changed by memory!")
            else:
                print(f"Memory-Adjusted Choice: {best_corrected_normalized} (same)")
        else:
            print(f"Memory-Adjusted Choice: {best_normalized_action} (no adjustment)")
        print("="*100)
        # self.add_to_memory(state_node.state, full_response)

        self._add_to_game_history(
            state_node.state,
            action_text,
            full_response,
            task_goal=state_node.instruction,
            url=url
        )

        return action_text.strip(), full_response


    def _parse_llm_response(self, full_response: str, top_actions: int = 3):
        """
        Parses the LLM's full string response to extract all options and the selected action.
        Returns all options dict and the selected action based on BEST ACTION number.
        
        Args:
            full_response: The full response string from the LLM
            top_actions: Number of action options to parse (default: 3)
            
        Returns:
            tuple: (options dict, selected_action string)
                options: Dictionary mapping option numbers to actions {1: "action1", 2: "action2", ...}
                selected_action: The best action chosen by the model
        """
        options = {}

        lines = full_response.strip().split('\n')
        for line in lines:
            line = line.strip()
                
                # Check for each possible option number
            for i in range(1, top_actions + 1):
                option_key = f"OPTION{i}:"
                if line.upper().startswith(option_key):
                    action = line.split(":", 1)[1].split("-")[0].strip()
                    options[i] = action
                    break
                
                # Check for BEST ACTION selection
            if line.upper().startswith("BEST ACTION:"):

                choice_num = int(line.split(":", 1)[1].strip())
                if choice_num in options:
                    selected_action = options[choice_num]
                        

        return options, selected_action


    def _add_to_game_history(self, state, action, full_response, task_goal=None, reward=None, score=None, url=None):
        self.game_history.append({
            "state": state,
            "action": action,
            "full_response": full_response,
            "reward": reward,
            "score": score,
            "url": url
        })

        step_index = len(self.game_history) - 1

        # Generate step summary
        from .utils.utils import generate_single_step_summary, normalize_action

        # Normalize action before generating summary
        normalized_action = action
        if action and state:
            try:
                normalized_data = normalize_action(action, state)
                normalized_action = normalized_data.get('normalized_action', action)
            except Exception as e:
                print(f"Warning: Failed to normalize action for summary: {e}")
                normalized_action = action

        summary = generate_single_step_summary(
            step_index=step_index,
            state=state,
            action=normalized_action,
            task_goal=task_goal,
            llm_model=getattr(self, 'llm_extract', self.args.llm_model),
            temperature=0.1,
            max_tokens=1000
        )
        self.step_summaries.append(summary)


    def update_game_history_reward(self, reward, score):
        """Update the last entry in game history with reward and score"""
        if self.game_history and len(self.game_history) > 0:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score
            if self.enable_cross_mem:
                self._recent_scores.append(score or 0)

    
    def _detect_victory_from_observation(self, final_observation_text: str) -> bool:
        if not final_observation_text or not isinstance(final_observation_text, str):
            return False
        text = final_observation_text.lower()
        # Heuristics for victory/credits screens common in text adventures
        victory_keywords = [
            "you have won", "victory", "congratulations", "congrats", "credits", "the end", "you win",
            # game-specific hints seen in logs
            "info room", "promoted", "win 310", "win 360"
        ]
        defeat_keywords = ["die", "died", "death", "killed", "game over", "defeat"]
        if any(k in text for k in victory_keywords) and not any(k in text for k in defeat_keywords):
            return True
        return False
    
    def extract_quoted_numbers(self, text: str):
        pattern = r"'(\d+)'"
        return re.findall(pattern, text)
    
    def find_elements(self, text: str, tag: str = None, with_id: int = None):
        results = []
        pattern = r"\[(\d+)\]\s+(\w+)\s+'(.*?)'"
        for line in text.splitlines():
            match = re.match(pattern, line.strip())
            if match:
                node_id, node_tag, node_value = match.groups()
                node_id = int(node_id)
                if (tag is None or node_tag == tag) and (with_id is None or node_id == with_id):
                    results.append({
                        "id": node_id,
                        "tag": node_tag,
                        "value": node_value
                    })
        return results


    def get_action(self, obs: dict) -> tuple[str, dict]:
        print("\n" + "="*100)
        print(f"STEP {len(self.game_history)}")
        print("="*100)

        # Clean up old screenshots at the start of the episode (step 0)
        if hasattr(self, '_exp_args') and hasattr(self._exp_args, 'exp_dir'):
            current_step = len(self.game_history)

            # Clean up old screenshots only at step 0
            if current_step == 0:
                import glob
                screenshot_pattern = str(Path(self._exp_args.exp_dir) / "screenshot_step_*.png")
                old_screenshots = glob.glob(screenshot_pattern)
                if old_screenshots:
                    print(f"[INFO] Cleaning up {len(old_screenshots)} old screenshots from previous runs")
                    for old_screenshot in old_screenshots:
                        try:
                            os.remove(old_screenshot)
                        except Exception as e:
                            print(f"[Warning] Failed to remove {old_screenshot}: {e}")

            # Pre-save current step's screenshot for trajectory context extraction
            # BrowserGym normally saves screenshots AFTER get_action returns (in save_step_info),
            # but we need it DURING generate_action for extract_effective_trajectory_context
            screenshot_path = Path(self._exp_args.exp_dir) / f"screenshot_step_{current_step}.png"
            current_screenshot = obs.get('screenshot')
            if current_screenshot is not None:
                try:
                    from PIL import Image
                    import numpy as np
                    if isinstance(current_screenshot, np.ndarray):
                        img = Image.fromarray(current_screenshot)
                        img.save(screenshot_path)
                        print(f"[INFO] Pre-saved screenshot for step {current_step}")
                except Exception as e:
                    print(f"[Warning] Failed to pre-save screenshot: {e}")

        # web_text = obs['pruned_html']
        web_text = obs['axtree_txt']

        # Get URL from observation
        url = obs.get('url', 'about:blank')
        print(f"URL: {url}")
        print(f"Task: {obs['goal']}")

        self.current_task_goal = obs['goal']

        # Get screenshot if use_screenshot_action is enabled
        screenshot = None
        if hasattr(self.args, 'use_screenshot_action') and self.args.use_screenshot_action:
            screenshot = obs.get('screenshot')

        state_node = StateNode(state=web_text, instruction=obs['goal'])
        action, raw_llm_output = self.generate_action(state_node, url=url, screenshot=screenshot)
        element_id = self.extract_quoted_numbers(action)
        target_element = self.find_elements(web_text, with_id=int(element_id[0])) if element_id else []
        
        # Extract JSON from raw_llm_output using robust extraction
        raw_llm_output_dict = {}
        if raw_llm_output and isinstance(raw_llm_output, str):
            raw_llm_output_dict = extract_json_from_response(raw_llm_output)

        ans_dict = {
            'think': raw_llm_output_dict.get('reasoning', '') if isinstance(raw_llm_output_dict, dict) else '',
            'action': action,
            'target_element': target_element
        }

        # print(f"\n→ FINAL ACTION: {action_display}")
        # print("="*100)

        # For debugging, save web info to temp file
        # os.makedirs("./temp", exist_ok=True)
        # log_path = "./temp/html_pruned_seq.json"

        # if os.path.exists(log_path):
        #     with open(log_path, "r", encoding="utf-8") as f:
        #         try:
        #             seq = json.load(f)
        #         except json.JSONDecodeError:
        #             seq = []
        # else:
        #     seq = []

        # seq.append(obs.get("html_pruned", ""))

        # with open(log_path, "w", encoding="utf-8") as f:
        #     json.dump(seq, f, ensure_ascii=False, indent=2)

        return action, ans_dict

    def calculate_exploration_probability(self, nearest_trajectories, action_rewards):
        """
        Calculate adaptive exploration probability based on the existence of stable high-reward actions.

        Key idea: If we have actions that are:
        1. Executed frequently (enough samples)
        2. Consistently high-performing (low variance, high mean)
        Then we should exploit more (explore less).

        Otherwise, we should explore more to find better actions.

        Args:
            nearest_trajectories: List of similar trajectories retrieved
            action_rewards: Dict mapping actions to their reward lists

        Returns:
            float: Exploration probability between 0.05 and 0.8
        """
        import numpy as np

        if not action_rewards:
            # No data yet, explore maximally
            return 0.8

        # Analyze each action's stability and performance
        action_scores = []
        for action, rewards in action_rewards.items():
            n_samples = len(rewards)
            if n_samples < 2:
                # Too few samples to judge stability, skip
                continue

            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)

            # Coefficient of variation (CV): std/mean, measures relative stability
            # Lower CV = more stable, but only meaningful for positive rewards
            if mean_reward > 0:
                cv = std_reward / mean_reward
            else:
                cv = float('inf')  # Unstable or bad action

            # Confidence score for this action:
            # - High mean reward (good performance)
            # - Low coefficient of variation (stable)
            # - Sufficient samples (reliable)

            # Normalize components
            performance_score = max(0, mean_reward)  # Raw performance
            stability_score = 1.0 / (1.0 + cv)  # High when CV is low (stable)
            sample_confidence = max(n_samples / 4.0, 1)  # Confidence increases with samples

            # Combined confidence: all three factors matter
            action_confidence = performance_score * stability_score * sample_confidence

            action_scores.append({
                'action': action,
                'confidence': action_confidence,
                'mean': mean_reward,
                'std': std_reward,
                'cv': cv,
                'n_samples': n_samples
            })

        if not action_scores:
            # No reliable actions found, explore
            print("[Exploration] No stable actions found -> High exploration (0.80)")
            return 0.8

        # Find the best action's confidence
        best_action = max(action_scores, key=lambda x: x['confidence'])
        max_confidence = best_action['confidence']

        # Exploration probability inversely related to max confidence
        # - Low confidence (uncertain actions) -> High exploration (0.8)
        # - High confidence (stable high-reward action exists) -> Low exploration (0.05)

        # Sigmoid-like mapping for smooth transition
        # confidence range: 0 to ~10+ (performance * stability * sample_confidence)
        exploration_prob = 1 / (1 + max_confidence/2)

        # Clamp to [0.05, 0.8]
        exploration_prob = max(0.05, min(0.8, exploration_prob))/2

        print(f"[Exploration] Best action '{best_action['action']}': "
              f"mean={best_action['mean']:.2f}, std={best_action['std']:.2f}, "
              f"cv={best_action['cv']:.2f}, samples={best_action['n_samples']}, "
              f"confidence={max_confidence:.2f} -> exploration_prob={exploration_prob:.3f}")

        return exploration_prob
