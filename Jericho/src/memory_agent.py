import math
import random
import os
import time
from .openai_helpers import chat_completion_with_retries
from .cross_episode_memory import CrossEpisodeMemory
from .utils import generate_trajectory_summary, generate_history_summary, generate_progress_analysis_with_recommendation

class MemoryAgent:
    def __init__(self, args, guiding_prompt: str = None):
        self.guiding_prompt = guiding_prompt or "Explore the environment and try to maximize your score."
        self.memory = [] # Used by agent
        self.game_history = [] # Used by evolutionary LLM
        self.args = args
        self.best_score = float("-inf")
        self.enable_cross_mem = getattr(self.args, 'enable_cross_mem', True)
        self.update_guiding_prompt = getattr(self.args, 'update_guiding_prompt', True)
        self.exploration_rate = getattr(self.args, 'exploration_rate', 0.65)
        self.exploration_alpha = getattr(self.args, 'exploration_alpha', 1.0)
        self.use_history_prompt = getattr(self.args, 'use_history_prompt', True)

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
            llm_model = getattr(self.args, 'llm_model')
            eval_llm_model = getattr(self.args, 'eval_llm_model', None)
            self.cross_mem = CrossEpisodeMemory(
                self.cross_mem_dir,
                gamma=gamma,
                llm_model=llm_model,
                eval_llm_model=eval_llm_model
            )

            # Load the latest guiding prompt from saved history if available and if update is enabled
            if self.update_guiding_prompt:
                latest_prompt = self.cross_mem.load_latest_guiding_prompt()
                if latest_prompt:
                    self.guiding_prompt = latest_prompt
            else:
                print(f"[INFO] Guiding prompt update is disabled. Using initial prompt: '{self.guiding_prompt}'")
        else:
            self.cross_mem_dir = None
            self.cross_mem = None


        # Simple loop detection buffer for scores
        self._recent_scores = []
    def add_to_memory(self, state, response):
        memory_entry = {"state": state, "response": response}
        self.memory.append(memory_entry)
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)  # Remove oldest entry if exceeding max_memory
    
    def _format_memory_for_prompt(self, state_node):
        current_history = self.game_history.copy()
        current_history.append({'state': state_node.state, 'action': ''})

        
        # Use the new centralized function to generate summary
        summary = generate_trajectory_summary(
            game_history=current_history,
            llm_model=self.args.llm_model,
            temperature=0.8,
            max_tokens=1000
        )
        # print("============summary=============")
        # print(summary)
        if summary:
            return summary
        else:
            return ""
    
    def start_episode(self):
        """
        """
        self.memory = []
        self.game_history = []
        self._recent_scores = []
        print(f"Using initial prompt: '{self.guiding_prompt}'")

        # Save the guiding prompt at the start of each episode
        if self.enable_cross_mem and self.cross_mem:
            self.cross_mem.save_guiding_prompt(self.guiding_prompt)

    def end_episode(self, state, score):
        """
        End an episode: update the current node's score and game history.
        """
        print(f"Ending episode with score: {score}.")
                    # Store complete episode in cross-episode memory
        if self.game_history:
            # Determine if episode was successful (score > 0 or won)
            success = self._detect_victory_from_observation(state)
            self.cross_mem.add_episode(
                    game_history=self.game_history,
                    state=state,
                    final_score=score,
                    success=success
                )

            # Generate prompt update recommendation based on the episode
            if self.enable_cross_mem and self.update_guiding_prompt:
                recommendation = self.cross_mem.generate_prompt_update(
                    game_history=self.game_history,
                    final_score=score,
                    success=success,
                    current_prompt=self.guiding_prompt,
                    use_history=self.use_history_prompt
                )

                # Update the guiding prompt with the recommendation
                new_prompt = recommendation.get('recommended_prompt', self.guiding_prompt)
                if new_prompt != self.guiding_prompt:
                    print(f"\n{'*'*60}")
                    print("UPDATING GUIDING PROMPT")
                    print(f"{'*'*60}")
                    print(f"Old: {self.guiding_prompt}")
                    print(f"New: {new_prompt}")
                    print(f"{'*'*60}\n")
                    self.guiding_prompt = new_prompt
                else:
                    print("\nGuiding prompt remains unchanged.\n")
            elif self.enable_cross_mem and not self.update_guiding_prompt:
                print("\n[INFO] Guiding prompt update is disabled. Skipping prompt update.\n")

        victory = self._detect_victory_from_observation(state)    
    
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
            # Use exploration_rate parameter to control the confidence calculation
            # Higher exploration_rate -> higher confidence -> lower exploration probability
            action_confidence = (performance_score * stability_score) ** self.exploration_rate * sample_confidence

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
        # Adjusted: Using max_confidence instead of max_confidence/2 for faster convergence
        exploration_prob = 1 / (1 + max_confidence)

        # Clamp to [0.05, 0.8]
        exploration_prob = max(0.05, min(0.8, exploration_prob))

        print(f"[Exploration] Best action '{best_action['action']}': "
              f"mean={best_action['mean']:.2f}, std={best_action['std']:.2f}, "
              f"cv={best_action['cv']:.2f}, samples={best_action['n_samples']}, "
              f"confidence={max_confidence:.2f} -> exploration_prob={exploration_prob:.3f}")

        return exploration_prob

    def update_scores(self, state_node, options_with_logits, k, r, memory_text, info=None):
        nearest_trajectories = self.cross_mem.retrieve_similar(
                game_history=self.game_history,
                current_state=state_node.state,
                current_summary=memory_text,
                k=k,
                r=r,
                info=info
            )
            
        if not nearest_trajectories:
            return {}
            
        # Add actions from nearest_trajectories to options_with_logits
        if not options_with_logits:
            options_with_logits = {}

        # Get existing actions
        existing_actions = set()
        for option_data in options_with_logits.values():
            if isinstance(option_data, dict) and 'action' in option_data:
                existing_actions.add(option_data['action'])

        # Aggregate action rewards from nearest_trajectories
        action_rewards = {}
        for sim, discounted_return, result_dict in nearest_trajectories:
            action = result_dict.get('action', '').strip()
            discounted_reward = result_dict.get('discounted_reward', 0)
            if action not in action_rewards:
                action_rewards[action] = []
            action_rewards[action].append(discounted_reward)
        
        for action in action_rewards:
            found = False
            for option_data in options_with_logits.values():
                if isinstance(option_data, dict) and option_data.get('action', '') == action:
                    found = True
                    break
            if not found:
                rewards = action_rewards[action]
                if len(rewards) >= 1 and sum(rewards) / len(rewards) > 0:
                    if options_with_logits:
                        new_option_num = max(options_with_logits.keys()) + 1
                    else:
                        new_option_num = 1
                    options_with_logits[new_option_num] = {
                        'action': action,
                        'normalized_prob': 0
                    }
        # Calculate average discounted reward for each action
        action_avg_rewards = {}
        for action, rewards in action_rewards.items():
            if rewards:  # Ensure not empty
                action_avg_rewards[action] = sum(rewards) / len(rewards)
            else:
                action_avg_rewards[action] = 0
        
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
        exploration_prob = self.calculate_exploration_probability(nearest_trajectories, action_rewards)

        alpha = self.exploration_alpha
        new_actions_added = False
        for option_num, option_data in options_with_logits.items():
            action = option_data['action']
            if action not in action_avg_rewards:
                rad = random.random()
                print(f"action: {action} rad:{rad}")
                if rad < exploration_prob:
                    # UCB-style exploration bonus: decreases as sample count increases
                    if count > 0:
                        action_avg_rewards[action] = overall_avg_reward + alpha / count
                    else:
                        action_avg_rewards[action] = overall_avg_reward + alpha
                else:
                    action_avg_rewards[action] = 0
                new_actions_added = True

        # If new actions were added, recalculate overall_avg_reward
        if new_actions_added:
            all_avg_rewards = list(action_avg_rewards.values())
            overall_avg_reward = sum(all_avg_rewards) / len(all_avg_rewards) if all_avg_rewards else 0

        # Calculate advantage value for each action
        action_advantages = {}
        for action, avg_reward in action_avg_rewards.items():
            action_advantages[action] = avg_reward - overall_avg_reward
        
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
        
        print(action_rewards)
        # Correct logit values in options_with_logits
        updated_options = {}
        if options_with_logits and normalized_advantages:
            print(f"\n=== Action Advantage Analysis & Logit Correction ===")
            print(f"Overall action average reward baseline: {overall_avg_reward:.4f}")
            
            for option_num, option_data in options_with_logits.items():
                if isinstance(option_data, dict) and 'action' in option_data:
                    action = option_data['action']
                    
                    # Get normalized advantage value
                    normalized_advantage = normalized_advantages.get(action, 0)
                    raw_advantage = action_advantages.get(action, 0)
                    avg_reward = action_avg_rewards.get(action, 0)

                    # Calculate episode-based weight for normalized advantage
                    # Weight increases from 1.0 (episode 1) to 1.5 (episode 50)
                    # Formula: weight = 1.0 + (current_episode / 50) * 0.5
                    # Clamped to max of 1.5 for episodes beyond 50

                    if self.enable_cross_mem and self.cross_mem:
                        current_episode = self.cross_mem.current_episode_number
                        episode_weight = min(1.0 + (current_episode / 50.0) * 0.5, 1.5)
                    else:
                        episode_weight = 1.0
                    # episode_weight = 1.5

                    # Apply episode weight to normalized advantage
                    weighted_normalized_advantage = normalized_advantage * episode_weight

                    # Correct logit (add weighted normalized advantage value)
                    normalized_prob = option_data.get('normalized_prob', 0)
                    corrected_logprob = normalized_prob + weighted_normalized_advantage
                    
                    # Create corrected option data
                    updated_options[option_num] = {
                        'action': action,
                        'normalized_advantage': normalized_advantage,
                        'corrected_logprob': corrected_logprob,
                        'avg_reward': avg_reward,
                        'raw_advantage': raw_advantage,
                        'token': option_data.get('token', ''),
                        'normalized_prob': option_data.get('normalized_prob', 0)
                    }
                    
                    print(f"  Action: {action}")
                    print(f"    Average reward: {avg_reward:.4f} | Raw advantage: {raw_advantage:.4f}")
                    print(f"    Normalized advantage: {normalized_advantage:.4f}")
                    print(f"    Original logprob: {normalized_prob:.4f} -> Corrected: {corrected_logprob:.4f}")
                else:
                    # If no matching advantage value, keep original value
                    updated_options[option_num] = option_data.copy()
                    updated_options[option_num]['corrected_logprob'] = option_data.get('logprob', 0)
                    updated_options[option_num]['normalized_advantage'] = 0
            print("="*50)
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


    def get_prompts(self, state_node, info=None):
        # Extract valid_actions and current_inventory from info
        valid_actions = info.get('valid', []) if (info and self.args.use_valid_actions) else None
        current_inventory = info.get('inv', None) if info else None

        # memory_text = self._format_memory_for_prompt(state_node)
        if self.game_history:
            summary, memory_text = generate_history_summary(self.game_history, state_node.state, self.args.llm_model, 0, 3000, current_inventory=current_inventory)
            # memory_text = generate_progress_analysis_with_recommendation(self.game_history, state_node.state, self.args.llm_model, 0, 3000)
        else:
            summary = ""
            memory_text = "No game history."
            # Add inventory even if no game history
            if current_inventory:
                memory_text += f"\n\n[INVENTORY]\n{current_inventory}"
        # analysis_text = generate_progress_analysis_with_recommendation(self.game_history, state_node.state, self.args.llm_model, 0, 3000)
        # print("============analysis_text=============")
        # print(analysis_text)
        sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the game state and memory of past interactions.

**EXPLORATION PRIORITY**: When you arrive at a NEW location you haven't fully explored before, you MUST thoroughly explore it FIRST before leaving."""
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        # Add reminder about action list order only when valid_actions are provided
        if valid_actions:
            sys_prompt += """\n\n**CRITICAL CONSTRAINT**: When REFERENCE ACTIONS are provided, you MUST ONLY choose actions from that list. Any action not in the REFERENCE ACTIONS list is INVALID and will fail. Do NOT create custom actions. The list is unordered - position doesn't indicate quality."""

        recent_history = ""
        if self.game_history:
            # Only keep the last 10 steps
            recent_game_history = self.game_history[-20:]
            start_index = max(0, len(self.game_history)-20)
            for idx, entry in enumerate(recent_game_history):
                actual_step = start_index + idx
                recent_history += f"Step {actual_step}:\n"
                recent_history += f"State: {entry.get('state', '')}\n"
                recent_history += f"Action: {entry.get('action', '')}\n"
                if entry.get('reward') is not None:
                    recent_history += f"Reward: {entry.get('reward', 0)}\n"
                recent_history += "\n"

        # Add valid actions section if available
        valid_actions_text = ""
        if valid_actions:
            valid_actions = valid_actions[:]  # make a copy to avoid side effects
            rng = random.Random(time.time_ns())
            rng.shuffle(valid_actions)
            print(f"Valid actions (shuffled): {valid_actions}")
            valid_actions_text = f"\nREFERENCE ACTIONS (ONLY VALID ACTIONS):\n{valid_actions}\n\n**STRICT REQUIREMENT**: These are the ONLY valid actions for the current state. You MUST select your actions EXCLUSIVELY from this list. Any action not in this list is INVALID and will be rejected by the game. Do NOT create, modify, or suggest any custom actions.**\n"

        # Check confidence mode from args
        confidence_mode = getattr(self.args, 'confidence_mode', 'logit')

        # Build response format based on confidence mode
        if confidence_mode == 'verbalized':
            # Build option fields with confidence
            option_fields = []
            for i in range(1, self.args.top_actions + 1):
                option_fields.append(f'    "reasoning{i}": "Why this action makes sense",')
                option_fields.append(f'    "option{i}": "action command",')
                option_fields.append(f'    "confidence{i}": 80,')
            option_format = '\n'.join(option_fields)

            user_prompt = f"""
GAME HISTORY:
{summary}
{memory_text}

RECENT STEPS:
{recent_history}

CURRENT STATE: {state_node.state}

TASK:
1. Analyze your progress: What have you achieved? What's your next objective?
2. **MANDATORY: Check the REFERENCE ACTIONS list below - you MUST ONLY select from this list**
3. Propose {self.args.top_actions} different actions with reasoning
4. For EACH action, provide your confidence as an integer from 0 to 100 (e.g., 80 means 80% confidence that this action will help achieve the goal)

RESPONSE FORMAT (JSON):
{{
    "progress_analysis": "What you've achieved and current challenges",
    "next_objective": "Your next goal",
{option_format}
    "best_action": 1
}}

IMPORTANT:
- **ABSOLUTE REQUIREMENT**: ALL actions (option1, option2, etc.) MUST be selected EXACTLY from the REFERENCE ACTIONS list below
- Actions NOT in the REFERENCE ACTIONS list are INVALID and will cause the game to fail
- DO NOT create custom actions, DO NOT modify actions from the list, DO NOT combine actions
- **CRITICAL**: The sum of all confidence values (confidence1 + confidence2 + ... + confidence{self.args.top_actions}) MUST equal 100
- The confidence values MUST have meaningful differences between them
- Higher confidence means you believe this action is more likely to succeed
- Lower confidence means you believe this action is less likely to succeed
- Pay attention to game hints and clues in state descriptions
- Don't repeat failed actions or create loops (e.g., north→south→north)

{valid_actions_text}
"""
        else:
            # Original format for logit mode
            user_prompt = f"""
GAME HISTORY:
{summary}
{memory_text}

RECENT STEPS:
{recent_history}

CURRENT STATE: {state_node.state}

TASK:
1. Analyze your progress: What have you achieved? What's your next objective?
2. **MANDATORY: Check the REFERENCE ACTIONS list below - you MUST ONLY select from this list**
3. Propose {self.args.top_actions} different actions with reasoning

RESPONSE FORMAT (JSON):
{{
    "progress_analysis": "What you've achieved and current challenges",
    "next_objective": "Your next goal",
    "reasoning1": "Why this action makes sense",
    "option1": "action command",
    "reasoning2": "Why this action makes sense",
    "option2": "action command",{' "reasoning3": "...",' if self.args.top_actions >= 3 else ''}{' "option3": "...",' if self.args.top_actions >= 3 else ''}
    "best_action": 1
}}

KEY RULES:
- **ABSOLUTE REQUIREMENT**: ALL actions (option1, option2, etc.) MUST be selected EXACTLY from the REFERENCE ACTIONS list below
- Actions NOT in the REFERENCE ACTIONS list are INVALID and will cause the game to fail
- DO NOT create custom actions, DO NOT modify actions from the list, DO NOT combine actions
- Pay attention to game hints and clues in state descriptions
- Don't repeat failed actions or create loops (e.g., north→south→north)
- Use single-word object names (e.g., "examine book" not "examine the old book")

{valid_actions_text}
"""
        return sys_prompt, user_prompt, memory_text

    # Generates the next action from the LLM based on its memory and the current state node.
    def generate_action(self, state_node, info=None):
        sys_prompt, user_prompt, memory_text = self.get_prompts(state_node, info=info)

        # Check confidence mode
        confidence_mode = getattr(self.args, 'confidence_mode', 'logit')

        # Dynamically create JSON Schema to force final choice to be valid option numbers only
        properties = {}
        required_fields = []
        valid_choices = []

        # Add progress analysis fields
        properties["progress_analysis"] = {
            "type": "string",
            "description": "Analysis of achievements and progress so far"
        }
        properties["next_objective"] = {
            "type": "string",
            "description": "Overall objective for the next steps"
        }
        required_fields.extend(["progress_analysis", "next_objective"])

        # Dynamically generate reasoning and option fields
        for i in range(1, self.args.top_actions + 1):
            reasoning_key = f"reasoning{i}"
            option_key = f"option{i}"

            properties[reasoning_key] = {
                "type": "string",
                "description": f"Reasoning for option {i}"
            }
            properties[option_key] = {
                "type": "string",
                "description": f"Option {i} possible action"
            }
            required_fields.append(reasoning_key)
            required_fields.append(option_key)
            valid_choices.append(i)

            # Add confidence field for verbalized mode
            if confidence_mode == 'verbalized':
                confidence_key = f"confidence{i}"
                properties[confidence_key] = {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": f"Confidence for option {i} (0-100)"
                }
                required_fields.append(confidence_key)

        # Add best_action field

        properties["best_action"] = {
            "type": "number",
            "minimum": 1,
            "maximum": self.args.top_actions,
            "description": f"The number of the best option (must be one of: {valid_choices})"
        }
        required_fields.extend([ "best_action"])
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "game_action_choice",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required_fields,
                    "additionalProperties": False
                }
            }
        }
        
        res_obj = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=3000,
            temperature=self.args.llm_temperature,
            top_logprobs=self.args.top_actions,
            response_format=response_format
        )

        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and res_obj.choices[0].message:
            full_response = res_obj.choices[0].message.content

            # Parse JSON response
            import json
            import re

            # Clean up markdown code blocks if present
            cleaned_response = full_response.strip()
            if cleaned_response.startswith('```'):
                # Remove ```json or ``` at the start and ``` at the end
                cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response)
                cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)

            try:
                json_response = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {full_response}")
                print(f"Cleaned response: {cleaned_response}")
                # Return default behavior
                return "look", full_response

            # Extract and print progress analysis
            progress_analysis = json_response.get("progress_analysis", "No analysis provided")
            next_objective = json_response.get("next_objective", "No objective specified")

            print("\n" + "="*50)
            print("=== PROGRESS ANALYSIS ===")
            print("="*50)
            print(f"Achievements: {progress_analysis}")
            print(f"\nNext Objective: {next_objective}")
            print("="*50 + "\n")

            # Extract all options with their reasoning
            all_options = {}
            all_reasonings = {}
            all_confidences = {}

            for i in range(1, self.args.top_actions + 1):
                reasoning_key = f"reasoning{i}"
                option_key = f"option{i}"
                reasoning = json_response.get(reasoning_key, "No reasoning provided")
                option = json_response.get(option_key, "look")
                all_reasonings[i] = reasoning
                all_options[i] = option

                # Extract confidence if in verbalized mode
                if confidence_mode == 'verbalized':
                    confidence_key = f"confidence{i}"
                    confidence = json_response.get(confidence_key, 50)  # Default to 50 if not provided
                    all_confidences[i] = confidence

            # Extract logprobs information for all tokens OR use verbalized confidence
            options_with_logits = {}
            option_confidences = {}  # Store confidence as 0-1 value for each option

            if confidence_mode == 'verbalized':
                # Use verbalized confidence values
                for i in all_options.keys():
                    confidence = all_confidences.get(i, 50)
                    # Normalize confidence from 0-100 to a probability-like value
                    normalized_prob = confidence / 100.0
                    options_with_logits[i] = {
                        'action': all_options[i],
                        'token': i,
                        'normalized_prob': normalized_prob,
                        'confidence': confidence
                    }
                    option_confidences[i] = normalized_prob

            elif hasattr(res_obj.choices[0], 'logprobs') and res_obj.choices[0].logprobs and res_obj.choices[0].logprobs.content:
                # Use logprobs (original method)
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
                                option_confidences[option_num] = normalized_prob
                        except (ValueError, TypeError):
                            # Skip tokens that cannot be converted to integers
                            continue

                selected_prob = math.exp(last_token_logprob.logprob)

            # # Now print all options with their reasoning and confidence
            # print("\n=== Generated Options with Reasoning ===")
            # for i in range(1, self.args.top_actions + 1):
            #     reasoning = all_reasonings.get(i, "No reasoning provided")
            #     option = all_options.get(i, "look")
            #     confidence_val = option_confidences.get(i, None)

            #     print(f"Option {i}:")
            #     print(f"  Reasoning: {reasoning}")
            #     print(f"  Action: {option}")
            #     if confidence_val is not None:
            #         print(f"  Confidence: {confidence_val:.3f}")
            #     else:
            #         print(f"  Confidence: N/A")

            best_choice = json_response.get("best_action", 1)
            action_text = all_options.get(best_choice, "look")
            best_confidence = option_confidences.get(best_choice, None)
            # print(f"\nBest Action Selected: Option {best_choice} - {action_text}")
            # if best_confidence is not None:
            #     print(f"Best Action Confidence: {best_confidence:.3f}")
            # print("="*50 + "\n")
                
        else:
            full_response = ""
            action_text = "look" # Default action
        updated_options_with_logits = self.update_scores(state_node, options_with_logits, k=self.args.retrieval_top_k, r=self.args.retrieval_threshold, memory_text=memory_text, info=info)
        if updated_options_with_logits.items():
            best_corrected_option = max(updated_options_with_logits.items(), key=lambda x: x[1].get('corrected_logprob', float('-inf')))
            best_option_num, best_option_data = best_corrected_option
            action_text = best_option_data['action']
        # self.add_to_memory(state_node.state, full_response)
        self._add_to_game_history(state_node.state, action_text, full_response)

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
        selected_action = "look"  # Default action
        
        if not full_response or not isinstance(full_response, str):
            return options, selected_action

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

    def _add_to_game_history(self, state, action, full_response, reward=None, score=None):
        self.game_history.append({
            "state": state,
            "action": action,
            "full_response": full_response,
            "reward": reward,
            "score": score
        })

    def update_game_history_reward(self, reward, score):
        """Update the last entry in game history with reward and score"""
        if self.game_history and len(self.game_history) > 0:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score
            if self.enable_cross_mem:
                # For positives: if delta_score>0, persist (state->action)
                if len(self._recent_scores) == 0:
                    prev = 0
                else:
                    prev = self._recent_scores[-1]
                delta = (score or 0) - (prev or 0)
                self._recent_scores.append(score or 0)
                # Store step data for end-of-episode batch storage
                # (removed individual step storage)

    
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
