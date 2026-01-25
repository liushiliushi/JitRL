import math
import os
import random
from .openai_helpers import chat_completion_with_retries
from .cross_episode_memory import CrossEpisodeMemory

class OurAgent:
    def __init__(self, args, guiding_prompt: str = None): 
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.memory = [] # Used by agent
        self.game_history = [] # Used by evolutionary LLM
        self.args = args
        self.user_prompt_evo = ""
        
        # Each node is a dict with keys: prompt, state_extractor, game_history, score, parent_idx, children_idxs
        self.nodes = []
        
        self.code = """
def extract_state(game_history):
    return "Game in progress."
"""

        self.best_node_idx = None
        self.best_score = float("-inf")
        self.last_episode_score = None
        self.freeze_on_win = getattr(self.args, 'freeze_on_win', True)
        self.win_freeze_threshold = getattr(self.args, 'win_freeze_threshold', 0)
        self.force_best_after_drop = getattr(self.args, 'force_best_after_drop', True)
        self.drop_threshold = getattr(self.args, 'drop_threshold', 50)

        # Auto-detection and latch for freezing after a detected win
        self.auto_freeze_on_win = True
        self.is_frozen = False
        # Internal: whether we auto-set the freeze threshold
        self._auto_set_threshold = False
        # Track last episode victory result
        self.last_episode_was_victory = False
        # After unfreezing due to missed win, force evolution in next start
        self.just_unfroze_due_to_missed_win = False

        # Cross-episode memory toggle
        self.enable_cross_mem = getattr(self.args, 'enable_cross_mem', True)

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
            max_trajectory_window = getattr(self.args, 'max_trajectory_window', 5)
            self.cross_mem = CrossEpisodeMemory(self.cross_mem_dir, gamma=gamma, max_trajectory_window=max_trajectory_window)
        else:
            self.cross_mem_dir = None
            self.cross_mem = None
        
        # Simple loop detection buffers
        self._recent_states = []
        self._recent_actions = []
        self._recent_scores = []
    
    def add_to_memory(self, state, response):
        memory_entry = {"state": state, "response": response}
        self.memory.append(memory_entry)
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)  # Remove oldest entry if exceeding max_memory
    
    def _format_memory_for_prompt(self):
        if not self.memory:
            return ""
            
        memory_text = "MEMORY (Recent few states and agent's responses):\n"
        for i, entry in enumerate(self.memory):
            memory_text += f"Memory {i+1}:\n"
            memory_text += f"STATE: {entry['state']}\n"
            if entry['response']:
                memory_text += f"AGENT'S RESPONSE: {entry['response']}\n"
        
        return memory_text

    def calculate_ucb(self, node_idx: int) -> float:
        """
        Calculate the UCB value for a node
        UCB = node's score + c * alpha^depth * sqrt(log(N_total / (1+N_children)))
        Where N_total is the length of the nodes list, alpha is the depth decay factor,
        And N_children is the number of children of this node
        """
        node = self.nodes[node_idx]
        num_children = len(node["children_idxs"])
        total_nodes = max(2, len(self.nodes))

        # If frozen (we detected a win), stop exploration entirely
        if self.is_frozen or (self.freeze_on_win and self.best_score is not None and self.win_freeze_threshold and self.best_score >= self.win_freeze_threshold):
            return node["score"]

        c = self.args.exploration_constant
        return node["score"] + c * (self.args.depth_constant ** node["depth"]) * math.sqrt(math.log(total_nodes) / (1 + num_children))
    
    def start_episode(self):
        """
        Select a node based on UCB, evolve the guiding prompt, and create a new node.
        If no nodes exist, use the initial guiding prompt.
        New node's score and history will be updated at the end of the episode.
        """
        self.memory = []
        self.game_history = []
        self._recent_states = []
        self._recent_actions = []
        self._recent_scores = []

        should_exploit_best = False
        # Condition 1: explicit win-freeze threshold reached in prior runs
        if self.freeze_on_win and self.win_freeze_threshold and self.best_score >= self.win_freeze_threshold:
            should_exploit_best = True
        # Condition 2: auto frozen due to detected win (no manual threshold needed)
        if self.is_frozen:
            should_exploit_best = True
        # Condition 3: large drop from best in last run
        elif self.force_best_after_drop and self.last_episode_score is not None and self.best_score - self.last_episode_score >= self.drop_threshold:
            should_exploit_best = True
        
        # If we just unfroze because the frozen prompt failed to win, we must evolve next
        if self.just_unfroze_due_to_missed_win:
            should_exploit_best = False

        if len(self.nodes) > 0:
            if should_exploit_best and self.best_node_idx is not None:
                parent_idx = self.best_node_idx
                parent_node = self.nodes[parent_idx]
                self.guiding_prompt = parent_node["prompt"]
                self.code = parent_node["code"]
                print(f"[OurAgent] Exploiting best node {parent_idx} (score={self.best_score}). No evolution.")
            else:
                parent_idx = max(range(len(self.nodes)), key=lambda i: self.calculate_ucb(i))
                parent_node = self.nodes[parent_idx]
                print(f"Parent node {parent_idx} at depth {parent_node['depth']} with UCB score: {self.calculate_ucb(parent_idx)}")
                neg_block = self._format_negative_block_for_evolve() if self.enable_cross_mem else ""
                self.guiding_prompt, self.code = self._evolve(parent_node["prompt"], parent_node["code"], parent_node["game_history"], neg_block=neg_block)
                print(f"Evolved guiding prompt and code: '{self.guiding_prompt}'")
                print(self.code)
        else:
            print(f"Using initial prompt and code: '{self.guiding_prompt}'")
            print(self.code)
            parent_idx = -1
            
        # Create a new node with evolved prompt as child of selected node
        new_node = {
            "prompt": self.guiding_prompt,
            "code": self.code,
            "depth": 0 if parent_idx < 0 else self.nodes[parent_idx]["depth"] + 1,
            "score": -1,
            "parent_idx": parent_idx,
            "children_idxs": [],
            "game_history": []
        }
            
        # Update parent's children list
        if parent_idx >= 0:
            self.nodes[parent_idx]["children_idxs"].append(len(self.nodes))
        self.nodes.append(new_node)
        
        # Reset this one-shot switch after we've chosen to evolve
        self.just_unfroze_due_to_missed_win = False
        
        # self.add_to_memory("=== START OF GAME ===", "")

    def end_episode(self, state, score):
        """
        End an episode: update the current node's score and game history.
        Store complete episode in cross-episode memory.
        """
        # self.add_to_memory("=== END OF GAME ===", "")
        self._add_to_game_history(state, '', '', '')
        self.nodes[-1]["game_history"] = self._format_game_history(self.game_history)
        self.nodes[-1]["score"] = score
        self.last_episode_score = score
        if score > self.best_score:
            self.best_score = score
            self.best_node_idx = len(self.nodes) - 1
            print(f"[OurAgent] New best score {self.best_score} at node {self.best_node_idx}.")

        # Store complete episode in cross-episode memory
        try:
            if self.game_history:
                # Use game_history instead of separate _recent_* lists
                # Calculate rewards and delta scores for each step
                rewards = []
                delta_scores = []
                aligned_scores = []
                aligned_states = []
                aligned_actions = []
                
                prev_score = 0
                for entry in self.game_history:
                    # Get score for this step, default to 0 if not available
                    current_score = entry.get('score', 0) or 0
                    delta = current_score - prev_score
                    
                    aligned_states.append(entry['state'])
                    aligned_actions.append(entry['action'])
                    rewards.append(delta)
                    delta_scores.append(delta)
                    aligned_scores.append(current_score)
                    prev_score = current_score
                
                # Determine if episode was successful (score > 0 or won)
                success = self._detect_victory_from_observation(state)
                
                self.cross_mem.add_episode(
                    states=aligned_states,
                    actions=aligned_actions,
                    rewards=rewards,
                    scores=aligned_scores,
                    delta_scores=delta_scores,
                    final_score=score,
                    success=success
                )
        except Exception as e:
            # Don't let memory storage errors break the game
            print(f"[Warning] Failed to store episode in cross-episode memory: {e}")

        # Evaluate victory on final observation and update frozen state
        victory = self._detect_victory_from_observation(state)
        self.last_episode_was_victory = victory
        if self.auto_freeze_on_win and victory:
            self.is_frozen = True
            # If no manual threshold provided, adopt the winning score as threshold for transparency
            if not self.win_freeze_threshold:
                self.win_freeze_threshold = score
                self._auto_set_threshold = True
            print(f"[OurAgent] Victory detected. Freezing exploration. Threshold set to {self.win_freeze_threshold}.")
        else:
            # If we were frozen but failed to win this episode, unfreeze and resume evolution
            if self.is_frozen and not victory:
                self.is_frozen = False
                if self._auto_set_threshold:
                    self.win_freeze_threshold = 0
                    self._auto_set_threshold = False
                # Ensure next episode evolves instead of re-exploiting immediately
                self.just_unfroze_due_to_missed_win = True
                print("[OurAgent] Missed win with frozen prompt. Unfreezing and resuming evolution.")

        # Persist any detected loop as negative memory at episode end (best-effort)
        if self.enable_cross_mem:
            loop_segment = self._detect_loop_segment()
            if loop_segment is not None:
                states_seg, actions_seg = loop_segment
                # Note: Loop detection logic preserved, but negative examples 
                # will be stored as part of complete episode in end_episode()

    def get_few_shot(self, state_node):
        try:
            # Pass the full trajectory history to retrieve_similar
            examples = self.cross_mem.retrieve_similar(
                query_states=self._recent_states + [state_node.state],
                query_actions=self._recent_actions,
                k=1
            )
        except Exception:
            examples = []
        if not examples:
            return "(none)", "(none)", 0, 0.0
        return examples[0]['step']['state'], examples[0]['step']['action'], examples[0]['reward'], examples[0]['similarity']

    
    def get_prompts(self, state_node):
        memory_text = self._format_memory_for_prompt()
        
        # Extract current state using the state extractor
        extracted_state = self._extract_current_state()
        state_summary = ""
        if extracted_state:
            state_summary = f"GAME STATE SUMMARY: {extracted_state}"

        # Few-shot from cross-episode positives
        few_shot = self._format_few_shot_from_cross_mem(self.game_history, state_node.state) if self.enable_cross_mem else "(disabled)"
        sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the game state and memory of past interactions."""
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        user_prompt = f"""
{state_summary}\n\nYour memory of the recent states and actions is: {memory_text}\n\n
Here are 2-3 successful examples from similar situations (learned across episodes):
{few_shot}

Your current state is: {state_node.state} \n\n
Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, northeast, up, etc.), examine X, say X, drop X, get X, open X, enter X, ask X about Y, look in X, give X to Y, and other context-specific commands. To avoid parsing errors, any such X or Y MUST be a *SINGLE WORD* that identifies an object in a way the game can understand. When stuck, explore all rooms and objects mentioned in room descriptions systematically and comprehensively. *DO NOT REPEAT* the same failed action multiple times, as it will not lead to new results. Do not use the "help" command.

Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

For example:
REASONING: I should examine the book to learn more about it.
ACTION: examine book
"""
        return sys_prompt, user_prompt, extracted_state

    # Generates the next action from the LLM based on its memory and the current state node.
    def generate_action(self, state_node, valid_actions=None):
        sys_prompt, user_prompt, extracted_state = self.get_prompts(state_node)
        
        res_obj = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=400,
            temperature=self.args.llm_temperature,
        )

        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and res_obj.choices[0].message:
            full_response = res_obj.choices[0].message.content
            action_text = self._parse_llm_response(full_response)
        else:
            print(f"Warning: LLM API call might have failed or returned empty. Defaulting action.")
            full_response = ""
            action_text = "look" # Default action
        state, action_text2, reward, simi = self.get_few_shot(state_node)
        action_text = action_text2 if simi > 0.95 and reward > 5 else action_text
        self.add_to_memory(state_node.state, full_response)
        self._add_to_game_history(state_node.state, action_text, full_response, extracted_state)
        
        # Track for loop detection
        self._recent_states.append(state_node.state)
        self._recent_actions.append(action_text)
        # score will be added on step via update_game_history_reward
        
        return action_text.strip(), full_response

    def _parse_llm_response(self, full_response: str):
        """
        Parses the LLM's full string response to extract action.
        """
        action_text = "look" # Default action

        if not full_response or not isinstance(full_response, str):
            return action_text

        lines = full_response.strip().split('\n')
        try:
            for line in lines:
                if line.upper().startswith("ACTION:"):
                    action_text = line.split(":", 1)[1].strip()
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Response was: '{full_response}'")

        return action_text
    
    def _add_to_game_history(self, state, action, full_response, extracted_state, reward=None, score=None):
        self.game_history.append({
            "state": state,
            "action": action,
            "full_response": full_response,
            "extracted_state": extracted_state,
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

    
    def _format_game_history(self, history):
        
        history_str = "GAME HISTORY:\n"
        for i, entry in enumerate(history):
            history_str += f"Step {i+1}:\n"
            history_str += f"STATE: {entry['state']}\n"
            if 'extracted_state' in entry and entry['extracted_state']:
                history_str += f"EXTRACTED STATE: {entry['extracted_state']}\n"
            if 'full_response' in entry and entry['full_response']:
                history_str += f"AGENT'S FULL RESPONSE: {entry['full_response']}\n"
            if 'action' in entry and entry['action']:
                history_str += f"ACTION TAKEN: {entry['action']}\n"
            history_str += "------------\n"
        
        return history_str
    
    def _evolve(self, cur_prompt, cur_code, cur_history_str, neg_block: str = ""):
        print(f"\nEvolving prompt. Current prompt: '{cur_prompt[:80]}...'\n\nCurrent state extractor code: '{cur_code[:80]}...'\n\n")

        sys_prompt_evo = "You are an expert at text adventure games. Your goal is to analyze the existing prompt, state extractor code (i.e. python code which outputs a concise summary of the game state to help the agent, using the game history as input), and game history, and generate a better prompt and state extractor code that will help an LLM agent achieve higher scores. Don't be overly concise; ignore oververbosity penalties."
        
        negative_section = ""
        if neg_block and self.enable_cross_mem:
            negative_section = f"\n\nAVOID THE FOLLOWING FAILURE PATTERNS (derived from prior episodes):\n{neg_block}\n\n"
        
        self.user_prompt_evo = f'''
Generate a new improved guiding prompt and state extractor code for a text adventure game agent. 

The LLM agent used the following guiding prompt (which may not be accurate; rewrite it completely if needed):
"{cur_prompt}"

Here is the history of that game session:
--- GAME HISTORY START ---
{cur_history_str}
--- GAME HISTORY END ---
{negative_section}
PART 1: Generate a new improved guiding prompt. Consider:
1. Identify useful actions that led to increases in score, or needed for progressing the game, ignoring useless actions. Give step-by-step instructions to perform these actions. ONLY give instructions that were strictly necessary for progressing the game or give rewards. Do not suggest possible future actions as they may not be correct.
2. Discourage actions that led to negative outcomes, getting stuck, or unproductive for too long.
3. When reaching the limit based on current game knowledge, list all possible rooms to search next, especially those that have been only lightly searched, and brainstorm possible next attempts, making clear that these are only guesses. Finally, suggest systematically exploring and interacting with rooms, objects, NPCs, inventory etc for clues.

PART 2: Generate a state extractor Python code in a <code>...</code> block that analyzes the game history log and summarizes what milestones the agent has completed so far, i.e. significant progression towards completing the game. This should be a Python function that:
1. Take the game history as input (a string with the log of the game states and agent's actions)
2. Extracts key information about the agent's current milestones reached (relevant to making progress in the game)
3. Returns a summary string of the current state (e.g., "Opened blue door.")

The state extractor can be tailored to the current game; it does not have to work for other games. Avoid complex code to avoid bugs; use simple checks like for particular *strings from the game environment* indicating that a certain milestone was completed, not the agent's actions or commands, since there are usually many ways to express an action. Be careful with using int() as numeric values are sometimes given in words. 
<code>
def extract_state(game_history):
    if "The blue door opens" in game_history:
        return "Opened blue door."
    else:
        return "Unknown state."
</code>

Format your response as follows with NO additional text. The function name MUST be extract_state, and should contain no comments.

[Your generated prompt here]
<code>
def extract_state(game_history):
    # [Return a string summarizing the current state]
</code>
'''
        try:
            response = chat_completion_with_retries(
                model=self.args.evolution_llm_model,
                sys_prompt=sys_prompt_evo,
                prompt=self.user_prompt_evo,
                max_tokens=3000,
                temperature=self.args.evol_temperature,
            )
            full_response = response.choices[0].message.content.strip()
            
            ret_prompt, ret_code = cur_prompt, cur_code
            new_prompt, new_code = self._parse_evolution_response(full_response)
            
            if new_prompt and len(new_prompt) > 10:
                ret_prompt = new_prompt
            else:
                print("Evolution LLM returned empty/short response, keeping current prompt")
            if new_code and self._validate_state_extractor(new_code) and len(new_code) > 10:
                ret_code = new_code
            else:                
                print("Evolution LLM returned invalid state extractor code, keeping current code")
            return ret_prompt, ret_code
                
        except Exception as e:
            print(f"Error during prompt evolution LLM call: {e}")
            return cur_prompt, cur_code  # Return current prompt if evolution fails
    
    def _parse_evolution_response(self, response):
        """
        Parse the response from the evolutionary LLM to extract:
        1. The new prompt
        2. The state extractor code
        
        Returns:
            tuple: (new_prompt, state_extractor_code)
        """
        # Extract state extractor code
        state_extractor_code = ""
        if "<code>" in response and "</code>" in response:
            code_start = response.find("<code>") + len("<code>")
            code_end = response.find("</code>")
            if code_start < code_end:
                state_extractor_code = response[code_start:code_end].strip()
                # Remove the code part from the response to get the prompt
                response = response[:response.find("<code>")].strip()
        
        # The remaining text is the prompt
        new_prompt = response.strip()
        
        return new_prompt, state_extractor_code
    
    def _validate_state_extractor(self, state_extractor_code):
        
        if not state_extractor_code:
            print("No valid state extractor code provided.")
            return False

        try:
            namespace = {}
            expected_fn_header = "def extract_state(game_history):"
            if expected_fn_header not in state_extractor_code:
                state_extractor_code = expected_fn_header + "\n    " + \
                    "# Default implementation if provided code lacks the proper function\n    " + \
                    "return \"No specific state extracted\"\n\n" + state_extractor_code
            
            exec(state_extractor_code, namespace)
            if "extract_state" in namespace and callable(namespace["extract_state"]):
                try:
                    namespace["extract_state"]("")
                    print("Updated state extractor code successfully.")
                    return True
                except Exception as e:
                    print(f"State extractor code failed basic test: {e}")
                    return False
            else:
                print("State extractor code does not contain valid extract_state function.")
                return False
        except Exception as e:
            print(f"Validation failed: {e}")
            return False
    
    def _extract_current_state(self):
        """
        Extract the current state from the game history using the state extractor code.
        Returns:
            str: The extracted state description or empty string if extraction fails.
        """
        if not hasattr(self, 'code') or not self.code:
            return ""
        
        try:
            namespace = {}
            exec(self.code, namespace)
            history_str = self._format_game_history(self.game_history)
            extracted_state = namespace["extract_state"](history_str)
            return str(extracted_state) if extracted_state else ""
        except Exception as e:
            print(f"Error extracting state: {e}")
            return ""

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

    # -------------------- Cross-episode helpers --------------------
    def _format_few_shot_from_cross_mem(self, game_history: list, current_state: str) -> str:
        try:
            # Extract states and actions from game_history
            states = [entry['state'] for entry in game_history]
            actions = [entry['action'] for entry in game_history]
            # Add current state
            states.append(current_state)
            
            # Pass the full trajectory history to retrieve_similar
            examples = self.cross_mem.retrieve_similar(
                query_states=states,
                query_actions=actions,
                k=3
            )
        except Exception:
            examples = []
        if not examples:
            return "(none)"
        lines = []
        for ex in examples:
            step = ex['step']
            lines.append(f"STATE: {step.get('state','')[:400]}")
            lines.append(f"ACTION: {step.get('action','')}")
            lines.append(f"GAIN: +{ex.get('reward',0)}")
            lines.append("---")
        return "\n".join(lines)

    def _detect_loop_segment(self):
        # Heuristic: look for last 20 steps repeating between two observations or zero-gain plateau for >= 10 steps
        if len(self._recent_states) < 12 or len(self._recent_scores) < 12:
            return None
        # Zero-gain plateau
        plateau = all((self._recent_scores[-i-1] == self._recent_scores[-1]) for i in range(10))
        if plateau:
            seg_len = min(40, len(self._recent_states))
            return self._recent_states[-seg_len:], self._recent_actions[-seg_len:]
        # Simple immediate alternation a<->b pattern
        a = self._recent_states[-1]
        b = self._recent_states[-2]
        alt = True
        for i in range(3, min(20, len(self._recent_states))):
            if (i % 2 == 0 and self._recent_states[-i] != b) or (i % 2 == 1 and self._recent_states[-i] != a):
                alt = False
                break
        if alt:
            seg_len = min(40, len(self._recent_states))
            return self._recent_states[-seg_len:], self._recent_actions[-seg_len:]
        return None

    def _format_negative_block_for_evolve(self) -> str:
        try:
            negatives = self.cross_mem.load_negative()
        except Exception:
            negatives = []
        if not negatives:
            return ""
        # Keep last few negative segments
        negatives = negatives[-3:]
        parts = []
        for neg in negatives:
            parts.append(f"Reason: {neg.get('reason','unknown')}, Length: {neg.get('length',0)}")
            # show last few actions only
            actions = neg.get('actions', [])[-10:]
            parts.append("Actions to avoid (tail): " + ", ".join(actions))
        return "\n".join(parts)
