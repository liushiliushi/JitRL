import os
import json
from .openai_helpers import chat_completion_with_retries


class AWMAgent:
    """
    Agent Workflow Memory (AWM) Agent

    Core idea: Maintain a single workflow per game that iteratively improves
    by inducing new workflows from old workflow + new episode trajectory.
    """

    def __init__(self, args, guiding_prompt: str = None):
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.memory = []  # Recent context for the agent
        self.game_history = []  # Current episode trajectory
        self.args = args

        # Game identification
        self.game_name = getattr(args, 'game_name', 'unknown_game')

        # Workflow storage path: output/{game}/awm/{model}/workflow.json
        output_path = getattr(args, 'output_path', 'output')
        model_slug = getattr(args, 'llm_model', 'model').replace('/', '_').replace('\\', '_')
        agent_type = getattr(args, 'agent_type', 'awm')
        self.workflow_dir = os.path.join(output_path, self.game_name, agent_type, model_slug)
        self.workflow_path = os.path.join(self.workflow_dir, 'workflow.json')

        # Current workflow (loaded at start_episode)
        self.workflow = None
        self.workflow_version = 0
        self.best_score = float('-inf')

        # LLM settings for induction
        self.induction_model = getattr(args, 'evolution_llm_model', args.llm_model)
        self.induction_temperature = getattr(args, 'induction_temperature', 0.7)
        self.induction_max_tokens = getattr(args, 'induction_max_tokens', 2000)

    def _load_workflow(self):
        """Load the workflow for the current game from disk."""
        if os.path.exists(self.workflow_path):
            try:
                with open(self.workflow_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.workflow = data.get('workflow', None)
                self.workflow_version = data.get('version', 0)
                self.best_score = data.get('best_score', float('-inf'))
                print(f"[AWMAgent] Loaded workflow v{self.workflow_version} for game '{self.game_name}'")
                return self.workflow
            except Exception as e:
                print(f"[AWMAgent] Error loading workflow: {e}")
                return None
        else:
            print(f"[AWMAgent] No existing workflow for game '{self.game_name}'")
            return None

    def _save_workflow(self, workflow: str, score: float):
        """Save the workflow for the current game to disk."""
        try:
            os.makedirs(self.workflow_dir, exist_ok=True)

            data = {
                'game_name': self.game_name,
                'version': self.workflow_version + 1,
                'workflow': workflow,
                'best_score': max(self.best_score, score),
                'total_episodes': self.workflow_version + 1
            }

            with open(self.workflow_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.workflow = workflow
            self.workflow_version = data['version']
            self.best_score = data['best_score']
            print(f"[AWMAgent] Saved workflow v{self.workflow_version} for game '{self.game_name}'")

        except Exception as e:
            print(f"[AWMAgent] Error saving workflow: {e}")

    def _format_trajectory(self, history):
        """Format game history into a readable trajectory string."""
        if not history:
            return "No trajectory recorded."

        trajectory_str = ""
        for i, entry in enumerate(history):
            trajectory_str += f"Step {i+1}:\n"
            trajectory_str += f"  STATE: {entry.get('state', '')[:500]}\n"
            trajectory_str += f"  ACTION: {entry.get('action', '')}\n"
            if entry.get('reward') is not None:
                trajectory_str += f"  REWARD: {entry.get('reward')}\n"
            if entry.get('score') is not None:
                trajectory_str += f"  SCORE: {entry.get('score')}\n"
            trajectory_str += "\n"

        return trajectory_str

    def _induce_workflow(self, old_workflow: str, trajectory: list, final_score: float) -> str:
        """
        Use LLM to induce a new workflow from old workflow + new trajectory.

        Args:
            old_workflow: The previous version of the workflow (may be None)
            trajectory: The game history from the current episode
            final_score: The final score achieved in this episode

        Returns:
            str: The new induced workflow
        """
        trajectory_str = self._format_trajectory(trajectory)

        # Build the induction prompt
        sys_prompt = """You are an expert at text adventure games. Your task is to create or update a game walkthrough/workflow based on gameplay experience.

A workflow is a step-by-step guide that describes the optimal sequence of actions to progress through the game. It should:
1. Be organized in chronological order of game progression
2. Include key actions that lead to rewards or progress
3. Note important items to collect and puzzles to solve
4. Be concise but complete
5. Abstract specific details when appropriate (e.g., "get the key" instead of "get the brass key" if the key name might vary)"""

        if old_workflow:
            user_prompt = f"""Please update the existing workflow based on the new gameplay trajectory.

【EXISTING WORKFLOW (v{self.workflow_version})】:
{old_workflow}

【NEW EPISODE TRAJECTORY】(Final Score: {final_score}):
{trajectory_str}

Please analyze both the existing workflow and the new trajectory, then generate an UPDATED workflow that:
1. Preserves proven effective steps from the existing workflow
2. Incorporates any new discoveries or better paths from this episode
3. Removes redundant or ineffective steps
4. Maintains chronological order of game progression
5. If the new trajectory achieved a higher score or found new areas, prioritize those insights

Output the updated workflow as a numbered list of steps. Each step should briefly describe the situation and the action to take.

UPDATED WORKFLOW:"""
        else:
            user_prompt = f"""Please create an initial workflow based on this gameplay trajectory.

【EPISODE TRAJECTORY】(Final Score: {final_score}):
{trajectory_str}

Please analyze the trajectory and generate a workflow that:
1. Captures the key actions that led to progress or rewards
2. Organizes steps in chronological order
3. Notes important items, locations, and puzzles
4. Abstracts specific details when appropriate

Output the workflow as a numbered list of steps. Each step should briefly describe the situation and the action to take.

WORKFLOW:"""

        try:
            response = chat_completion_with_retries(
                model=self.induction_model,
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                max_tokens=self.induction_max_tokens,
                temperature=self.induction_temperature,
            )

            if response and hasattr(response, 'choices') and response.choices and response.choices[0].message:
                new_workflow = response.choices[0].message.content.strip()
                print(f"[AWMAgent] Successfully induced new workflow")
                return new_workflow
            else:
                print(f"[AWMAgent] LLM returned empty response, keeping old workflow")
                return old_workflow or "No workflow available."

        except Exception as e:
            print(f"[AWMAgent] Error during workflow induction: {e}")
            return old_workflow or "No workflow available."

    def add_to_memory(self, state, response):
        """Add a state-response pair to the agent's short-term memory."""
        memory_entry = {"state": state, "response": response}
        self.memory.append(memory_entry)
        if len(self.memory) > self.args.max_memory:
            self.memory.pop(0)

    def _format_memory_for_prompt(self):
        """Format recent memory for inclusion in the prompt."""
        if not self.memory:
            return ""

        memory_text = "RECENT MEMORY:\n"
        for i, entry in enumerate(self.memory):
            memory_text += f"Memory {i+1}:\n"
            memory_text += f"  STATE: {entry['state'][:300]}...\n"
            if entry['response']:
                # Extract just the action from the response
                response_preview = entry['response'][:100] if entry['response'] else ""
                memory_text += f"  RESPONSE: {response_preview}...\n"

        return memory_text

    def start_episode(self):
        """Start a new episode: clear memory and load the current workflow."""
        self.memory = []
        self.game_history = []

        # Load the existing workflow for this game
        self._load_workflow()

        if self.workflow:
            print(f"[AWMAgent] Starting episode with workflow v{self.workflow_version}")
            print(f"[AWMAgent] Workflow preview: {self.workflow[:200]}...")
        else:
            print(f"[AWMAgent] Starting episode without workflow (first run)")

    def end_episode(self, state, score):
        """
        End an episode: induce a new workflow from old workflow + trajectory.

        Args:
            state: The final state of the game
            score: The final score achieved
        """
        print(f"[AWMAgent] Ending episode with score: {score}")

        # Add final state to history
        if self.game_history:
            self.game_history[-1]['final_state'] = state

        # Induce new workflow from old workflow + this episode's trajectory
        new_workflow = self._induce_workflow(
            old_workflow=self.workflow,
            trajectory=self.game_history,
            final_score=score
        )

        # Save the new workflow
        self._save_workflow(new_workflow, score)

        print(f"[AWMAgent] Workflow updated to v{self.workflow_version}")

    def get_prompts(self, state_node, info=None):
        """Generate prompts for the LLM, including workflow guidance."""
        memory_text = self._format_memory_for_prompt()

        # Build system prompt
        sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the game state, your memory, and the workflow guide."""

        if self.guiding_prompt:
            sys_prompt += f"\n\nGeneral guide: {self.guiding_prompt}"

        # Build workflow section
        workflow_section = ""
        if self.workflow:
            workflow_section = f"""
WORKFLOW GUIDE (learned from previous episodes):
{self.workflow}

Use this workflow as a reference. Follow the steps when applicable, but adapt to the current situation if needed.
"""
        else:
            workflow_section = """
WORKFLOW GUIDE: No workflow available yet. This is the first episode - explore and discover!
"""

        # Build user prompt
        user_prompt = f"""
{workflow_section}

{memory_text}

CURRENT STATE: {state_node.state}

Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, south, east, west, up, down), examine X, get X, open X, use X, etc.

Important:
- Follow the workflow guide when applicable
- If the workflow suggests an action for your current situation, prioritize it
- If you're in an unfamiliar situation not covered by the workflow, explore carefully
- Do not repeat failed actions
- Keep commands simple (1-3 words)

Your response MUST strictly follow this format:
REASONING: [Brief explanation of your choice based on workflow and current state]
ACTION: [short command]

Example:
REASONING: According to the workflow, I should open the mailbox to get the leaflet.
ACTION: open mailbox
"""
        return sys_prompt, user_prompt

    def generate_action(self, state_node, valid_actions=None, info=None):
        """Generate the next action based on workflow guidance and current state."""
        sys_prompt, user_prompt = self.get_prompts(state_node, info=info)

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
            print(f"[AWMAgent] LLM call failed, using default action")
            full_response = ""
            action_text = "look"

        # Add to memory and game history
        self.add_to_memory(state_node.state, full_response)
        self._add_to_game_history(state_node.state, action_text, full_response)

        return action_text.strip(), full_response

    def _parse_llm_response(self, full_response: str) -> str:
        """Parse the LLM response to extract the action."""
        action_text = "look"  # Default action

        if not full_response or not isinstance(full_response, str):
            return action_text

        lines = full_response.strip().split('\n')
        for line in lines:
            if line.upper().startswith("ACTION:"):
                action_text = line.split(":", 1)[1].strip()
                break

        return action_text

    def _add_to_game_history(self, state, action, full_response, reward=None, score=None):
        """Add an entry to the game history."""
        self.game_history.append({
            "state": state,
            "action": action,
            "full_response": full_response,
            "reward": reward,
            "score": score
        })

    def update_game_history_reward(self, reward, score):
        """Update the last entry in game history with reward and score."""
        if self.game_history and len(self.game_history) > 0:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score
