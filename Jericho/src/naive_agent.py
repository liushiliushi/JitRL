from .openai_helpers import chat_completion_with_retries

class NaiveAgent:
    def __init__(self, args, guiding_prompt: str = None):
        self.guiding_prompt = guiding_prompt or "Explore systematically and examine objects to make progress."
        self.args = args
        
    def add_to_memory(self, state, response):
        """Naive agent has no memory - this method does nothing."""
        pass
    
    def _format_memory_for_prompt(self):
        """Naive agent has no memory - returns empty string."""
        return ""
    
    def start_episode(self):
        """Start a new episode: no memory to clear."""
        print(f"Using initial prompt: '{self.guiding_prompt}'")

    def end_episode(self, state, score):
        """End an episode: no memory to update."""
        print(f"Ending episode with score: {score}.")

    def get_prompts(self, state_node):
        """
        Generate basic prompts using only current state.
        """
        return self._generate_basic_prompt(state_node.state)

    def generate_action(self, state_node, info=None):
        """
        Generates the next action using only current state.
        """
        sys_prompt, user_prompt = self.get_prompts(state_node)
        
        res_obj = chat_completion_with_retries(
            model=self.args.llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=100,
            temperature=self.args.llm_temperature,
        )

        if res_obj and hasattr(res_obj, 'choices') and res_obj.choices and res_obj.choices[0].message:
            full_response = res_obj.choices[0].message.content
            action_text = self._parse_llm_response(full_response)
        else:
            print(f"Warning: LLM API call might have failed or returned empty. Defaulting action.")
            full_response = ""
            action_text = "look" # Default action
            
        return action_text.strip(), full_response

    def _generate_basic_prompt(self, current_state):
        """Generate basic prompt with only current state."""
        sys_prompt = """You are an expert player aiming to complete a text-based adventure game. Points are given for making progress in the game. Select promising actions based on the current game state only."""
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        user_prompt = f"""
Your current state is: {current_state}

Type your next action as if you were playing the game directly. It should be a short command that can be understood by the game parser. Common actions include: look, inventory, directions (north, northeast, up, etc.), examine X, say X, drop X, get X, open X, enter X, ask X about Y, look in X, give X to Y, and other context-specific commands. To avoid parsing errors, any such X or Y MUST be a *SINGLE WORD* that identifies an object in a way the game can understand. When stuck, explore all rooms and objects mentioned in room descriptions systematically and comprehensively. *DO NOT REPEAT* the same failed action multiple times, as it will not lead to new results. Do not use the "help" command.

Your response MUST strictly follow this format and include nothing else:
REASONING: [A short, concise explanation of your choice, 1-2 sentences]
ACTION: [short word or phrase for text command to execute]

For example:
REASONING: I should examine the book to learn more about it.
ACTION: examine book
"""
        return sys_prompt, user_prompt
    
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