import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from browsergym.experiments import Agent, AbstractAgentArgs
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

from .utils.openai_helpers import chat_completion_with_retries, extract_json_from_response, TokenLimitExceededError
from .utils.chat_api import ChatModelArgs
from .dynamic_prompting import ActionSpace

from .strategy_space import StrategySpace
from .exploration import Explorer
from .guidance import Guidance, MilestoneTracker
from .evolution import Evolution

from . import dynamic_prompting


class StateNode:
    def __init__(self, state, instruction, reward=0.0):
        self.state = state
        self.instruction = instruction
        self.reward = reward
        self.response = ""


@dataclass
class JitRLAgentArgs(AbstractAgentArgs):
    chat_model_args: ChatModelArgs = None
    flags: dynamic_prompting.Flags = field(default_factory=lambda: dynamic_prompting.Flags())
    args: any = None  # To hold the parsed arguments

    def make_agent(self):
        return BrowserGymJitRLAgent(
            args=self.args,
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            guiding_prompt=None
        )


class BrowserGymJitRLAgent(Agent):
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

        self.memory = []
        self.web_history = []
        self.best_score = float("-inf")

        self.action_space = ActionSpace(self.flags)

        # ── Four-module system ──

        # Module 0: Strategy Space
        strategy_structure = getattr(self.args, 'strategy_structure', 'dag')
        seed_file = getattr(self.args, 'strategy_seed_file', 'strategy_seeds.json')
        # Build save directory for strategy persistence
        game_dir = getattr(self.args, 'output_path', 'output')
        game_dir = os.path.join(game_dir, getattr(self.args, 'game_name', 'game'))
        model_slug = getattr(self.args, 'llm_model', 'model').replace('/', '_').replace('\\', '_')
        agent_type = getattr(self.args, 'agent_type', 'memory')
        strategy_save_dir = os.path.join(game_dir, agent_type, model_slug, 'strategy')

        self.strategy_space = StrategySpace(
            structure=strategy_structure,
            seed_file=seed_file,
            save_dir=strategy_save_dir,
        )

        # Module 2: Exploration
        exploration_method = getattr(self.args, 'exploration_method', 'thompson')
        exploration_c = getattr(self.args, 'exploration_c', 1.414)
        exploration_epsilon = getattr(self.args, 'exploration_epsilon', 0.2)
        self.explorer = Explorer(
            method=exploration_method,
            c=exploration_c,
            epsilon=exploration_epsilon,
        )

        # Module 1: Guidance
        guidance_mode = getattr(self.args, 'guidance_mode', 'hierarchical')
        self.guidance = Guidance(mode=guidance_mode)

        # Module 3: Evolution
        evolution_method = getattr(self.args, 'evolution_method', 'reflection')
        evolution_interval = getattr(self.args, 'evolution_interval', 5)
        llm_model = getattr(self.args, 'llm_model', 'gpt-4o')
        self.evolution = Evolution(
            method=evolution_method,
            interval=evolution_interval,
            llm_model=llm_model,
        )

        # Milestone tracker (set during start_episode)
        self.milestone_tracker = None
        self.current_path = []
        self.current_domain = ""

        # Step summaries for memory
        self.step_summaries = []

        print(chat_model_args)
        print(f"\n--- STARTING EVALUATION ---")
        print(f"Task: WebArena, Agent LLM Model: {self.args.llm_model}")
        print(f"Agent Type: {self.args.agent_type}, Runs for statistics: {self.args.eval_runs}")
        print(f"Strategy: structure={strategy_structure}, guidance={guidance_mode}, "
              f"exploration={exploration_method}, evolution={evolution_method}")
        print(f"Base Seed for evaluation session: {self.args.seed}")
        self.start_episode()

    def _detect_domain(self) -> str:
        """Detect the domain from task_sites in args."""
        task_sites = getattr(self.args, 'task_sites', [])
        domain_keywords = {
            'shopping_admin': ['shopping_admin', 'admin'],
            'shopping': ['shopping', 'onestopshop'],
            'reddit': ['reddit', 'forum'],
            'gitlab': ['gitlab'],
            'wikipedia': ['wikipedia', 'wiki'],
            'map': ['map', 'openstreetmap'],
        }
        for site in task_sites:
            site_lower = site.lower() if isinstance(site, str) else str(site).lower()
            for domain, keywords in domain_keywords.items():
                if any(kw in site_lower for kw in keywords):
                    return domain
        return ""

    def start_episode(self):
        """Initialize a new episode."""
        self.memory = []
        self.game_history = []
        self._recent_scores = []
        self.step_summaries = []

        # Detect domain and select strategy path
        self.current_domain = self._detect_domain()
        self.current_path = self.explorer.select_path(self.strategy_space, self.current_domain)
        self.milestone_tracker = MilestoneTracker(
            self.current_path,
            llm_model=getattr(self.args, 'llm_model', 'gpt-4o')
        )

        if self.current_path:
            path_str = " → ".join([n.milestone for n in self.current_path])
            print(f"[Episode] Domain: {self.current_domain or 'unknown'}")
            print(f"[Episode] Selected strategy path: {path_str}")
        else:
            print(f"[Episode] No strategy path available for domain '{self.current_domain}'")

    def end_episode(self, state, score, success, llm_analysis, task_goal=None, user_instruction=None, screenshots_dir=None):
        """End an episode: update strategy stats and evolve."""
        print(f"Ending episode with score: {score}.")

        # Update strategy space stats
        if self.current_path:
            self.strategy_space.update_path_stats(self.current_path, success)

        # Trigger evolution
        episode_data = {
            "task_goal": task_goal or getattr(self, 'current_task_goal', ''),
            "game_history": self.game_history,
            "path": [{"id": n.id, "milestone": n.milestone} for n in self.current_path],
        }
        self.evolution.on_episode_end(
            episode_data, success, self.strategy_space, self.current_domain
        )

        # Persist strategy space
        self.strategy_space.save()

    def get_prompts(self, state_node):
        """Build system prompt and user prompt with strategy guidance."""
        # Enhance send_msg_to_user description
        original_prompt = self.action_space.prompt
        if "send_msg_to_user(text: str)" in original_prompt:
            self.action_space._prompt = original_prompt.replace(
                "send_msg_to_user(text: str)\n    Description: Sends a message to the user.",
                "send_msg_to_user(text: str)\n    Description: Sends final answer to user and terminates the task. WARNING: This immediately ends the task - you cannot take any more actions afterward. ONLY use when you have a definitive final answer. DO NOT use to ask questions or explain your thinking."
            )

        # Build memory text from game_history
        memory_text = ""
        if self.game_history:
            memory_parts = []
            for idx, entry in enumerate(self.game_history):
                raw_action = entry.get('action', 'N/A')
                step_text = f"Step {idx}: {raw_action}"
                if entry.get('url'):
                    step_text += f" | URL: {entry.get('url')}"
                # Include reasoning if available
                reasoning = self._extract_action_reasoning(entry.get('full_response'), entry.get('action'))
                if reasoning:
                    step_text += f"\n  Reasoning: {reasoning}"
                memory_parts.append(step_text)
            memory_text = "\n".join(memory_parts)

        # Generate guidance text from strategy path
        milestone_idx = self.milestone_tracker.current_idx if self.milestone_tracker else 0
        guidance_text = self.guidance.generate(
            self.current_path, self.strategy_space,
            milestone_idx=milestone_idx,
            observation=state_node.state[:1000]
        )

        # System prompt
        sys_prompt = f"""You are an intelligent web agent that interacts with real web pages on behalf of the user. Your goal is to accurately follow the user's natural language instructions by selecting and executing appropriate web actions.

User's instructions: {state_node.instruction}"""

        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"

        # Inject strategy guidance
        if guidance_text:
            sys_prompt += f"\n\n{guidance_text}"

        sys_prompt += f"""

## Available Actions
The available actions you can take are:
{self.action_space.prompt}

## Response Format
You must respond with a JSON object containing:
- "reasoning": Your step-by-step reasoning about the current state, what has been done, and what to do next
- "action": A single action command to execute (string)

IMPORTANT:
- If an action failed or gave no progress before, don't try it again in the same context
- Send message to the user once the instruction is fulfilled

CRITICAL: send_msg_to_user() is a TASK-ENDING action - once called, the task IMMEDIATELY TERMINATES.
Before using send_msg_to_user():
1. Verify you have gathered ALL information to COMPLETELY answer: "{state_node.instruction}"
2. Ensure your message is CONCISE, ON-TOPIC, and answers EVERY part of the instruction
3. Do NOT use it to explain your process, ask questions, or report partial progress

Example JSON response:
{{"reasoning": "I need to click the search button to find the product", "action": "click('a51')"}}"""

        # JSON format enforcement for non-OpenAI models
        from .utils.openai_helpers import _is_openai_model
        if not _is_openai_model(self.args.llm_model):
            sys_prompt += """

**CRITICAL: YOU MUST OUTPUT ONLY VALID JSON**

YOUR RESPONSE MUST BE PURE JSON ONLY. Follow these rules strictly:
- Do NOT write any explanatory text before or after the JSON
- Do NOT use markdown code blocks
- Output ONLY the raw JSON object starting with { and ending with }
- The JSON must be valid and parseable

CORRECT: {"reasoning": "I need to click the submit button", "action": "click('123')"}
INCORRECT: Let me think... {"reasoning": "...", "action": "..."}"""
        else:
            sys_prompt += "\n\nOutput your response as a JSON object with the specified fields."

        # User prompt
        user_prompt = f"""Your web browsing history and current state:
"""
        if memory_text:
            user_prompt += f"""{memory_text}

"""
        user_prompt += f"""=== Current step ({len(self.game_history)}) ===
Step {len(self.game_history)}: State:
{state_node.state}

Analyze the current state and provide a single action to execute."""

        return sys_prompt, user_prompt, memory_text

    def _extract_action_reasoning(self, full_response, action):
        """Extract reasoning from the LLM response."""
        if not full_response or not action:
            return None
        try:
            if isinstance(full_response, str):
                response_dict = json.loads(full_response)
            else:
                response_dict = full_response
            return response_dict.get('reasoning', None)
        except (json.JSONDecodeError, KeyError, AttributeError):
            return None

    def generate_action(self, state_node, url=None, screenshot=None):
        """Generate a single action from the LLM."""
        sys_prompt, user_prompt, memory_text = self.get_prompts(state_node)

        # Prepare screenshot if provided
        image_content = None
        if screenshot is not None:
            import base64
            from io import BytesIO
            from PIL import Image
            import numpy as np

            if isinstance(screenshot, np.ndarray):
                screenshot = Image.fromarray(screenshot)
            buffered = BytesIO()
            screenshot.save(buffered, format="PNG")
            image_content = base64.b64encode(buffered.getvalue()).decode('utf-8')
            print(f"[INFO] Screenshot encoded for action generation (base64 length: {len(image_content)})")

        # JSON schema for structured output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "web_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Step-by-step reasoning about the current state and action selection"
                        },
                        "action": {
                            "type": "string",
                            "description": "The web action to execute"
                        }
                    },
                    "required": ["reasoning", "action"],
                    "additionalProperties": False
                }
            }
        }

        # Retry logic
        max_retries = 5
        json_response = None
        full_response = None

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                print(f"[Retry {attempt}/{max_retries}]")

            current_user_prompt = user_prompt
            if attempt >= 4:
                current_user_prompt = user_prompt + f"""\n
CRITICAL REMINDER (Attempt {attempt}/{max_retries}):
Your previous {attempt-1} attempts failed. You MUST respond with ONLY a JSON object.
Start with {{ and end with }}. Nothing else.
Example: {{"reasoning": "...", "action": "click('123')"}}"""

            try:
                res_obj = chat_completion_with_retries(
                    model=self.args.llm_model,
                    sys_prompt=sys_prompt,
                    prompt=current_user_prompt,
                    max_tokens=4000,
                    temperature=self.args.llm_temperature,
                    top_logprobs=0,
                    response_format=response_format,
                    image_content=image_content,
                )
            except TokenLimitExceededError:
                raise

            if not res_obj or not hasattr(res_obj, 'choices') or not res_obj.choices:
                continue

            full_response = res_obj.choices[0].message.content
            json_response = extract_json_from_response(full_response)

            # Validate: must have "action" field
            if json_response and isinstance(json_response, dict) and json_response.get('action'):
                break
            else:
                if attempt > 1:
                    print(f"  Invalid response: missing 'action' field")
                json_response = None

        # Fallback: try to extract action from natural language
        if not json_response or not json_response.get('action'):
            print(f"\n!!! Failed to get valid response after {max_retries} attempts !!!")
            if full_response:
                action = self._extract_action_from_text(full_response)
                if action:
                    return action, full_response
            print("Could not extract valid action. Using default 'noop'")
            return "noop(1000)", full_response or ""

        action_text = json_response['action']

        print("\n" + "=" * 100)
        print("ACTION SELECTION")
        print("=" * 100)
        print(f"Reasoning: {json_response.get('reasoning', 'N/A')[:200]}")
        print(f"Action: {action_text}")
        print("=" * 100)

        self._add_to_game_history(
            state_node.state, action_text, full_response,
            task_goal=state_node.instruction, url=url
        )

        return action_text.strip(), full_response

    def _extract_action_from_text(self, text):
        """Try to extract an action from natural language response."""
        action_patterns = [
            r"(click\(['\"][\w-]+['\"]\))",
            r"(fill\(['\"][\w-]+['\"]\s*,\s*['\"].*?['\"]\))",
            r"(send_msg_to_user\(['\"].*?['\"]\))",
            r"(scroll\(['\"][\w-]+['\"]\))",
            r"(select_option\(['\"][\w-]+['\"]\s*,\s*['\"].*?['\"]\))",
        ]
        for pattern in action_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                action = match.group(1)
                print(f"Extracted action from text: {action}")
                return action
        return None

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

        normalized_action = action
        if action and state:
            try:
                normalized_data = normalize_action(action, state)
                normalized_action = normalized_data.get('normalized_action', action)
            except Exception as e:
                print(f"Warning: Failed to normalize action for summary: {e}")

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
        """Update the last entry in game history with reward and score."""
        if self.game_history and len(self.game_history) > 0:
            self.game_history[-1]["reward"] = reward
            self.game_history[-1]["score"] = score

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
        print("\n" + "=" * 100)
        print(f"STEP {len(self.game_history)}")
        print("=" * 100)

        # Clean up old screenshots at step 0
        if hasattr(self, '_exp_args') and hasattr(self._exp_args, 'exp_dir'):
            current_step = len(self.game_history)
            if current_step == 0:
                import glob
                screenshot_pattern = str(Path(self._exp_args.exp_dir) / "screenshot_step_*.png")
                old_screenshots = glob.glob(screenshot_pattern)
                if old_screenshots:
                    print(f"[INFO] Cleaning up {len(old_screenshots)} old screenshots")
                    for old_screenshot in old_screenshots:
                        try:
                            os.remove(old_screenshot)
                        except Exception as e:
                            print(f"[Warning] Failed to remove {old_screenshot}: {e}")

            # Pre-save current step's screenshot
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

        web_text = obs['axtree_txt']
        url = obs.get('url', 'about:blank')
        print(f"URL: {url}")
        print(f"Task: {obs['goal']}")

        self.current_task_goal = obs['goal']

        # Milestone tracking: check if current milestone is completed
        if self.milestone_tracker and not self.milestone_tracker.is_complete:
            completed = self.milestone_tracker.check_completion(web_text)
            if completed:
                milestone = self.milestone_tracker.path[self.milestone_tracker.current_idx - 1]
                print(f"[Milestone] Advanced past: {milestone.milestone}")

        # Get screenshot if enabled
        screenshot = None
        if hasattr(self.args, 'use_screenshot_action') and self.args.use_screenshot_action:
            screenshot = obs.get('screenshot')

        state_node = StateNode(state=web_text, instruction=obs['goal'])
        action, raw_llm_output = self.generate_action(state_node, url=url, screenshot=screenshot)
        element_id = self.extract_quoted_numbers(action)
        target_element = self.find_elements(web_text, with_id=int(element_id[0])) if element_id else []

        raw_llm_output_dict = {}
        if raw_llm_output and isinstance(raw_llm_output, str):
            raw_llm_output_dict = extract_json_from_response(raw_llm_output)

        ans_dict = {
            'think': raw_llm_output_dict.get('reasoning', '') if isinstance(raw_llm_output_dict, dict) else '',
            'action': action,
            'target_element': target_element
        }

        return action, ans_dict
