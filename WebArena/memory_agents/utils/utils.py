import numpy as np
import json
import re
from urllib.parse import urlparse
from .openai_helpers import chat_completion_with_retries, _is_openai_model, _needs_max_completion_tokens
import os


def normalize_url(url: str) -> str:
    """
    Normalize URL to a standardized format for similarity comparison.

    Strategy:
    1. Map domain:port to website name (from environment variables)
    2. Extract path and remove query parameters/fragments
    3. Replace numeric IDs with 'ID'
    4. Replace base64/hash-like strings with 'HASH'
    5. Format: website_name/path

    Examples:
        http://172.28.176.115:8083/admin/customer/edit/123/
        -> shopping_admin/admin/customer/edit/ID

        http://172.28.176.115:7770/wiki/Article
        -> wikipedia/wiki/Article

    Args:
        url: Original URL

    Returns:
        Normalized URL with website name prefix
    """
    if not url or url == 'N/A':
        return 'unknown'

    try:
        # Domain:port to website name mapping (from WebArena environment)
        # These mappings are based on common WebArena-Lite setup
        domain_mapping = {
            ':8083': 'shopping_admin',  # OpenCart admin
            ':7770': 'wikipedia',        # MediaWiki
            ':9999': 'shopping',         # OpenCart shopping site
            ':3000': 'gitlab',           # GitLab
            ':8888': 'reddit',           # Reddit-like forum
            ':8080': 'map',              # Map service
        }

        # Parse URL
        parsed = urlparse(url)

        # Get website name from port
        website_name = 'unknown_site'
        for port_pattern, site_name in domain_mapping.items():
            if port_pattern in url or f':{parsed.port}' == port_pattern:
                website_name = site_name
                break

        path = parsed.path

        # Remove leading/trailing slashes
        path = path.strip('/')

        # Split path into segments
        segments = path.split('/')

        normalized_segments = []
        for i, segment in enumerate(segments):
            if not segment:
                continue

            # Stop at 'filter' - everything after is just filter parameters
            # Examples: filter/cmVwb3J0X3BlcmlvZD1kYXk=/form_key/PPOtHm54S8FK1Tvf
            if segment == 'filter':
                break

            # Check if segment is a number (likely an ID)
            if segment.isdigit():
                normalized_segments.append('ID')
            # Check if segment is a form_key keyword (keep it)
            elif segment == 'form_key':
                normalized_segments.append('form_key')
            # Check if segment looks like a hash/base64/token
            # Heuristics for hash detection:
            # 1. Contains base64 special chars (+ / =) -> definitely a hash
            # 2. Mixed case with no underscores and length > 15 -> likely a hash/token
            # 3. All uppercase letters with length > 10 -> likely a token
            # BUT: Keep meaningful path segments like 'report_customer', 'order_status'
            elif (
                # Base64 strings (contain +, /, or =)
                any(char in segment for char in ['+', '/', '='])
                # Random tokens (mixed case, no underscores, long)
                or (len(segment) > 15 and not '_' in segment and segment != segment.lower() and segment != segment.upper())
                # Session tokens (all letters, mixed case, long)
                or (len(segment) > 12 and re.match(r'^[A-Za-z0-9]+$', segment) and segment != segment.lower() and segment != segment.upper() and not '_' in segment)
            ):
                normalized_segments.append('HASH')
            else:
                # Keep the segment as-is (including meaningful paths like 'report_customer')
                normalized_segments.append(segment)

        # Join segments with website name prefix
        if normalized_segments:
            normalized_path = f"{website_name}/{'/'.join(normalized_segments)}"
        else:
            normalized_path = website_name

        return normalized_path

    except Exception as e:
        # Fallback: return path as-is if parsing fails
        return url


def softmax(a, T=1):
    a = np.array(a) / T
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def parse_action(action_str):
    """
    Parse an action string to extract action type, bid, and parameters.

    Examples:
        "select_option('1240', 'Price')" -> {'type': 'select_option', 'bid': '1240', 'params': ['Price']}
        "fill('260', 'switch card holder')" -> {'type': 'fill', 'bid': '260', 'params': ['switch card holder']}
        "click('223')" -> {'type': 'click', 'bid': '223', 'params': []}
        "scroll(0, 500)" -> {'type': 'scroll', 'bid': None, 'params': [0, 500]}

    Args:
        action_str: Raw action string

    Returns:
        dict with keys: 'type', 'bid', 'params', 'raw'
    """
    if not action_str or not isinstance(action_str, str):
        return {'type': None, 'bid': None, 'params': [], 'raw': action_str}

    # Pattern: action_type('bid', params...)
    pattern = r"^(\w+)\((.*)\)$"
    match = re.match(pattern, action_str.strip())

    if not match:
        return {'type': None, 'bid': None, 'params': [], 'raw': action_str}

    action_type = match.group(1)
    args_str = match.group(2)

    # Parse arguments
    params = []
    bid = None

    if args_str:
        # Split by comma, but respect quoted strings
        # Simple approach: use json to parse as array
        try:
            # Wrap in array brackets to parse as JSON array
            args_array = json.loads(f"[{args_str}]")

            # First argument is typically the bid (if it's a string that looks like a number)
            if len(args_array) > 0:
                first_arg = args_array[0]
                # Check if first arg is a bid (string of digits or number)
                if isinstance(first_arg, str) and first_arg.isdigit():
                    bid = first_arg
                    params = args_array[1:]  # Remaining args are params
                elif isinstance(first_arg, int):
                    # Some actions like scroll(0, 500) don't have bid
                    params = args_array
                else:
                    # For actions like send_msg_to_user('message')
                    params = args_array
        except:
            # Fallback: manual parsing
            parts = [p.strip().strip("'\"") for p in args_str.split(',')]
            if parts and parts[0].isdigit():
                bid = parts[0]
                params = parts[1:]
            else:
                params = parts

    return {
        'type': action_type,
        'bid': bid,
        'params': params,
        'raw': action_str
    }


def extract_element_description(bid, state_text):
    """
    Extract element description from accessibility tree state for a given bid.

    Args:
        bid: Browser ID (e.g., '1240')
        state_text: Full accessibility tree text

    Returns:
        dict with keys: 'role', 'label', 'full_description', or None if not found
    """
    if not bid or not state_text:
        return None

    # Pattern: [bid] role 'label', attributes...
    # Example: [1240] combobox 'Sort by', expanded=False
    pattern = rf"\[{bid}\]\s+(\w+)\s+'([^']*)'[^\n]*"
    match = re.search(pattern, state_text)

    if match:
        role = match.group(1)
        label = match.group(2)
        full_line = match.group(0)
        return {
            'role': role,
            'label': label,
            'full_description': full_line
        }

    # Alternative pattern without label: [bid] role attributes...
    # Example: [1240] button, disabled=True
    pattern_no_label = rf"\[{bid}\]\s+(\w+)[^\n]*"
    match_no_label = re.search(pattern_no_label, state_text)

    if match_no_label:
        role = match_no_label.group(1)
        full_line = match_no_label.group(0)
        return {
            'role': role,
            'label': '',
            'full_description': full_line
        }

    return None


def normalize_action(action_str, state_text, normalize_send_msg=True):
    """
    Normalize an action by replacing bid with semantic element description.

    Args:
        action_str: Raw action string (e.g., "select_option('1240', 'Price')")
        state_text: Full accessibility tree text
        normalize_send_msg: If True, normalize send_msg_to_user to 'xxx'.
                           If False, keep original message content.

    Returns:
        dict with keys:
            - 'raw_action': original action
            - 'action_type': action type
            - 'element': element description (role + label)
            - 'params': action parameters
            - 'semantic_action': human-readable action description
            - 'normalized_action': standardized action for comparison
    """
    parsed = parse_action(action_str)

    if not parsed['type']:
        return {
            'raw_action': action_str,
            'action_type': None,
            'element': None,
            'params': [],
            'semantic_action': action_str,
            'normalized_action': action_str
        }

    # Special handling for send_msg_to_user / send_message_to_user actions
    if parsed['type'] in ['send_msg_to_user', 'send_message_to_user']:
        if normalize_send_msg:
            # For comparison purposes, replace specific message content with 'xxx'
            normalized = f"{parsed['type']}('xxx')"
            semantic = f"{parsed['type']} with message: xxx"
            params = ['xxx']
        else:
            # Keep original message for display/evaluation
            original_msg = parsed['params'][0] if parsed['params'] else ''
            normalized = f"{parsed['type']}({json.dumps(original_msg)})"
            semantic = f"{parsed['type']} with message: {original_msg}"
            params = parsed['params']

        return {
            'raw_action': action_str,
            'action_type': parsed['type'],
            'element': None,
            'params': params,
            'semantic_action': semantic,
            'normalized_action': normalized
        }

    element_desc = None
    element_str = None

    # Extract element description if bid exists
    if parsed['bid']:
        element_desc = extract_element_description(parsed['bid'], state_text)
        if element_desc:
            # Create semantic element string
            if element_desc['label']:
                element_str = f"{element_desc['role']}[{element_desc['label']}]"
            else:
                element_str = f"{element_desc['role']}"

    # Build normalized action (bid replaced with semantic element)
    if element_str:
        params_str = ', '.join([json.dumps(p) for p in parsed['params']])
        if params_str:
            normalized = f"{parsed['type']}(<{element_str}>, {params_str})"
            semantic = f"{parsed['type']} on {element_str} with params: {params_str}"
        else:
            normalized = f"{parsed['type']}(<{element_str}>)"
            semantic = f"{parsed['type']} on {element_str}"
    else:
        # No bid or element not found, use original format
        params_str = ', '.join([json.dumps(p) for p in parsed['params']])
        normalized = f"{parsed['type']}({params_str})"
        semantic = f"{parsed['type']} with params: {params_str}"

    return {
        'raw_action': action_str,
        'action_type': parsed['type'],
        'element': element_desc,
        'params': parsed['params'],
        'semantic_action': semantic,
        'normalized_action': normalized
    }


def denormalize_action(normalized_action: str, state_text: str) -> str:
    """
    Convert a normalized action back to raw action by finding the corresponding bid in current state.

    For example:
        normalized_action: "click(<link[\ue60a REPORTS]>)"
        state_text: "...[325] link '\ue60a REPORTS'..."
        returns: "click('325')"

    Args:
        normalized_action: Normalized action string like "click(<link[label]>)" or "fill(<textbox[label]>, 'text')"
        state_text: Current accessibility tree state

    Returns:
        Raw action string with correct bid, or None if element not found
    """
    import re

    if not normalized_action or not state_text:
        return None

    # Parse normalized action: action_type(<role[label]>, params...)
    # Examples:
    #   click(<link[\ue60a REPORTS]>)
    #   fill(<textbox[Search]>, "query")
    #   select_option(<combobox[Period]>, "Month")

    # Pattern to match: action_type(<role[label]>, optional_params)
    pattern = r"^(\w+)\(<([^>\[\]]+)\[([^\]]*)\]>(?:,\s*(.+))?\)$"
    match = re.match(pattern, normalized_action)

    if not match:
        # Try pattern without label: action_type(<role>)
        pattern_no_label = r"^(\w+)\(<([^>\[\]]+)>(?:,\s*(.+))?\)$"
        match_no_label = re.match(pattern_no_label, normalized_action)
        if match_no_label:
            action_type = match_no_label.group(1)
            role = match_no_label.group(2)
            label = None
            params_str = match_no_label.group(3)
        else:
            # Cannot parse, return None
            return None
    else:
        action_type = match.group(1)
        role = match.group(2)
        label = match.group(3)
        params_str = match.group(4)

    # Find element in state_text by role and label
    # State format: [bid] role 'label' or [bid] role
    if label:
        # Escape special regex characters in label
        escaped_label = re.escape(label)
        # Pattern: [bid] role 'label'
        state_pattern = rf"\[(\d+)\]\s+{re.escape(role)}\s+'{escaped_label}'"
        state_match = re.search(state_pattern, state_text)

        if not state_match:
            # Try without exact label match (partial match)
            state_pattern_partial = rf"\[(\d+)\]\s+{re.escape(role)}\s+'[^']*{escaped_label}[^']*'"
            state_match = re.search(state_pattern_partial, state_text)
    else:
        # No label, just match by role
        state_pattern = rf"\[(\d+)\]\s+{re.escape(role)}"
        state_match = re.search(state_pattern, state_text)

    if not state_match:
        return None

    bid = state_match.group(1)

    # Build raw action
    if params_str:
        raw_action = f"{action_type}('{bid}', {params_str})"
    else:
        raw_action = f"{action_type}('{bid}')"

    return raw_action


def calculate_action_similarity(action1_normalized, action2_normalized):
    """
    Calculate similarity between two normalized actions.

    Args:
        action1_normalized: dict from normalize_action()
        action2_normalized: dict from normalize_action()

    Returns:
        float: similarity score between 0.0 and 1.0
    """
    if not action1_normalized or not action2_normalized:
        return 0.0

    norm1 = action1_normalized.get('normalized_action', '')
    norm2 = action2_normalized.get('normalized_action', '')

    # Exact match
    if norm1 == norm2:
        return 1.0

    # Action type must match
    type1 = action1_normalized.get('action_type')
    type2 = action2_normalized.get('action_type')

    if type1 != type2:
        return 0.0

    # Compare elements
    elem1 = action1_normalized.get('element')
    elem2 = action2_normalized.get('element')

    element_similarity = 0.0
    if elem1 and elem2:
        # Role must match
        if elem1.get('role') != elem2.get('role'):
            return 0.0

        # Label similarity
        label1 = elem1.get('label', '').lower()
        label2 = elem2.get('label', '').lower()

        if label1 == label2:
            element_similarity = 1.0
        elif label1 and label2:
            # Partial label match
            if label1 in label2 or label2 in label1:
                element_similarity = 0.7
            else:
                element_similarity = 0.3

    # Compare parameters
    params1 = action1_normalized.get('params', [])
    params2 = action2_normalized.get('params', [])

    params_similarity = 0.0
    if params1 == params2:
        params_similarity = 1.0
    elif len(params1) == len(params2):
        # Check if params are similar
        matches = sum(1 for p1, p2 in zip(params1, params2) if p1 == p2)
        params_similarity = matches / len(params1) if params1 else 1.0

    # Weighted combination
    # Element similarity is most important (60%), params are secondary (40%)
    if elem1 or elem2:
        similarity = 0.6 * element_similarity + 0.4 * params_similarity
    else:
        # No elements (e.g., scroll action), only compare params
        similarity = params_similarity

    return similarity


def game_file(game_name):
    rom_dict = {'zork1': 'zork1.z5', 
                'zork3': 'zork3.z5', 
                'spellbrkr' : 'spellbrkr.z3',
                'advent': 'advent.z5',                 
                'detective': 'detective.z5', 
                'pentari': 'pentari.z5',
                'enchanter': 'enchanter.z3',
                'library' : 'library.z5',
                'balances' : 'balances.z5',
                'ztuu' : 'ztuu.z5',
                'ludicorp' : 'ludicorp.z5',
                'deephome' : 'deephome.z5',
                'temple' : 'temple.z5',
                'anchor' : 'anchor.z8',
                'awaken' : 'awaken.z5',
                'zenon' : 'zenon.z5'
                }
                
    return rom_dict[game_name]

# Placeholder for initial prompt loading utilities
# These should be implemented based on the actual structure of initial_prompts.json
# and how generic prompts are defined.

def load_initial_prompts(file_path):
    """
    Loads initial prompts from a JSON file.
    Expects a JSON file with a "prompts" key, which is a list of strings.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        if "prompts" not in data or not isinstance(data["prompts"], list):
            print(f"Warning: Prompt file {file_path} is missing 'prompts' list or it's not a list. Returning empty list.")
            return []
        return data["prompts"]
    except FileNotFoundError:
        print(f"Warning: Initial prompts file {file_path} not found. Returning empty list.")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from {file_path}. Returning empty list.")
        return []

def get_generic_initial_prompts(num_prompts=1):
    """
    Returns a list of generic initial prompts.
    """
    # This is a basic implementation. You might want to make these more sophisticated.
    generic_prompts = [
        "Analyze the game state and choose the best action to maximize the score.",
        "Think step-by-step to understand the current situation and decide the next move.",
        "Your goal is to achieve the highest score. Observe, think, and act.",
        "Explore the environment, interact with objects, and solve puzzles to progress.",
        "Be methodical. Consider all available actions and their potential outcomes."
    ]
    return generic_prompts[:num_prompts]


def generate_single_step_summary(step_index, state, action=None, task_goal=None, llm_model="gpt-4", temperature=0.1, max_tokens=300):
    """
    Generates a FUNCTIONAL summary for a SINGLE step, focusing on page structure and available actions
    while omitting specific task-related content (product names, prices, locations, etc.).

    This enables similar pages (e.g., different product detail pages) to have high similarity,
    allowing the agent to leverage experiences from similar page types across different tasks.

    Args:
        step_index: The step number (e.g., 0, 1, 2, ...)
        state: The state text for this step
        action: The action taken at this step (None for the last/current step)
        task_goal: The overall task goal (currently not used in functional summarization)
        llm_model: The LLM model to use for generation
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens for the summary

    Returns:
        str: A functional summary describing page type, available elements, and sections
    """
    if not state:
        return f"Step {step_index}: [State: empty]"

    # System prompt for functional page summarization
    sys_prompt = """You are an expert at abstracting web page functionality and structure.
Your task is to describe the PAGE TYPE and FUNCTIONAL ELEMENTS available on the page,
while OMITTING specific task-related content like product names, locations, prices, dates, user-generated content, etc.

Focus on:
- What TYPE of page this is (e.g., "Admin dashboard", "Product detail", "Search results")
- What FUNCTIONAL ELEMENTS are available (buttons, links, forms, input fields)
- What SECTIONS/AREAS exist on the page (reviews section, description area, navigation menu)

OMIT completely:
- Specific product/location/person/company names
- Specific prices, dates, numbers, ratings
- Specific review content, article text, or user comments
- Task-specific query terms or search keywords

Simply describe the page structure and what actions can be performed, without mentioning the specific content."""

    # Limit state to reasonable size to avoid token overflow
    state_content = state

    # Build user prompt with structured output format
    if action:
        user_prompt = f"""Describe the web page's functionality and structure, omitting all specific content:

Step {step_index}:
State: {state_content}
Action: {action}

Provide a functional summary using this EXACT format:

Step {step_index}:
State: [Page Type] page with: [list functional elements separated by commas, e.g., "button1, button2, link1, form1"], sections: [list page sections separated by commas]
Action: {action}

Requirements:
- Always start with page type (e.g., "Admin dashboard", "Product detail", "Search results")
- List functional elements in alphabetical order
- Use generic names (e.g., "navigation menu", "search bar", not specific labels)
- Omit ALL specific content: product names, prices, dates, user names, etc.

Example:
State: Admin dashboard page with: customer link, dashboard link, export button, filter button, sales link, search input, sections: header, main content area, navigation menu, table"""
    else:
        # Current step (no action yet)
        user_prompt = f"""Describe the web page's functionality and structure, omitting all specific content:

Step {step_index}:
State: {state_content}

Provide a functional summary using this EXACT format:

Step {step_index}:
State: [Page Type] page with: [list functional elements separated by commas, e.g., "button1, button2, link1, form1"], sections: [list page sections separated by commas]

Requirements:
- Always start with page type (e.g., "Admin dashboard", "Product detail", "Search results")
- List functional elements in alphabetical order
- Use generic names (e.g., "navigation menu", "search bar", not specific labels)
- Omit ALL specific content: product names, prices, dates, user names, etc.

Example:
State: Admin dashboard page with: customer link, dashboard link, export button, filter button, sales link, search input, sections: header, main content area, navigation menu, table"""

    response = chat_completion_with_retries(
        model=llm_model,
        sys_prompt=sys_prompt,
        prompt=user_prompt,
        max_tokens=1000,
        temperature=temperature
    )

    if response and hasattr(response, 'choices') and response.choices:
        summary = response.choices[0].message.content.strip()
        # Clean up if LLM adds extra formatting
        if summary.startswith('```'):
            summary = summary.replace('```', '').strip()
        return summary
    else:
        # Fallback
        return generate_single_step_summary_fallback(step_index, state, action)




def generate_single_step_summary_fallback(step_index, state, action=None):
    """Fallback method for single step summarization if LLM fails."""
    state_snippet = state[:100].replace('\n', ' ').strip() if state else "empty"
    if action:
        return f"Step {step_index}: [State: {state_snippet}...] [Action: {action}]"
    else:
        return f"Step {step_index}: [State: {state_snippet}...]"


def generate_trajectory_summary(game_history, llm_model="gpt-4", temperature=0.8, max_tokens=500):
    """
    DEPRECATED: Use generate_single_step_summary instead for better efficiency.

    This function is kept for backward compatibility but now generates summaries
    step-by-step using the new single-step approach.
    """
    if not game_history:
        return ""

    summaries = []
    for i, entry in enumerate(game_history):
        state = entry.get('state', '')
        action = entry.get('action', '') if i < len(game_history) - 1 else None
        summary = generate_single_step_summary(i, state, action, llm_model, temperature, max_tokens)
        summaries.append(summary)

    return '\n'.join(summaries)


def generate_trajectory_summary_fallback(game_history):
    """
    Fallback method for trajectory summarization if LLM fails.
    
    Args:
        game_history: List of game history entries
        
    Returns:
        str: A simple fallback summary
    """
    if not game_history:
        return ""
    
    # Return last few states and actions in simple format
    summary_text = "Recent actions: "
    recent_history = game_history[-3:] if len(game_history) > 3 else game_history
    
    actions = [entry['action'] for entry in recent_history if entry.get('action')]
    if actions:
        summary_text += ", ".join(actions) + ". "
    
    # Add final state and score info
    last_entry = game_history[-1]
    if last_entry.get('score') is not None:
        summary_text += f"Current score: {last_entry['score']}. "
    
    # Add brief state description (first 100 chars of last state)
    if last_entry.get('state'):
        state_snippet = last_entry['state'][:100].replace('\n', ' ').strip()
        summary_text += f"State: {state_snippet}..."
    
    return summary_text


def calculate_summary_similarity(summary1: str, summary2: str, llm_model: str = "gpt-4", temperature: float = 0.3, max_tokens: int = 50) -> float:
    """
    Calculate similarity score between two summaries using LLM.
    
    Args:
        summary1: First trajectory summary
        summary2: Second trajectory summary  
        llm_model: The LLM model to use for scoring
        temperature: Temperature for LLM generation (lower = more consistent)
        max_tokens: Maximum tokens for the response
        
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not summary1 or not summary2:
        return 0.0
    
    sys_prompt = """You are an expert at evaluating the semantic similarity between game trajectory summaries. 
You must output ONLY a single decimal number between 0.0 and 1.0 representing the similarity score.
BE STRICT in your evaluation:
- 0.0-0.5: Not similar trajectories.
- 0.5-0.79: Similar trajectories.
- 0.8+: Nearly identical trajectories
CRITICAL: The LAST STATE in each summary represents the current game situation and is more important."""
    
    user_prompt = f"""Compare these two game trajectory summaries and output ONLY a similarity score between 0.0 and 1.0.

Summary 1:
{summary1}

Summary 2:
{summary2}

Output only the numerical score (e.g., 0.65):"""
    
    try:
        response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if response and hasattr(response, 'choices') and response.choices:
            score_text = response.choices[0].message.content.strip()
            # Extract numerical value from response
            try:
                score = float(score_text)
                # Ensure score is within valid range
                return max(0.0, min(1.0, score))
            except ValueError:
                print(f"[Warning] Failed to parse similarity score: {score_text}")
                return 0.0
        else:
            print("[Warning] No response from LLM for similarity calculation")
            return 0.0
    except Exception as e:
        print(f"[Warning] Failed to calculate LLM similarity: {e}")
        return 0.0


def detect_loops_in_trajectory(game_history, window_size=3):
    """
    Detect loop patterns in trajectory.

    Args:
        game_history: List of game history records
        window_size: Window size for loop detection

    Returns:
        list: Boolean list indicating whether each step is in a loop
    """
    if not game_history or len(game_history) < window_size * 2:
        return [False] * len(game_history)

    loop_flags = [False] * len(game_history)

    for i in range(len(game_history)):
        current_step = game_history[i]
        current_state = current_step.get('state', '')
        current_action = current_step.get('action', '')

        loop_count = 0
        for j in range(max(0, i - 10), i):
            prev_step = game_history[j]
            prev_state = prev_step.get('state', '')
            prev_action = prev_step.get('action', '')

            if (current_state and prev_state and
                len(current_state) > 20 and len(prev_state) > 20):
                shorter = min(len(current_state), len(prev_state))
                common_chars = sum(c1 == c2 for c1, c2 in zip(current_state[:shorter], prev_state[:shorter]))
                similarity = common_chars / shorter if shorter > 0 else 0

                if similarity > 0.8 and current_action == prev_action:
                    loop_count += 1

        if loop_count >= 2:
            loop_flags[i] = True

    return loop_flags


def evaluate_with_screenshots(game_history, screenshots_dir, success, task_goal, llm_model="gpt-4o", temperature=0.3):
    """
    Visual evaluation using screenshots instead of text-based page states.

    Args:
        game_history: List of game history records
        screenshots_dir: Path to screenshots directory
        success: Whether the task was successful
        task_goal: Task goal
        llm_model: Model to use
        temperature: Temperature parameter

    Returns:
        tuple: (scores list, reasonings list)
    """
    import base64
    from .openai_helpers import _get_client

    client = _get_client(llm_model)

    # Build action history text (without states, only normalized actions)
    actions_text = ""
    for i, entry in enumerate(game_history):
        raw_action = entry.get('action', '')
        state_text = entry.get('state', '')

        # Normalize action for display
        try:
            normalized_data = normalize_action(raw_action, state_text)
            display_action = normalized_data.get('normalized_action', raw_action)
        except:
            display_action = raw_action

        actions_text += f"Step {i}: {display_action}\n"

    # Build prompt
    sys_prompt = """You are evaluating web agent actions using WEBPAGE SCREENSHOTS.

CORE EVALUATION PRINCIPLE:
You will see screenshots of webpages AFTER each action was taken.
Score each action based on whether it contributed to achieving the task goal.

SCORING PRINCIPLE:
- Action contributed to the task goal → High positive score (2-3)
- Action was not useful for the task → Low or negative score
- Uncertain about contribution → 0 score

MANDATORY FORMAT:
For each step, your "detailed_reasoning" MUST include:
1. "After this action, the webpage screenshot shows: [describe what you see in the screenshot]"
2. "This result is [helpful/unhelpful] because: [reason]"
3. "Score: [number] - [repeat/avoid]"

SCORING SCALE: Use -3 to +3
- Positive scores: The action was useful for achieving the task goal
- Negative scores: The action was unhelpful or counterproductive
- Zero: Uncertain about usefulness"""

    task_section = f"TASK GOAL: {task_goal}\n\n" if task_goal else ""
    result_text = "SUCCESS ✓" if success else "FAILURE ✗"

    user_prompt = f"""Score each action based on the WEBPAGE SCREENSHOTS, NOT on whether the action itself seems reasonable.

{task_section}==========================================
FINAL TASK RESULT: {result_text}
==========================================

You will be shown {len(game_history)} screenshots, one after each action.

ACTIONS TAKEN:
{actions_text}

Now analyze each screenshot and provide scores in JSON format:
{{
  "step_analysis": [
    {{
      "step": 0,
      "action": "action taken",
      "detailed_reasoning": "After this action, the webpage screenshot shows: [describe]. This result is [helpful/unhelpful] because: [reason]. Score: [number] - [repeat/avoid]",
      "score": 2,
      "key_observations": "What the screenshot shows"
    }}
  ],
  "overall_assessment": "Brief summary"
}}"""

    # Prepare message content with screenshots
    content = [{"type": "text", "text": user_prompt}]

    # Add screenshots
    for i in range(len(game_history)):
        screenshot_path = os.path.join(screenshots_dir, f"screenshot_step_{i}.png")
        if os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                    "detail": "low"  # Use low detail to save tokens
                }
            })
            content.append({
                "type": "text",
                "text": f"[Screenshot after Step {i}]"
            })

    # Call vision API
    print(f"[INFO] Calling vision API with {len(game_history)} screenshots...")
    # Build API parameters
    api_params = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content}
        ],
        "temperature": temperature,
    }
    # Use max_completion_tokens for newer models (gpt-4o, o1, etc.)
    if _needs_max_completion_tokens(llm_model):
        api_params["max_completion_tokens"] = 4096
    else:
        api_params["max_tokens"] = 4096
    response = client.chat.completions.create(**api_params)

    scores_text = response.choices[0].message.content.strip()

    # Clean up markdown
    if scores_text.startswith('```json'):
        scores_text = scores_text.replace('```json', '').replace('```', '').strip()
    elif scores_text.startswith('```'):
        scores_text = scores_text.replace('```', '').strip()

    response_data = json.loads(scores_text)

    # Extract scores and reasonings
    scores = []
    reasonings = []

    if 'step_analysis' in response_data:
        step_analyses = response_data['step_analysis']
        analysis_dict = {analysis.get('step', -1): analysis for analysis in step_analyses if analysis.get('step', -1) >= 0}

        print("\n=== LLM Step Analysis (Screenshot-based) ===")
        for i in range(len(game_history)):
            if i in analysis_dict:
                analysis = analysis_dict[i]
                scores.append(analysis.get('score', 0))
                reasonings.append(analysis.get('detailed_reasoning', 'No reasoning'))

                # Display normalized action
                entry = game_history[i]
                raw_action = entry.get('action', 'Unknown')
                state_text = entry.get('state', '')
                try:
                    normalized_data = normalize_action(raw_action, state_text)
                    display_action = normalized_data.get('normalized_action', raw_action)
                except:
                    display_action = raw_action

                print(f"Step {i}: {display_action}")
                print(f"  Reasoning: {analysis.get('detailed_reasoning', 'No reasoning')}")
                print(f"  Score: {analysis.get('score', 0)}")
                print(f"  Observations: {analysis.get('key_observations', 'None')}")
                print()
            else:
                scores.append(0)
                reasonings.append('Missing from LLM analysis')
                print(f"Step {i}: [MISSING - FILLED WITH 0]")
                print()
    else:
        print("[WARNING] No step_analysis in response, falling back to zeros")
        scores = [0] * len(game_history)
        reasonings = ["No analysis available"] * len(game_history)

    return scores, reasonings


def evaluate_step_scores_with_llm(
    game_history,
    state,
    final_score,
    success,
    llm_analysis,
    llm_model="gpt-4-turbo",
    temperature=0.3,
    user_instruction=None,
    task_goal=None,
    screenshots_dir=None,
    evaluate_success=False):
    """
    Use LLM to assign scores for each step in the trajectory.

    Args:
        game_history: List of game history records
        state: Final state
        final_score: Final score
        success: Whether successful (overridden by LLM judgment if evaluate_success=True)
        llm_analysis: LLM analysis result
        llm_model: LLM model to use
        temperature: LLM temperature parameter
        user_instruction: Custom user instruction
        task_goal: Task goal/instruction
        screenshots_dir: Optional screenshots directory path. If provided, uses visual evaluation
        evaluate_success: Whether to let LLM judge task success (default False)

    Returns:
        If evaluate_success=False: tuple: (scores list, reasonings list)
        If evaluate_success=True: tuple: (scores list, reasonings list, llm_success_judgment)
    """
    if not game_history:
        return []

    # Check if using screenshot-based evaluation
    use_screenshots = screenshots_dir is not None and os.path.exists(screenshots_dir)
    if use_screenshots:
        print(f"[INFO] Using screenshot-based evaluation from directory: {screenshots_dir}")

        # For screenshot evaluation, ensure we use a vision-capable model
        vision_model = llm_model
        if llm_model not in ["gpt-4-turbo", "gpt-4-vision-preview", "gpt-4o", "google/gemini-2.5-flash"]:
            vision_model = "gpt-4-turbo"
            print(f"[INFO] Switching from {llm_model} to {vision_model} for vision support")

        return evaluate_with_screenshots(
            game_history=game_history,
            screenshots_dir=screenshots_dir,
            success=success,
            task_goal=task_goal,
            llm_model=vision_model,
            temperature=temperature
        )

    trajectory_text = ""
    for i, entry in enumerate(game_history):
        trajectory_text += f"Step {i}:\n"
        trajectory_text += f"State: {entry.get('state', '')}\n"
        if entry.get('action'):
            raw_action = entry.get('action')
            state_text = entry.get('state', '')

            try:
                normalized_data = normalize_action(raw_action, state_text, normalize_send_msg=False)
                display_action = normalized_data.get('normalized_action', raw_action)
            except:
                display_action = raw_action

            trajectory_text += f"Action: {display_action}\n"
        score = entry.get('score', 0)
        reward = entry.get('reward', 0)
        trajectory_text += f"Reward: {reward}\n"
        trajectory_text += "\n"
    trajectory_text += f"Step: {len(game_history)}:\nState: {state}\n"
    sys_prompt = """You are evaluating web agent actions for a training database.

EVALUATION PROCESS:
1. Analyze: What happened after this action? What was the result?
2. Determine: Was this action USEFUL or HARMFUL (or neither)?
3. Assess: How CERTAIN are you? (certain / somewhat uncertain / very uncertain)
4. Score: Based on usefulness AND certainty

SCORING GUIDE (-3 to +3):
For USEFUL actions:
- **+3**: Clearly useful AND you are certain
- **+2**: Useful but you are somewhat uncertain
- **+1**: Might be useful but you are very uncertain

For HARMFUL/USELESS actions:
- **-3**: Clearly harmful/useless AND you are certain
- **-2**: Harmful/useless but you are somewhat uncertain
- **-1**: Might be harmful/useless but you are very uncertain

For NEUTRAL actions:
- **0**: Cannot determine OR no real effect OR effect immediately undone

MANDATORY FORMAT:
"Result: [what happened]. Usefulness: [useful/harmful/neutral]. Certainty: [certain/somewhat uncertain/very uncertain]. Score: [score] - [repeat/avoid]"
"""


    # Build user instruction section
    user_instruction_section = ""
    if user_instruction:
        user_instruction_section = f"""
USER CUSTOM INSTRUCTION (HIGHEST PRIORITY):
{user_instruction}

You MUST follow the above user instruction when analyzing and scoring the actions.
"""

    # Build task goal section
    task_goal_section = ""
    if task_goal:
        task_goal_section = f"""
TASK GOAL: {task_goal}

"""

    # Build JSON format instructions based on model type and whether we evaluate success
    if evaluate_success:
        success_eval_fields = """,
  "task_success": true,
  "success_reasoning": "Explain why the task succeeded or failed based on task goal and final state"
"""
    else:
        success_eval_fields = ""

    if not _is_openai_model(llm_model):
        # Non-GPT models need detailed JSON formatting instructions
        json_format_instructions = f"""
**CRITICAL: YOU MUST OUTPUT ONLY VALID JSON**

YOUR RESPONSE MUST BE PURE JSON ONLY. Follow these rules strictly:
- Do NOT write any explanatory text, thoughts, or natural language before or after the JSON
- Do NOT use markdown code blocks like ```json ... ``` or ``` ... ```
- Do NOT add comments or notes
- Your response must start with {{ and end with }}
- Use double quotes for all strings
- Ensure all JSON syntax is correct (commas, brackets, braces)

JSON FORMAT (output this structure exactly):
{{
  "step_analysis": [
    {{
      "step": 0,
      "action": "action taken",
      "detailed_reasoning": "MUST follow format: 'Result: [what happened]. Usefulness: [useful/harmful/neutral]. Certainty: [certain/somewhat uncertain/very uncertain]. Score: [number] - [repeat/avoid]'",
      "score": 2,
      "key_observations": "Summary of webpage result"
    }}
  ],
  "overall_assessment": "Brief overall summary"{success_eval_fields}
}}

CRITICAL: "detailed_reasoning" must ALWAYS follow the format: "Result: ... Usefulness: ... Certainty: ... Score: ..."
Score range: -3 to +3 (use ±3 for certain cases, only lower magnitude when uncertain)
OUTPUT ONLY THE JSON, NOTHING ELSE:"""
    else:
        # GPT models support response_format, so just need simple instructions
        json_format_instructions = f"""
JSON FORMAT:
{{
  "step_analysis": [
    {{
      "step": 0,
      "action": "action taken",
      "detailed_reasoning": "MUST follow format: 'Result: [what happened]. Usefulness: [useful/harmful/neutral]. Certainty: [certain/somewhat uncertain/very uncertain]. Score: [number] - [repeat/avoid]'",
      "score": 2,
      "key_observations": "Summary of webpage result"
    }}
  ],
  "overall_assessment": "Brief overall summary"{success_eval_fields}
}}

CRITICAL: "detailed_reasoning" must ALWAYS follow the format: "Result: ... Usefulness: ... Certainty: ... Score: ..."
Score range: -3 to +3 (use ±3 for certain cases, only lower magnitude when uncertain)
Provide complete JSON response:"""

    # Build different prompts based on whether we're evaluating success
    if evaluate_success:
        # When evaluating success, don't tell LLM the result - let it judge
        user_prompt = f"""You are evaluating a web agent's trajectory to determine if it successfully completed the task.

{task_goal_section}
==========================================
YOUR TASK: Evaluate if the agent successfully achieved the task goal
==========================================

Based on the trajectory below, you need to:
1. Score each action based on whether it led to helpful webpage results toward the goal
2. Determine if the OVERALL TASK was completed successfully

For each step, you MUST write "detailed_reasoning" in this exact order:
1. **FIRST: Describe the webpage result** - Start with "After this action, the webpage showed:" and describe what appeared in the NEXT step's State
2. **SECOND: Evaluate if it's helpful** - Write "This result is [helpful/unhelpful] because:" and explain why
   - Helpful: relevant products, correct page, useful data appeared toward the goal
   - Unhelpful: wrong page, irrelevant content, error message appeared
3. **THIRD: Give score with justification** - Write "Score: [number] - Future agents should [repeat/avoid] this"
   - If it led to helpful webpage content → Give POSITIVE score
   - If it led to unhelpful webpage content → Give NEGATIVE score

**CRITICAL: You MUST analyze the webpage result BEFORE giving the score. DO NOT skip step 1 and 2!**

After analyzing all steps, determine:
- task_success: true if the task goal was achieved, false otherwise
- success_reasoning: Explain why the task succeeded or failed based on the final state and task goal

Trajectory:
{trajectory_text}

{user_instruction_section}{json_format_instructions}"""
    else:
        # When not evaluating success, we can use the known success value for scoring
        user_prompt = f"""Score each action based on the WEBPAGE RESULT it produced, NOT on whether the action itself seems reasonable.

{task_goal_section}
==========================================
FINAL TASK RESULT: {"SUCCESS ✓" if success else "FAILURE ✗"}
==========================================

IMPORTANT: Consider the final result when scoring each step.
- This task {"SUCCEEDED - reward actions that led to useful webpage results" if success else "FAILED - penalize actions that led to wrong webpage results"}

For each step, you MUST write "detailed_reasoning" in this exact order:
1. **FIRST: Describe the webpage result** - Start with "After this action, the webpage showed:" and describe what appeared in the NEXT step's State
2. **SECOND: Evaluate if it's helpful** - Write "This result is [helpful/unhelpful] because:" and explain why
   - Helpful: relevant products, correct page, useful data appeared
   - Unhelpful: wrong page, irrelevant content, error message appeared
3. **THIRD: Give score with justification** - Write "Score: [number] - Future agents should [repeat/avoid] this"
   - If it led to helpful webpage content → Give POSITIVE score
   - If it led to unhelpful webpage content → Give NEGATIVE score

**CRITICAL: You MUST analyze the webpage result BEFORE giving the score. DO NOT skip step 1 and 2!**

Trajectory:
{trajectory_text}

{user_instruction_section}{json_format_instructions}"""


    # Build JSON schema based on whether we need success evaluation
    if evaluate_success:
        response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "step_analysis_scores",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "step_analysis": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "step": {"type": "integer"},
                                            "action": {"type": "string"},
                                            "detailed_reasoning": {"type": "string"},
                                            "score": {
                                                "type": "number",
                                                "minimum": -3,
                                                "maximum": 3
                                            },
                                            "key_observations": {"type": "string"}
                                        },
                                        "required": ["step", "action", "detailed_reasoning", "score", "key_observations"],
                                        "additionalProperties": False
                                    }
                                },
                                "overall_assessment": {"type": "string"},
                                "task_success": {
                                    "type": "boolean",
                                    "description": "Whether the task was completed successfully based on the task goal and final state"
                                },
                                "success_reasoning": {
                                    "type": "string",
                                    "description": "Explanation for why the task succeeded or failed"
                                }
                            },
                            "required": ["step_analysis", "overall_assessment", "task_success", "success_reasoning"],
                            "additionalProperties": False
                        }
                    }
                }
    else:
        response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "step_analysis_scores",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "step_analysis": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "step": {"type": "integer"},
                                            "action": {"type": "string"},
                                            "detailed_reasoning": {"type": "string"},
                                            "score": {
                                                "type": "number",
                                                "minimum": -3,
                                                "maximum": 3
                                            },
                                            "key_observations": {"type": "string"}
                                        },
                                        "required": ["step", "action", "detailed_reasoning", "score", "key_observations"],
                                        "additionalProperties": False
                                    }
                                },
                                "overall_assessment": {"type": "string"}
                            },
                            "required": ["step_analysis", "overall_assessment"],
                            "additionalProperties": False
                        }
                    }
                }

    # Outer retry loop for missing steps (max 2 attempts)
    max_missing_retries = 2
    missing_steps_threshold = 0.3  # Retry if more than 30% steps are missing

    for missing_retry in range(max_missing_retries):
        if missing_retry > 0:
            print(f"\n[Retry for Missing Steps {missing_retry}/{max_missing_retries-1}] Too many steps were missing in previous attempt. Retrying with stronger instructions...")
            # Add warning to user_prompt
            user_prompt = f"""**CRITICAL WARNING: In your previous attempt, you missed analyzing many steps. You MUST analyze ALL {len(game_history)} steps from 0 to {len(game_history)-1}. Do not skip any steps!**

""" + user_prompt

        # Inner retry loop for JSON parsing failures (max 3 attempts per missing_retry)
        max_json_retries = 3
        response_data = None

        for attempt in range(max_json_retries):
            try:
                if attempt == 0:
                    print("llm_model:", llm_model)
                else:
                    print(f"[Retry {attempt}/{max_json_retries-1}] Retrying LLM call due to JSON parsing error...")

                response = chat_completion_with_retries(
                            model=llm_model,
                            sys_prompt=sys_prompt,
                            prompt=user_prompt,
                            max_tokens=8192,
                            temperature=temperature,
                            response_format=response_format
                        )
            except Exception as e:
                print(f"[Error] LLM call failed on attempt {attempt+1}: {e}")
                if attempt == max_json_retries - 1:
                    print("[Warning] All LLM call attempts failed. Falling back to reward-based scoring")
                    return [(entry.get('reward') or 0) for entry in game_history], ["No reasoning available" for _ in game_history]
                continue

            if response and hasattr(response, 'choices') and response.choices:
                scores_text = response.choices[0].message.content.strip()

                # Clean up markdown code block wrapper
                if scores_text.startswith('```json'):
                    scores_text = scores_text.replace('```json', '').replace('```', '').strip()
                elif scores_text.startswith('```'):
                    scores_text = scores_text.replace('```', '').strip()

                try:
                    response_data = json.loads(scores_text)
                    # JSON parsing succeeded, break out of retry loop
                    print(f"[Success] JSON parsed successfully on attempt {attempt+1}")
                    break
                except json.JSONDecodeError as e:
                    print(f"[Warning] JSON decode error on attempt {attempt+1}: {e}")
                    print(f"[Warning] Problematic JSON text (first 500 chars): {scores_text[:500]}")
                    if attempt == max_json_retries - 1:
                        print(f"[Warning] All {max_json_retries} attempts failed. Falling back to reward-based scoring")
                        return [(entry.get('reward') or 0) for entry in game_history], ["No reasoning available" for _ in game_history]
            else:
                print(f"[Warning] No valid response from LLM on attempt {attempt+1}")
                if attempt == max_json_retries - 1:
                    print("[Warning] All attempts failed. Falling back to reward-based scoring")
                    return [(entry.get('reward') or 0) for entry in game_history], ["No reasoning available" for _ in game_history]

        # If we got here, we have valid response_data from JSON parsing loop
        if response_data is None:
            print("[Warning] No valid response data after all JSON parsing attempts. Falling back to reward-based scoring")
            return [(entry.get('reward') or 0) for entry in game_history], ["No reasoning available" for _ in game_history]

        if isinstance(response_data, dict):
            if 'step_analysis' in response_data:
                step_analyses = response_data['step_analysis']

                analysis_dict = {}
                for analysis in step_analyses:
                    step_num = analysis.get('step', -1)
                    if step_num >= 0:
                        analysis_dict[step_num] = analysis

                scores = []
                reasonings = []
                missing_steps = []

                print("\n=== LLM Step Analysis ===")
                for i in range(len(game_history)):
                    if i in analysis_dict:
                        analysis = analysis_dict[i]
                        step_score = analysis.get('score', 0)
                        reasoning = analysis.get('detailed_reasoning', 'No reasoning')
                        scores.append(step_score)
                        reasonings.append(reasoning)

                        entry = game_history[i]
                        raw_action = entry.get('action', 'Unknown')
                        state_text = entry.get('state', '')

                        try:
                            normalized_data = normalize_action(raw_action, state_text)
                            display_action = normalized_data.get('normalized_action', raw_action)
                        except:
                            display_action = raw_action

                        print(f"Step {i}: {display_action}")
                        print(f"  Reasoning: {reasoning}")
                        print(f"  Score: {step_score}")
                        print(f"  Observations: {analysis.get('key_observations', 'None')}")
                        print()
                    else:
                        # LLM missed this step, filled with 0
                        scores.append(0)
                        reasonings.append('Missing from LLM analysis - filled with score 0')
                        missing_steps.append(i)
                        print(f"Step {i}: [MISSING FROM LLM ANALYSIS - FILLED WITH 0]")
                        print(f"  Action: {game_history[i].get('action', 'Unknown')}")
                        print(f"  Score: 0 (auto-filled)")
                        print()

                # Check if too many steps are missing, trigger outer retry if needed
                missing_ratio = len(missing_steps) / len(game_history)
                if missing_steps:
                    print(f"[Warning] LLM missed analyzing steps: {missing_steps}")
                    print(f"[Info] Missing ratio: {missing_ratio:.1%} ({len(missing_steps)}/{len(game_history)})")

                    if missing_ratio > missing_steps_threshold and missing_retry < max_missing_retries - 1:
                        print(f"[Warning] Too many missing steps ({missing_ratio:.1%} > {missing_steps_threshold:.0%}). Will retry with stronger instructions.")
                        # Don't return yet - break to outer loop to retry
                        break
                    else:
                        print(f"[Info] Filled missing steps with score 0")

                if 'overall_assessment' in response_data:
                    print(f"Overall Assessment: {response_data['overall_assessment']}")
                    print("=" * 50)

                # Extract success evaluation if requested
                if evaluate_success:
                    llm_success = response_data.get('task_success', False)
                    success_reasoning = response_data.get('success_reasoning', 'No reasoning provided')
                    print(f"\n{'='*50}")
                    print(f"LLM Success Evaluation:")
                    print(f"  Task Success: {llm_success}")
                    print(f"  Reasoning: {success_reasoning}")
                    print(f"{'='*50}\n")
                    return scores, reasonings, llm_success

                # If we got here without breaking, we have acceptable results
                return scores, reasonings
            else:
                print(f"[Warning] No step_analysis found in JSON format: {response_data}")
                return [(entry.get('reward') or 0) for entry in game_history], ["No reasoning available" for _ in game_history]
        elif isinstance(response_data, list):
            scores = response_data
            reasonings = ["Legacy format - no reasoning available" for _ in game_history]
            return scores, reasonings
        else:
            print(f"[Warning] Unexpected JSON format: {response_data}")
            return [(entry.get('reward') or 0) for entry in game_history], ["No reasoning available" for _ in game_history]

    # If we exhausted all missing_retries without returning, fall back to reward-based scoring
    print("[Warning] Exhausted all retries for missing steps. Falling back to reward-based scoring")
    return [(entry.get('reward') or 0) for entry in game_history], ["No reasoning available" for _ in game_history]




def generate_trajectory_context_for_vector(states, actions, earlier_summary,current_step, episode_number, llm_model="gpt-5", temperature=0.3, max_tokens=1000):
    try:

            # Get detailed environment info for current state
        detailed_current_state = find_detailed_environment_info(states, actions, current_step, episode_number, llm_model, temperature)

            # Combine earlier summary with detailed current state
        current_state_text = f"step {current_step}: State: {detailed_current_state}"

        if earlier_summary:
            full_summary = earlier_summary + "\n" # + current_state_text + states[current_step]
        else:
            full_summary = current_state_text
        if detailed_current_state == "":
            detailed_current_state = states[current_step]
        else:
            detailed_current_state = detailed_current_state  + "\n" + states[current_step]
        return full_summary, detailed_current_state

    except Exception as e:
        print(f"[Warning] Failed to generate LLM trajectory context: {e}")
        return generate_trajectory_context_fallback(states, actions, current_step)


def generate_trajectory_context_fallback(states, actions, current_step):
    """
    Fallback method for trajectory context generation if LLM fails.

    Args:
        states: List of all states in the episode
        actions: List of all actions in the episode
        current_step: Current step index

    Returns:
        str: Simple fallback trajectory context
    """
    if not states or current_step < 0 or current_step >= len(states):
        return ""

    # Simple fallback: concatenate states and actions
    context_parts = []
    for i in range(current_step + 1):
        if i < len(states):
            context_parts.append(f"State {i}: {states[i]}")
        if i < len(actions) and actions[i] and i < current_step:
            context_parts.append(f"Action {i}: {actions[i]}")

    return "\n".join(context_parts)


def find_detailed_environment_info(states, actions, current_step, episode_number, llm_model="gpt-5", temperature=0.3, max_tokens=800, max_history_steps=10):
    """
    Find complete environment information based on current_state and game trajectory using LLM.
    Returns empty string if current state has sufficient info, otherwise finds detailed info from history.
    Only considers the most recent max_history_steps states for environment information.

    Args:
        states: List of all states in the episode
        actions: List of all actions in the episode
        current_step: Current step index
        llm_model: LLM model to use
        temperature: LLM temperature parameter
        max_tokens: Maximum tokens
        max_history_steps: Maximum number of recent steps to consider (default: 10)

    Returns:
        str: Complete environment information from history, or empty string if current state is sufficient
    """
    if not states or current_step < 0 or current_step >= len(states):
        return ""
    elif current_step == 0:
        return ""

    current_state = states[current_step]

    # Limit to most recent max_history_steps states for environment search
    start_step = max(0, current_step - max_history_steps + 1)  # Include current step, so look at max_history_steps steps total

    # Build game history text for context (only from recent max_history_steps steps)
    history_text = ""
    for i in range(start_step, current_step):
        history_text += f"Step {i}:\n"
        if i < len(states):
            history_text += f"State: {states[i]}\n"
        if i < len(actions):
            history_text += f"Action: {actions[i]}\n"
        history_text += "\n"

    sys_prompt = """You are a game environment analysis expert. Your task is to analyze the current state and determine if it contains sufficient environment information.

A state has sufficient environment information if it includes location description (room, area, place details) or spatial relationships or layout details.
If the currentstate contains the marker "<<location>>", it ALWAYS has sufficient environment information and you should return "None"

If the current state has sufficient environment information, return "None".
If the current state lacks sufficient information (like "you can't see anything" or other simple prompts), find the most relevant detailed environment description from the game history and return it EXACTLY as it appears in the history, without any modification or reasoning.

PRIORITY RULE: When searching for environment information in the history, ALWAYS prioritize states that contain the "<<location>>" marker, as these contain the most detailed and accurate location information."""

    user_prompt = f"""Analyze the current state and game history:

Game History:
{history_text}

Current State:
{current_state}

Task:
1. First determine if the current state contains sufficient environment information
2. If YES: Return "None"
3. If NO: Find the most relevant detailed environment description from the game history and copy it EXACTLY as it appears, without any changes
4. PRIORITY: When searching the history, give highest priority to states containing "<<location>>" markers as they have the most detailed environment information

Return either "None" or the exact text from history:"""

    try:
        response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if response and hasattr(response, 'choices') and response.choices:
            result = response.choices[0].message.content.strip()

            # Remove triple backticks if present
            if result.startswith('```'):
                result = result.replace('```', '').strip()

            # If result is empty string or very short, current state was sufficient
            if not result or result == 'None' or len(result) < 10:
                print("[Info] Current state has sufficient environment info")
                return ""
            else:

                print(f"[Info] Found detailed environment info from history:\n{result}")
                return result
        else:
            print("[Warning] No response from LLM for environment info analysis")
            return find_environment_info_fallback(states, current_step)

    except Exception as e:
        print(f"[Warning] Failed to analyze environment info with LLM: {e}")
        return find_environment_info_fallback(states, current_step)

def generate_history_summary(current_state=None, llm_model="gpt-5o", temperature=0.3, max_tokens=1000, current_inventory=None):
    """
    Generate a summary of the current web page state.

    Args:
        current_state: Current web page state (axtree text)
        llm_model: LLM model to use
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens for the summary
        current_inventory: Optional current inventory information

    Returns:
        str: A summary of the current web page state
    """
    if not current_state:
        return "No current state available."

    sys_prompt = """You are an expert at analyzing web page states and summarizing key information for web navigation agents.
Your task is to extract and summarize the most important elements, options, and context from the current web page state."""

    user_prompt = f"""Analyze the following current web page state and provide a concise summary of key information.

Current web page state:
{current_state}...

Please provide a structured summary covering:
1. Page type/purpose (e.g., search page, product page, form, results page)
2. Key interactive elements available (buttons, links, input fields) with their IDs
3. Important information displayed on the page

Format the summary to be clear and actionable for a web navigation agent. Focus on elements that can be interacted with."""

    try:
        response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if response and hasattr(response, 'choices') and response.choices:
            summary = response.choices[0].message.content.strip()
            return summary
        else:
            print("[Warning] No response from LLM for summary generation")
            return "No current state available."

    except Exception as e:
        print(f"[Warning] Failed to generate summary with LLM: {e}")
        return "No current state available."


def generate_history_summary_fallback(game_history, current_state):
    """
    Fallback method for generating history summary if LLM fails.

    Args:
        game_history: List of game history entries
        current_state: Current web page state

    Returns:
        str: A simple fallback summary
    """
    if not current_state:
        return "No current state available."

    # Simple fallback: show recent actions and truncated current state
    summary = "Recent browsing context:\n"

    if game_history:
        recent_actions = [entry.get('action', '') for entry in game_history[-3:] if entry.get('action')]
        if recent_actions:
            summary += f"Recent actions: {', '.join(recent_actions)}\n"

    # Add truncated current state
    summary += f"\nCurrent page (truncated): {current_state[:300]}..."

    return summary

def find_environment_info_fallback(states, current_step, max_search_steps=10):
    """
    Fallback method when LLM call fails, finds environment info through simple matching.
    Only considers the most recent max_search_steps states for environment information.

    Args:
        states: List of all states in the episode
        current_step: Current step index
        max_search_steps: Maximum number of recent steps to search (default: 10)

    Returns:
        str: Found environment information or original current_state
    """
    if not states or current_step < 0 or current_step >= len(states):
        return ""

    current_state = states[current_step]

    # If current_state is already detailed (length > 50), return directly
    if len(current_state) > 50:
        return current_state

    # Limit search to most recent max_search_steps states
    start_step = max(0, current_step - max_search_steps + 1)  # Include current step, so look at max_search_steps steps total

    # Search for most recent detailed state description from recent history
    for i in range(current_step, start_step - 1, -1):
        state = states[i]
        if len(state) > 50:  # Consider states with length > 50 as detailed
            # Check if contains location-related information
            location_keywords = ['room', 'hallway', 'corridor', 'chamber', 'area', 'space', 'place']
            if any(keyword in state.lower() for keyword in location_keywords):
                return state

    # If no suitable detailed description found, return last non-empty state from recent history
    for i in range(current_step, start_step - 1, -1):
        state = states[i]
        if state.strip():
            return state

    return current_state


def dump_obs(obs: dict, path: str = "obs_dump.json"):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obs, f, indent=4, ensure_ascii=False, default=str)
        print(f"[INFO] obs dumped to {path}")
    except Exception as e:
        print(f"[ERROR] Failed to dump obs: {e}")


def extract_effective_trajectory_context(
    normalized_actions,
    screenshots_dir,
    current_step,
    llm_model="google/gemini-2.5-flash-preview-09-2025",
    temperature=0.3,
    max_tokens=1000
):
    """
    Extract only the effective actions that led to the current page state.

    This function uses LLM with screenshots to identify which actions in the history
    are actually relevant for reaching the current page state, removing redundant
    and ineffective actions.

    Args:
        normalized_actions: List of normalized action strings from step 0 to current_step-1
        screenshots_dir: Directory containing screenshots (screenshot_step_0.png, etc.)
        current_step: Current step index
        llm_model: LLM model to use (must support vision)
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens for response

    Returns:
        str: A clean trajectory context string with only effective actions (e.g., "action1 -> action2 -> action3")

    Example:
        Input actions: ["select_option(...)", "click(...)", "clear(...)", "clear(...)", "click(...)", ...]
        Output: "select_option(...) -> click(...) -> clear(...)"
    """
    import base64
    from .openai_helpers import _get_client

    # Print original actions for debugging
    print(f"\n{'='*80}")
    print(f"[TRAJECTORY EXTRACTION] Step {current_step}")
    print(f"{'='*80}")
    print(f"Original normalized actions ({len(normalized_actions)} actions):")
    for i, action in enumerate(normalized_actions):
        print(f"  Step {i}: {action}")
    print(f"{'='*80}\n")

    if not normalized_actions or current_step <= 0:
        result = "No action"
        print(f"[TRAJECTORY EXTRACTION RESULT] {result}\n")
        return result

    # Check if we have the current step screenshot
    current_screenshot_path = os.path.join(screenshots_dir, f"screenshot_step_{current_step}.png")
    if not os.path.exists(current_screenshot_path):
        print(f"[Warning] Current screenshot not found at {current_screenshot_path}, falling back to simple join")
        result = " -> ".join(normalized_actions)
        print(f"[TRAJECTORY EXTRACTION RESULT] {result}\n")
        return result

    # Use _get_client to support both OpenAI and OpenRouter
    client = _get_client(llm_model)

    # Build action history text
    actions_text = ""
    for i, action in enumerate(normalized_actions):
        actions_text += f"Step {i}: {action}\n"

    # System prompt
    sys_prompt = """You are an expert at extracting effective action sequences to identify unique page states.

YOUR MISSION:
We want to find similar page states from past browsing sessions to reuse successful actions. We already found pages with the SAME URL, but the same URL can have DIFFERENT STATES (e.g., different filter settings, form values, etc.).

Your job: Extract the MINIMAL action sequence that DEFINES the current page state. This action sequence will be used to match against historical states - if two states have the same URL + same action sequence, they are truly equivalent states.

CORE PRINCIPLE:
Extract actions that DETERMINE the final page state. These are actions whose effects are VISIBLE in the final page.

WHAT TO KEEP:
1. Actions whose effects are VISIBLE in the final page
   - select_option(Show By, Year) → If final page shows "Show By: Year" → KEEP (defines filter state)
   - fill(From, "01/01/2000") → If final page shows "From: 01/01/2000" → KEEP (defines form state)
   - click(tab) → If final page shows that tab selected → KEEP (defines view state)

2. Actions that DETERMINE what content appears
   - Filters, dropdowns, checkboxes that affect displayed data
   - Tab selections, view mode switches
   - Search/query inputs

WHAT TO REMOVE:
1. Actions whose effects are NOT visible (overwritten/replaced)
   - fill(input, "A") -> fill(input, "B") → final shows "B" → REMOVE first fill, KEEP only "B"
   - select_option(X) -> select_option(Y) → final shows Y → REMOVE X, KEEP only Y

2. Actions that don't affect final state
   - Intermediate navigation that led to current page
   - Actions completely erased by later operations

CRITICAL: Refresh/Reload doesn't always reset everything!
- Many states PERSIST after Refresh (filters, selections, etc.)
- Always check the FINAL PAGE to see what's actually there
- If an effect is visible → The action is effective → KEEP IT

WHY THIS MATTERS:
With the correct action sequence, we can find past states that are TRULY equivalent to the current state, and confidently reuse the successful actions from those past experiences."""

    user_prompt = f"""TASK: Extract the action sequence that DEFINES the current page state (Step {current_step}).

CONTEXT:
We're building a memory system that finds similar states from past browsing. We already matched by URL, but need to identify the EXACT STATE. Two pages with the same URL but different filters/forms/tabs are DIFFERENT states.

Your extracted action sequence will be the "state signature" - if a past page has the same URL + same action sequence, we know it's the SAME STATE and can safely reuse its successful actions.

ACTIONS TAKEN (Step 0 to Step {len(normalized_actions) - 1}):
{actions_text}

You'll see screenshots for each step + the FINAL current page.

YOUR PROCESS:
1. Look at the FINAL page carefully - what defines its state?
2. For each action, check: "Is this action's effect VISIBLE in the final page?"
3. If YES → Include it (it defines the state)
4. If NO → Exclude it (it was overwritten or doesn't affect final state)

EXAMPLES:

Example 1 - Filter defines state:
Actions: click(Refresh) -> select_option(Show By, Year) -> click(Refresh)
Final page: Shows "Show By: Year" in dropdown
Analysis: The Year filter is VISIBLE, it defines what data is shown
→ Output: "select_option(<combobox[Show By:]>, \"Year\")"
→ Why: This identifies the state as "reports page filtered by Year"

Example 2 - Overwritten input:
Actions: fill(From, "2020") -> fill(From, "2023")
Final page: Shows "From: 2023"
Analysis: Only "2023" is visible, "2020" was overwritten
→ Output: "fill(<textbox[From:]>, \"2023\")"
→ Why: State is "From=2023", not "From=2020"

Example 3 - Multiple state-defining actions:
Actions: select_option(Show By, Year) -> click(Refresh) -> fill(From, "01/01/2000")
Final page: Shows "Show By: Year" AND "From: 01/01/2000"
Analysis: Both filter and date are visible
→ Output: "select_option(<combobox[Show By:]>, \"Year\") -> fill(<textbox[From:]>, \"01/01/2000\")"
→ Why: State is "Year view + starting from 2000" - both actions define it

REMEMBER: We need to match states EXACTLY. If you miss an action, we might reuse actions from a DIFFERENT state (wrong filter/form) and fail!

OUTPUT FORMAT:

Step 1: Reasoning
- Examine FINAL page: What defines its current state?
- Check each action: Is its effect visible in final page?
- Identify the minimal action sequence that creates this state

Step 2: Final Answer
[Action sequence separated by " -> ", or "No action" if no actions define the state]"""

    # Prepare message content with screenshots
    content = [{"type": "text", "text": user_prompt}]

    # First, add the CURRENT page screenshot prominently
    content.append({
        "type": "text",
        "text": f"\n=== FINAL/CURRENT PAGE (Step {current_step}) - THIS IS WHAT MATTERS ===\nFocus on this page and identify which previous actions are still relevant to it."
    })
    with open(current_screenshot_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_data}",
            "detail": "low"
        }
    })
    content.append({
        "type": "text",
        "text": "=== END OF CURRENT PAGE ===\n"
    })

    # Then add historical screenshots for reference
    content.append({
        "type": "text",
        "text": "\n=== HISTORICAL SCREENSHOTS (for reference only) ===\nCompare these with the current page to identify which actions are still relevant.\n"
    })

    for i in range(len(normalized_actions)):
        screenshot_path = os.path.join(screenshots_dir, f"screenshot_step_{i}.png")
        if os.path.exists(screenshot_path):
            with open(screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            content.append({
                "type": "text",
                "text": f"\n[Step {i}: {normalized_actions[i]}]"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}",
                    "detail": "low"
                }
            })

    try:
        print(f"[INFO] Calling vision API to extract effective trajectory context...")
        # Build API parameters
        api_params = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": content}
            ],
            "temperature": temperature,
        }
        # Use max_completion_tokens for newer models (gpt-4o, o1, etc.)
        if _needs_max_completion_tokens(llm_model):
            api_params["max_completion_tokens"] = max_tokens
        else:
            api_params["max_tokens"] = max_tokens
        response = client.chat.completions.create(**api_params)

        result = response.choices[0].message.content.strip()

        # Clean up any markdown or extra formatting
        if result.startswith('```'):
            result = result.replace('```', '').strip()

        # Parse the two-step format
        # Expected format:
        # Step 1: Reasoning
        # [reasoning text]
        # Step 2: Final Answer
        # [action sequence]

        final_answer = None
        reasoning = None

        # Try to extract Step 2: Final Answer
        if "Step 2:" in result or "Final Answer" in result:
            # Split by "Step 2:" or "Final Answer"
            parts = re.split(r'Step 2:\s*Final Answer', result, flags=re.IGNORECASE)
            if len(parts) == 2:
                reasoning_part = parts[0]
                final_answer_part = parts[1].strip()

                # Extract reasoning from Step 1
                if "Step 1:" in reasoning_part or "Reasoning" in reasoning_part:
                    reasoning_match = re.search(r'Step 1:\s*Reasoning\s*(.+)', reasoning_part, flags=re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning = reasoning_match.group(1).strip()

                # Clean up final answer
                final_answer = final_answer_part.strip().strip('"\'')

                # Print reasoning for debugging
                if reasoning:
                    print(f"[INFO] LLM Reasoning:\n{reasoning}")

                result = final_answer if final_answer else "No action"
                print(f"\n[TRAJECTORY EXTRACTION RESULT] {result}")
                print(f"{'='*80}\n")
                return result

        # Fallback: if the format is not as expected, try to use the entire result
        # But warn the user
        print(f"[WARNING] LLM response not in expected format. Full response:\n{result}")

        # Remove any quotation marks wrapping the entire result
        result = result.strip('"\'')

        # Try to extract just the action sequence (last line often contains the answer)
        lines = [line.strip() for line in result.split('\n') if line.strip()]
        if lines:
            # Use the last non-empty line as the answer
            final_answer = lines[-1].strip('"\'')
            result = final_answer if final_answer else "No action"
            print(f"[INFO] Using last line as answer: {result}")
            print(f"\n[TRAJECTORY EXTRACTION RESULT] {result}")
            print(f"{'='*80}\n")
            return result

        result = "No action"
        print(f"\n[TRAJECTORY EXTRACTION RESULT] {result}")
        print(f"{'='*80}\n")
        return result

    except Exception as e:
        print(f"[Warning] Failed to extract effective trajectory with LLM: {e}")
        print("[Warning] Falling back to simple join")
        result = " -> ".join(normalized_actions)
        print(f"\n[TRAJECTORY EXTRACTION RESULT] {result}")
        print(f"{'='*80}\n")
        return result


def summarize_trajectory_context(trajectory_context, llm_model="gpt-4o", temperature=0.2, max_tokens=600):
    """
    Summarize each step in trajectory_context into a single line:
    "Step X: State: <1-2 sentence page summary>, Action: <action>"

    - Prefers LLM to output summaries for all steps at once.
    - Falls back to regex parsing if LLM fails.
    """
    if not trajectory_context:
        return ""

    sys_prompt = (
        "You convert a multi-step web browsing context into per-step summaries. "
        "For EACH step present, output exactly one summary in this format: "
        "Step <N>:\nState: <1-2 short sentences summarizing distinctive page features>,\nAction: <action or None>. "
        "Do not invent details; use only visible content. Keep summaries crisp and high-signal."
    )

    user_prompt = f"""Given the following browsing context, produce one line per step.

Context:
{trajectory_context}

Rules:
1) One summary per step found in the context (Step 0, Step 1, ...)
2) Format EXACTLY: "Step N:\nState: <summary>\nAction: <action or None>"
3) Summaries are 1-2 short sentences highlighting page purpose and standout UI/content
4) If a step has no explicit action in the context, use "None"
Output only the lines, no extra text.
"""

    try:
        response = chat_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        if response and hasattr(response, 'choices') and response.choices:
            lines = response.choices[0].message.content.strip()
            if lines.startswith('```'):
                lines = lines.replace('```', '').strip()
            one_block = "\n".join([ln.strip() for ln in lines.splitlines() if ln.strip()])
            # Basic validation: at least one "Step N:" line
            if re.search(r"^Step\s+\d+\s*:\s*State\s*:", one_block, re.IGNORECASE | re.MULTILINE):
                return one_block
    except Exception as e:
        print(f"[Warning] Failed to summarize trajectory_context with LLM: {e}")

    # Fallback
    return summarize_trajectory_context_fallback(trajectory_context)


def summarize_trajectory_context_fallback(trajectory_context, max_state_len=220):
    """
    Fallback: Parse multi-step paragraphs and generate "Step N: State: ..., Action: ...".
    """
    if not trajectory_context:
        return ""

    step_iter = list(re.finditer(r"Step\s+(\d+)\s*:\s*", trajectory_context, re.IGNORECASE))
    if not step_iter:
        state_text = None
        for m in re.finditer(r"State\s*:\s*(.+)", trajectory_context, re.IGNORECASE):
            state_text = m.group(1).strip()
        if not state_text:
            state_text = trajectory_context.strip()
        if len(state_text) > max_state_len:
            state_text = state_text[:max_state_len].rstrip() + "..."
        return f"State: {state_text}, Action: None"

    blocks = []
    for idx, m in enumerate(step_iter):
        step_no = int(m.group(1))
        start = m.end()
        end = step_iter[idx + 1].start() if idx + 1 < len(step_iter) else len(trajectory_context)
        block = trajectory_context[start:end]
        blocks.append((step_no, block))

    lines = []
    for step_no, block in blocks:
        state_match = re.search(r"State\s*:\s*(.+)", block, re.IGNORECASE)
        state_text = state_match.group(1).strip() if state_match else block.strip()
        action_match = re.search(r"Action\s*:\s*(.+)", block, re.IGNORECASE)
        action_text = action_match.group(1).strip() if action_match else "None"

        state_text = state_text.replace('\n', ' ').strip()
        if len(state_text) > max_state_len:
            state_text = state_text[:max_state_len].rstrip() + "..."

        lines.append(f"Step {step_no}: State: {state_text}, Action: {action_text}")

    lines.sort(key=lambda s: int(re.search(r"Step\s+(\d+)", s).group(1)))
    return "\n".join(lines)