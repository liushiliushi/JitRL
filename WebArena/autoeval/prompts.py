def build_obs_simplifier_prompt(cap, intent, response) -> str:
    prompt = f"""Given the following user question and context, extract part of the context that is unbiased, so that using that text alone would be good context for providing an unbiased answer to the user query.

**User Query**: The bot responded with "{response}", does it execute this task "{intent}" successfully?

**Full Context**:
```md
{cap}
```

Start your answer with “Unbiased text context (includes all relevant content):"
"""
    return prompt


def build_naive_last_frame_eval_prompt(cap, intent, response) -> str:
    prompt = f"""**User Intent**: {intent}

**Bot's Final Observation**:

```md
{cap}
```    

**Bot's response to the user**: {response if response else "None"}.

---

Based on the provided user intent, the caption of bot's final observation and its response, did the bot successfully execute the task? Please reason step by step.

Note:
- The trajectory descriptions are essentially noisy captions of the screenshots captured during bot's execution. And you should infer what actions the bot took yourself.
- You should categorize the execution into one of the three status:
    - task-possible-bot-success: The bot successfully executed the task.
    - task-possible-bot-fail: The bot failed to execute the task.
    - task-impossible: The task is impossible to execute in nature given the user intent and the environment. For example, if the user wants to buy a product that does not exist in the environment. You should carefully distinguish this from bot-fail.

Format your response as a valid json:
{{
    "thoughts": "{{Your thoughts here, discuss if and how the trajectory progress towards the task and then reason about the final status. You should provide an explicit reason when determining the final status.}}",
    "status": "task-possible-bot-success" or "task-possible-bot-fail" or "task-impossible"
}}"""
    return prompt


def build_naive_multi_frame_eval_prompt(caps, intent, response) -> str:
    captions_str = "\n".join(
        [f"{idx+1}:\n```md\n{caption}\n```\n" for idx, caption in enumerate(caps[-3:])]
    )
    prompt = f"""**User Intent**: {intent}

**Bot's observation through execution**:

{captions_str}

**Bot's response to the user**: {response if response else "None"}.

---

Based on the provided user intent, bot's observation in captions and its response, did the bot successfully execute the task? Please reason step by step.

Note:
- You should categorize the execution into one of the three status:
    - task-possible-bot-success: The bot successfully executed the task.
    - task-possible-bot-fail: The bot failed to execute the task.
    - task-impossible: The task is impossible to execute in nature given the user intent and the environment. For example, if the user wants to buy a product that does not exist in the environment. You should carefully distinguish this from bot-fail.

Format your response as a valid json:
{{
    "thoughts": "{{Your thoughts here, discuss if and how the trajectory progress towards the task and then reason about the final status. You should provide an explicit reason when determining the final status.}}",
    "status": "task-possible-bot-success" or "task-possible-bot-fail" or "task-impossible"
}}"""
    return prompt


def extract_content(text, start_tag):
    """
    Extract the content that follows 'Info:' in a given string.

    :param text: A string that may contain lines starting with 'Info:'
    :return: The content that follows 'Info:' or None if not found
    """
    # Split the text into lines
    lines = text.split("\n")

    # Loop through each line to find a line that starts with 'Info:'
    for line in lines:
        if line.startswith(start_tag):
            # Extract and return the content after 'Info:'
            return line[len(start_tag) :].strip()

    # Return None if 'Info:' is not found in any line
    return ""


def build_text_eval_prompt(
    cap, intent, response, last_actions, ground_truth
) -> tuple[str, str]:
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, map info, comparison of map routes, etc. The bot's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the bot encounters an exception and respond with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked to list the top-searched items in a shop, the agent should order the items by the number of searches, and then return the top items. If the ordering action is missing, the task is likely to fail.
2. Site navigation: The user wants to navigate to a specific page. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.

CRITICAL EVALUATION CRITERIA - EVIDENCE MUST COME FIRST:
IMPORTANT: The correctness of the answer is SECONDARY to whether it can be derived from the webpage. This is crucial to prevent reward hacking.

EVALUATION PRIORITY (IN ORDER):
1. FIRST: Identify any search/filter conditions in the user intent (e.g., specific year, price range, rating, category)
2. SECOND: Check if the webpage shows results that match those conditions
3. THIRD: Check if the webpage contains the information needed to derive the answer
4. FOURTH: Only if all above are YES, then check if the bot's answer is correct

- AUTOMATIC FAILURE (Even if answer is correct):
  * The webpage does NOT contain the specific information the bot claims to have found
  * The bot's answer cannot be derived or verified from what's visible on the current page
  * The information appears to come from somewhere else (memory, prior knowledge, lucky guess)
  * The webpage lacks key details that would be necessary to answer the question
  * **IMPORTANT: If the user intent specifies search/filter conditions (year, category, price, rating, etc.), the webpage must clearly show that these conditions were applied**
  * Example: If asked about "2022 data" but the page only shows "2023 data" or not showing specific time information, it's FAILURE even if the bot somehow provides correct 2022 data
  * Example: If asked about "products under $50" but the page shows unfiltered results or products over $50, it's FAILURE
  * Example: If asked to "search for X", but the page doesn't show search results for X, it's FAILURE

- SUCCESS (Must satisfy ALL):
  * If user intent has search/filter conditions, the webpage clearly shows those conditions were applied correctly
  * The webpage contains all information necessary to derive the answer
  * The bot's answer correctly reflects what's on the page
  * The answer addresses the specific constraints mentioned in the user intent

- KEY PRINCIPLE: We are evaluating "Can a reasonable person looking at ONLY this webpage derive this answer while verifying all user intent conditions were met?" If NO, it's automatic FAILURE regardless of answer correctness.

- WHY: Allowing correct answers without evidence or without proper filtering teaches the agent bad habits (guessing, using prior knowledge, ignoring constraints) that won't generalize to new tasks.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <FIRST identify any search/filter conditions in user intent. SECOND check: Were those conditions properly applied on the webpage? THIRD check: Is the required information present? FOURTH check: Is the bot's answer correct? Explain your reasoning for ALL checks, then provide final status>
Status: "success" or "failure"
"""
    prompt = f"""User Intent: {intent}

Action History:
{last_actions}

The detailed final state of the webpage:

```md
{cap}
```

Bot response to the user: {response if response else "N/A"}.

Evaluate the response of bot for semantic equivalence to the reference, not exact string match.
Reference answer: {ground_truth if ground_truth else "N/A"}.

---

CRITICAL EVALUATION STEPS (Follow this exact order):
1. STOP: First analyze the user intent - does it contain specific search/filter conditions (year, price, category, rating, keywords)?
2. If YES, check: Does the webpage clearly show these conditions were applied? (e.g., filtered results, search query visible, correct category selected)
3. If conditions were NOT properly applied → Automatic FAILURE, regardless of answer correctness
4. Examine the "detailed final state of the webpage" above carefully
5. Ask: "Can the bot's answer be derived purely from what's visible on this page?"
6. If the answer is NO (information not on page) → Automatic FAILURE, regardless of answer correctness
7. If the answer is YES (information is on page AND conditions properly applied) → Then check if bot's response matches the reference
8. Mark as "success" ONLY if: ALL conditions applied correctly AND information is on page AND answer is correct
9. Mark as "failure" if: conditions not applied OR information missing OR answer incorrect

REMEMBER: Proper constraint application and evidence presence are more important than answer correctness. A lucky correct guess is still a FAILURE.

"""
    return prompt, system_msg


def build_vision_eval_prompt(
    intent, response, last_actions
) -> tuple[str, str]:
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, map info, comparison of map routes, etc. The bot's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the bot encounters an exception and respond with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked to list the top-searched items in a shop, the agent should order the items by the number of searches, and then return the top items. If the ordering action is missing, the task is likely to fail.
2. Site navigation: The user wants to navigate to a specific page. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Carefully examine the bot's action history and the final state of the webpage to determine whether the bot successfully completes the task. No need to consider the bot's response.

CRITICAL EVALUATION CRITERIA - EVIDENCE MUST COME FIRST:
IMPORTANT: The correctness of the answer is SECONDARY to whether it can be derived from the visible webpage screenshot. This is crucial to prevent reward hacking.

EVALUATION PRIORITY (IN ORDER):
1. FIRST: Identify any search/filter conditions in the user intent (e.g., specific year, price range, rating, category)
2. SECOND: Check if the screenshot shows results that match those conditions
3. THIRD: Check if the screenshot contains the information needed to derive the answer
4. FOURTH: Only if all above are YES, then check if the bot's answer is correct

- AUTOMATIC FAILURE (Even if answer is correct):
  * The screenshot does NOT contain the specific information the bot claims to have found
  * The bot's answer cannot be derived or verified from what's visible in the screenshot
  * The information appears to come from somewhere else (memory, prior knowledge, lucky guess)
  * The screenshot lacks key details that would be necessary to answer the question
  * **IMPORTANT: If the user intent specifies search/filter conditions (year, category, price, rating, etc.), the screenshot must clearly show that these conditions were applied**
  * Example: If asked about "2022 data" but the screenshot only shows "2023 data" or not showing specific time information, it's FAILURE even if the bot somehow provides correct 2022 data
  * Example: If asked about "products under $50" but the screenshot shows unfiltered results or products over $50, it's FAILURE
  * Example: If asked to "search for X", but the screenshot doesn't show search results for X, it's FAILURE

- SUCCESS (Must satisfy ALL):
  * If user intent has search/filter conditions, the screenshot clearly shows those conditions were applied correctly
  * The screenshot contains all information necessary to derive the answer
  * The bot's answer correctly reflects what's visible in the screenshot
  * The answer addresses the specific constraints mentioned in the user intent

- KEY PRINCIPLE: We are evaluating "Can a reasonable person looking at ONLY this screenshot derive this answer while verifying all user intent conditions were met?" If NO, it's automatic FAILURE regardless of answer correctness.

- WHY: Allowing correct answers without evidence or without proper filtering teaches the agent bad habits (guessing, using prior knowledge, ignoring constraints) that won't generalize to new tasks.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <FIRST identify any search/filter conditions in user intent. SECOND check: Were those conditions properly applied (visible in the screenshot)? THIRD check: Is the required information visible? FOURTH check: Is the bot's answer correct? Explain your reasoning for ALL checks, then provide final status>
Status: "success" or "failure"
"""
    prompt = f"""User Intent: {intent}

Action History:
{last_actions}

Bot response to the user: {response if response else "N/A"}.

The last snapshot of the web page is shown in the image.

---

CRITICAL EVALUATION STEPS (Follow this exact order):
1. STOP: First analyze the user intent - does it contain specific search/filter conditions (year, price, category, rating, keywords)?
2. If YES, check: Does the screenshot clearly show these conditions were applied? (e.g., filtered results, search query visible, correct category selected)
3. If conditions were NOT properly applied → Automatic FAILURE, regardless of answer correctness
4. Carefully examine the webpage screenshot
5. Ask: "Can the bot's answer be derived purely from what's visible in this screenshot?"
6. If the answer is NO (information not in screenshot) → Automatic FAILURE, regardless of answer correctness
7. If the answer is YES (information is in screenshot AND conditions properly applied) → Then check if bot's response is correct
8. Mark as "success" ONLY if: ALL conditions applied correctly AND information is visible AND answer is correct
9. Mark as "failure" if: conditions not applied OR information not visible OR answer incorrect

REMEMBER: Proper constraint application and evidence presence are more important than answer correctness. A lucky correct guess is still a FAILURE."""
    return prompt, system_msg
