"""
Enhanced prompt update generation that references high-scoring episodes.
"""

import json
import os
from typing import List, Dict, Any, Optional
from .openai_helpers import claude_completion_with_retries


def load_episodes_history(episodes_jsonl_path: str) -> List[Dict[str, Any]]:
    """Load all episodes from episodes.jsonl file."""
    episodes = []
    if os.path.exists(episodes_jsonl_path):
        with open(episodes_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
    return episodes


def get_top_episodes(episodes: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Get the top K episodes by final_score."""
    # Sort by final_score in descending order
    sorted_episodes = sorted(episodes, key=lambda x: x.get('final_score', -999), reverse=True)
    return sorted_episodes[:top_k]


def extract_episode_summary(episode: Dict[str, Any], max_steps: int = 30) -> str:
    """Extract a concise summary of an episode's key actions and rewards."""
    steps = episode.get('steps', [])
    final_score = episode.get('final_score', 0)
    num_steps = len(steps)

    summary = f"Episode Score: {final_score} (in {num_steps} steps)\n"
    summary += "Key actions:\n"

    # Extract steps with positive rewards or high LLM scores
    important_steps = []
    for step in steps[:max_steps]:  # Limit to first max_steps
        reward = step.get('reward', 0)
        llm_score = step.get('llm_step_score', 0)

        # Include steps with positive rewards or high LLM scores
        if reward > 0 or llm_score >= 3:
            important_steps.append({
                'step_num': step.get('step_num', 0),
                'action': step.get('action', ''),
                'reward': reward,
                'score': step.get('score', 0),
                'llm_score': llm_score,
                'reasoning': step.get('llm_reasoning', '')[:150]  # Truncate reasoning
            })

    # Show the important steps
    for step in important_steps[:15]:  # Limit to 15 important steps
        summary += f"  Step {step['step_num']}: {step['action']} "
        summary += f"(Reward: {step['reward']:+d}, Score: {step['score']}, LLM: {step['llm_score']})\n"
        if step['reasoning']:
            summary += f"    → {step['reasoning']}\n"

    if len(important_steps) > 15:
        summary += f"  ... and {len(important_steps) - 15} more important steps\n"

    return summary


def generate_prompt_with_history(
    game_history: List[Dict[str, Any]],
    final_score: float,
    success: bool,
    current_prompt: str,
    episodes_jsonl_path: str,
    llm_model: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 0.8,
    max_tokens: int = 3000,
    top_k_episodes: int = 3
) -> Dict[str, Any]:
    """
    Generate a prompt update recommendation based on current episode AND historical high-scoring episodes.

    Args:
        game_history: Complete game history from current episode
        final_score: Final score achieved in current episode
        success: Whether current episode was successful
        current_prompt: The current guiding prompt being used
        episodes_jsonl_path: Path to episodes.jsonl containing historical episodes
        llm_model: LLM model to use
        temperature: LLM temperature parameter
        max_tokens: Maximum tokens for the response
        top_k_episodes: Number of top episodes to reference (default 3)

    Returns:
        dict: Contains 'recommended_prompt', 'reasoning', and 'key_insights'
    """

    if not game_history:
        return {
            'recommended_prompt': current_prompt,
            'reasoning': 'No game history available for analysis',
            'key_insights': []
        }

    # Load historical episodes
    all_episodes = load_episodes_history(episodes_jsonl_path)
    print(f"[Info] Loaded {len(all_episodes)} historical episodes")

    # Get top K high-scoring episodes
    top_episodes = get_top_episodes(all_episodes, top_k=top_k_episodes)
    print(f"[Info] Top {len(top_episodes)} episodes scores: {[ep.get('final_score', 0) for ep in top_episodes]}")

    # Build current episode trajectory summary
    current_trajectory_text = ""
    for i, entry in enumerate(game_history[:40]):  # Limit to first 40 steps
        current_trajectory_text += f"Step {i}:\n"
        current_trajectory_text += f"State: {entry.get('state', '')[:150]}...\n"
        if entry.get('action'):
            current_trajectory_text += f"Action: {entry.get('action')}\n"
        if entry.get('reward') is not None:
            current_trajectory_text += f"Reward: {entry.get('reward')}\n"
        current_trajectory_text += "\n"

    if len(game_history) > 40:
        current_trajectory_text += f"... (omitted {len(game_history) - 40} more steps)\n"

    # Build high-scoring episodes summaries
    historical_summaries = ""
    for i, episode in enumerate(top_episodes, 1):
        historical_summaries += f"\n{'='*60}\n"
        historical_summaries += f"HIGH-SCORING EPISODE #{i}\n"
        historical_summaries += f"{'='*60}\n"
        historical_summaries += extract_episode_summary(episode)
        historical_summaries += "\n"

    sys_prompt = """You are an expert game strategy analyst specializing in creating PRECISE, ACTIONABLE step-by-step instructions.

You will be given:
1. The CURRENT episode trajectory (which just finished)
2. MULTIPLE HIGH-SCORING episodes from history (successful past attempts)

Your job is to:
1. Compare the current episode with the high-scoring episodes
2. Identify what the high-scoring episodes did RIGHT that the current episode missed
3. Identify common successful patterns across multiple high-scoring episodes
4. Create an EXTREMELY DETAILED strategic prompt with EXACT action sequences

CRITICAL ANALYSIS FRAMEWORK:
When comparing episodes, focus on:
- **Exact Action Sequences**: What PRECISE sequence of commands do high-scoring episodes execute? (e.g., 'open window' then 'west' then 'take sword')
- **Critical Ordering**: Which actions MUST happen in a specific order? (e.g., 'echo' BEFORE 'take bar')
- **Repetition Counts**: How many times should certain actions repeat? (e.g., 'wait' exactly 7 times)
- **Time-Sensitive Actions**: Which actions have timing constraints?
- **Common Mistakes**: What specific actions do failed episodes execute that successful ones avoid?

OUTPUT FORMAT:
You must output in JSON format with these fields:

{
    "comparative_analysis": {
        "current_episode_score": <score>,
        "top_episodes_scores": [<score1>, <score2>, ...],
        "what_current_did_well": ["specific action/pattern 1", "specific action/pattern 2"],
        "what_current_missed": ["specific opportunity 1", "specific opportunity 2"],
        "common_success_patterns": ["exact pattern 1 seen in multiple high-scoring episodes", "exact pattern 2"],
        "key_differences": "Describe the main differences between current episode and high-scoring ones"
    },
    "reasoning": {
        "useful_actions": ["exact action 1 that helped", "exact action 2 that helped"],
        "useless_actions": ["exact action 1 that wasted time", "exact action 2 that failed"],
        "successful_sequence_from_top_episodes": "Describe the EXACT common successful sequence from high-scoring episodes with specific commands",
        "patterns_to_avoid": ["specific pattern 1", "specific pattern 2"],
        "strategic_insights": "What overall strategy emerges from comparing all episodes?"
    },
    "recommended_prompt": "<SEE DETAILED FORMAT BELOW>",
    "key_insights": ["insight 1", "insight 2", "insight 3"]
}

CRITICAL: RECOMMENDED_PROMPT FORMAT REQUIREMENTS

Your recommended_prompt MUST follow these rules:

1. **Use EXPLICIT commands in backticks**:
   ✅ GOOD: "Execute \`open window\`, then \`west\`"
   ❌ BAD: "Enter the house through the window"

2. **Specify EXACT repetition counts**:
   ✅ GOOD: "Execute \`wait\` exactly 7 times"
   ❌ BAD: "Wait for a while"

3. **Include CRITICAL ordering with warnings**:
   ✅ GOOD: "⚠️ CRITICAL: In Loud Room, FIRST execute \`echo\`, wait 1 turn, THEN execute \`take bar\`"
   ❌ BAD: "Get the bar from the Loud Room"

4. **Add expected score checkpoints**:
   ✅ GOOD: "After taking bar, verify score increased to 50 (+10 points)"
   ❌ BAD: "Take the bar"

5. **Use numbered steps for complex sequences**:
   ✅ GOOD: "1) \`north\` 2) \`take sword\` 3) \`take lantern\`"
   ❌ BAD: "Get equipment from the north"

6. **Specify exact navigation**:
   ✅ GOOD: "Execute \`east\` 3 times to reach Loud Room"
   ❌ BAD: "Go to the Loud Room"

7. **Include conditional logic where needed**:
   ✅ GOOD: "If score < 50, you missed the bar; go back to Loud Room"
   ❌ BAD: "Continue exploring"

EXAMPLE OF EXCELLENT recommended_prompt:
"First, execute the entry sequence: \`north\` (behind house), \`open window\`, \`west\` (kitchen). In Living Room: \`take sword\`, \`take lantern\`, \`turn on lantern\`, \`push rug\`, \`open trap\`, \`down\`. ⚠️ CRITICAL: Immediately \`north\` to troll, then \`hit troll with sword\` repeatedly until troll flees. Execute \`east\` to East-West Passage, then \`east\` 3 times to Loud Room. ⚠️ CRITICAL SEQUENCE: 1) \`echo\` 2) wait 1 turn 3) \`take bar\` (expect +10 points, score→50). Next, navigate to Dam (\`up\`, \`east\`), then \`north\` twice to Maintenance Room, \`take wrench\`, \`take screwdriver\`. Return \`south\` twice to Dam, execute \`push bolt with wrench\`. Finally, go \`west\` to Reservoir South, execute \`wait\` exactly 7 times, then \`north\` 3 times to reach reservoir bed and \`take jewels\` (expect +25 points)."

EXAMPLE OF BAD recommended_prompt:
"Enter the house, get equipment, defeat the troll, find treasures in various rooms, and try to maximize your score."

IMPORTANT GUIDELINES:
- Your recommended_prompt should be EXTREMELY SPECIFIC with exact commands
- Include ALL critical action sequences from high-scoring episodes
- Mark time-sensitive or order-critical steps with ⚠️ (use the unicode character, not emoji codes)
- Include score checkpoints (e.g., "expect score to be 50 at this point")
- Specify exact repetition counts (e.g., "execute `wait` 7 times", not "wait several times")
- Use backticks for all game commands
- Length: Aim for 3-8 sentences with dense, actionable information
- If current episode score is already among the best, PRESERVE the successful sequence and only add minor refinements

CRITICAL JSON FORMATTING:
- Output ONLY valid JSON, no extra text before or after
- Do NOT use backslash escapes in the JSON strings (e.g., use `north` not \`north\`)
- The backticks inside strings do NOT need escaping in JSON
- Use double quotes for JSON keys and string values
- Example of correct JSON:
  {"recommended_prompt": "Execute `north`, then `open window`"}
  NOT: {"recommended_prompt": "Execute \\`north\\`, then \\`open window\\`"}
"""

    user_prompt = f"""Analyze the current episode and compare it with historical high-scoring episodes to create improved strategic guidance.

CURRENT GUIDING PROMPT:
"{current_prompt}"

CURRENT EPISODE RESULT:
- Final Score: {final_score}
- Success: {"Yes" if success else "No"}
- Total Steps: {len(game_history)}

CURRENT EPISODE TRAJECTORY:
{current_trajectory_text}

{'='*80}
HISTORICAL HIGH-SCORING EPISODES FOR REFERENCE:
{'='*80}
{historical_summaries}

{'='*80}
TASK:
{'='*80}

1. COMPARE the current episode with the high-scoring episodes above
2. IDENTIFY what successful episodes did that current episode didn't (or vice versa)
3. FIND common patterns that appear across MULTIPLE high-scoring episodes
4. CREATE an improved strategic prompt that incorporates the best practices from top episodes

CRITICAL QUESTIONS TO ANSWER WITH EXACT COMMANDS:
1. What are the EXACT first 5-10 commands (in backticks) that high-scoring episodes consistently execute?
2. Which key items/locations do ALL top episodes reach, and what EXACT commands get them there?
3. What did the current episode do differently (better or worse) in terms of SPECIFIC actions?
4. If current episode scored lower: What SPECIFIC command sequences did it miss that top episodes used?
5. If current episode scored well: What EXACT new command sequences did it discover?
6. Are there critical ACTION ORDERINGS (e.g., must do A before B) that the current episode violated?
7. How many times did successful episodes repeat certain commands (e.g., \`wait\` count, \`hit troll\` count)?

Based on this comparative analysis, generate:
- A detailed comparative_analysis showing SPECIFIC action differences between current and top episodes
- Comprehensive reasoning about what EXACT actions work and what EXACT actions don't
- An improved recommended_prompt with PRECISE command sequences, using backticks for every game command

REMEMBER - YOUR PROMPT MUST BE EXTREMELY PRECISE:
- Use backticks for EVERY game command (e.g., \`take sword\`, \`open window\`)
- Specify EXACT repetition counts (e.g., "execute \`wait\` 7 times", not "wait")
- Include order-critical warnings with ⚠️ (e.g., "⚠️ MUST \`echo\` BEFORE \`take bar\`")
- Add score checkpoints (e.g., "expect score 50 after bar, 75 after jewels")
- Give EXACT navigation (e.g., "\`east\` 3 times" not "go to Loud Room")
- PRIORITIZE patterns that appear in MULTIPLE top episodes (more reliable)
- If current episode found something NEW that led to success, incorporate the EXACT commands
- Your prompt should be a STEP-BY-STEP action sequence, not high-level goals
"""

    try:
        print(f"[Info] Generating prompt recommendation with historical context using {llm_model}")

        result_text = claude_completion_with_retries(
            model=llm_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        if not result_text:
            print("[Warning] Failed to get Claude response, falling back to current prompt")
            return {
                'recommended_prompt': current_prompt,
                'reasoning': 'Failed to get Claude response',
                'key_insights': []
            }

        # Parse JSON response
        import re
        result = None

        # Try 1: Direct JSON parsing
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"[Warning] JSON parse error: {e}")

            # Try 2: Extract from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    print("[Info] Successfully extracted JSON from markdown code block")
                except json.JSONDecodeError as e2:
                    print(f"[Warning] Markdown JSON parse failed: {e2}")

            # Try 3: Find any JSON object
            if result is None:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        print("[Info] Successfully extracted JSON via regex")
                    except json.JSONDecodeError as e3:
                        print(f"[Warning] Regex JSON parse failed: {e3}")

            # Try 4: Clean up common escape issues and retry
            if result is None:
                # Replace problematic backslash sequences
                cleaned_text = result_text
                # Fix common escape issues
                cleaned_text = cleaned_text.replace('\\n', '\\\\n')
                cleaned_text = cleaned_text.replace('\\t', '\\\\t')
                # Remove any remaining single backslashes before quotes
                cleaned_text = re.sub(r'\\([^"\\nrtbf/u])', r'\1', cleaned_text)

                try:
                    result = json.loads(cleaned_text)
                    print("[Info] Successfully parsed JSON after cleanup")
                except json.JSONDecodeError as e4:
                    print(f"[Warning] Cleanup parse failed: {e4}")

            # Give up if nothing worked
            if result is None:
                print("[Warning] Could not parse JSON response after all attempts, using default")
                print(f"[Debug] First 500 chars of response: {result_text[:500]}")
                return {
                    'recommended_prompt': current_prompt,
                    'reasoning': 'Could not parse LLM response',
                    'key_insights': []
                }

        # Extract the key fields
        recommended_prompt = result.get('recommended_prompt', current_prompt)
        reasoning_obj = result.get('reasoning', {})
        comparative_analysis = result.get('comparative_analysis', {})

        # Build comprehensive reasoning text
        reasoning_text = ""
        if comparative_analysis:
            reasoning_text += "=== COMPARATIVE ANALYSIS ===\n"
            reasoning_text += f"Current Score: {comparative_analysis.get('current_episode_score', final_score)}\n"
            reasoning_text += f"Top Episodes Scores: {comparative_analysis.get('top_episodes_scores', [])}\n"
            reasoning_text += f"\nWhat current episode did well:\n"
            for item in comparative_analysis.get('what_current_did_well', []):
                reasoning_text += f"  + {item}\n"
            reasoning_text += f"\nWhat current episode missed:\n"
            for item in comparative_analysis.get('what_current_missed', []):
                reasoning_text += f"  - {item}\n"
            reasoning_text += f"\nCommon success patterns:\n"
            for item in comparative_analysis.get('common_success_patterns', []):
                reasoning_text += f"  * {item}\n"
            reasoning_text += f"\nKey differences: {comparative_analysis.get('key_differences', '')}\n"

        if reasoning_obj:
            reasoning_text += "\n=== STRATEGIC REASONING ===\n"
            reasoning_text += f"Strategic Insights: {reasoning_obj.get('strategic_insights', '')}\n"

        key_insights = result.get('key_insights', [])

        return {
            'recommended_prompt': recommended_prompt,
            'reasoning': reasoning_text or result.get('reasoning', 'No detailed reasoning provided'),
            'key_insights': key_insights,
            'comparative_analysis': comparative_analysis,
            'full_result': result
        }

    except Exception as e:
        print(f"[Error] Failed to generate prompt with history: {e}")
        import traceback
        traceback.print_exc()
        return {
            'recommended_prompt': current_prompt,
            'reasoning': f'Error during generation: {str(e)}',
            'key_insights': []
        }
