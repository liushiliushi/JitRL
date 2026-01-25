import os
import json
from autoeval.prompts import *


class Evaluator:
    def __init__(self, lm_clients, log_save_path=None):
        self.lm_clients = lm_clients
        self.log_save_path = log_save_path

    def __call__(self, info, ground_truth, user_instruction, client="gpt-4o", version="naive"):
        assert (client in self.lm_clients), \
            f"Client {client} not found in {self.lm_clients.keys()}"
        if version == "text":
            eval_info, eval_str, prompt = self.eval_text(info, client, ground_truth)
        elif version == "vision":
            eval_info, eval_str, prompt = self.eval_vision(info, client)
        else:
            raise NotImplementedError(f"Version {version} not implemented")

        if self.log_save_path:
            with open(self.log_save_path + "/outputs.jsons", "w") as f:
                f.write(
                    json.dumps(
                        {
                            "id": info["traj_name"],
                            "user_instruction": user_instruction,
                            "eval_llm": client,
                            "eval_info": eval_info,
                        }
                    )
                    + "\n"
                )
            with open(self.log_save_path + "/outputs.jsonl", "w") as f:
                f.write(
                    json.dumps(
                        {
                            "id": info["traj_name"],
                            "user_instruction": user_instruction,
                            "eval_llm": client,
                            "eval_info": eval_info,
                        }
                    )
                    + "\n"
                )
            with open(f"{self.log_save_path}/{info['traj_name']}.md", "w") as md_file:
                md_file.write(f"## Intent\n\n{info['intent']}\n\n")
                md_file.write(f"## RM\n\n{eval_str}\n\n")
                md_file.write(f"## Final Response {info['response']}\n\n")
                
                if "states" in info and info['states']:
                    md_file.write("## States\n\n")
                    for idx, state_info in enumerate(info["states"]):
                        md_file.write(f"### Step {idx}\n")
                        md_file.write(f"**Action**: {state_info.get('action', 'N/A')}\n\n")
                        md_file.write(f"**Reward**: {state_info.get('reward', 0)}\n\n")
                        md_file.write(f"**Score**: {state_info.get('score', 0)}\n\n")
                        md_file.write(f"**State**:\n```\n{state_info.get('state', '')[:500]}...\n```\n\n")
                
                if "captions" in info and info['captions'] is not None:
                    md_file.write("## Captions\n\n")
                    for idx, cap in enumerate(info["captions"]):
                        md_file.write(f"===============")
                        md_file.write(f"{cap}\n")
                md_file.write("\n## Images\n\n")
                for idx, img in enumerate(info["image_paths"]):
                    rel_img_path = os.path.relpath(img, self.log_save_path)
                    md_file.write(f"![Image {idx+1}]({rel_img_path})\n")

                if "config" in info:
                    md_file.write("## Config\n\n")
                    cofig_str = json.dumps(info["config"], indent=4)
                    md_file.write(f"```json\n{cofig_str}\n```\n")
                if prompt:
                    md_file.write("## Main Prompt\n\n")
                    md_file.write(f"```md\n{prompt}\n```\n")
        return eval_info, prompt

    # def eval_text2(self, info, client, ground_truth):
    #     response = info["response"] if info["response"] else "None"
    #     lm_client = self.lm_clients[client]
    #     action_history = ""
    #     for idx, act in enumerate(info["actions"]):
    #         action_history += f"{idx+1}: {act}\n"
        
    #     # Use states data if available, otherwise fall back to captions
    #     if "states" in info and info["states"]:
    #         # Use the state from the last step
    #         final_state = info["states"][-1].get("state", "")
    #         print(f"Using state data from states.json (length: {len(final_state)} chars)")
    #     else:
    #         # Fall back to captions
    #         final_state = info["captions"][-1] if info["captions"] else ""
    #         print(f"Using caption as state (length: {len(final_state)} chars)")
        
    #     prompt, sys_msg = build_text_eval_prompt(
    #         final_state, info["intent"], response, action_history, ground_truth
    #     )
    #     msg_str, _ = lm_client.one_step_chat(prompt, system_msg=sys_msg)
    #     msg_dict = {
    #         "thoughts": extract_content(msg_str, "Thoughts:"),
    #         "status": extract_content(msg_str, "Status:").replace('"', ""),
    #     }
    #     return msg_dict, msg_str, prompt

    def eval_text(self, info, client, ground_truth):
        response = info["response"] if info["response"] else "None"
        lm_client = self.lm_clients[client]

        # Special handling for "N/A" ground truth
        if ground_truth and ground_truth.strip().upper() == "N/A":
            system_msg = """You are an evaluator checking if a bot correctly identified that there is no relevant information or result.

When the reference answer is "N/A", it means there is no expected result (e.g., no matching items, no relevant data).
The bot should correctly recognize and communicate this "empty result" situation.

Format your response as:
Thoughts: <Your reasoning about whether the bot correctly identified the absence of results>
Status: "success" or "failure"
"""

            prompt = f"""Bot's Response: {response}

Expected Situation: There is no relevant information or result (Reference Answer: N/A)

Does the bot's response correctly identify that there is no relevant information/result?
- Look for phrases like "no results", "not found", "no matching items", "N/A", "none", "don't seem to be any", etc.
- If the bot correctly identifies the absence of results, return "success"
- If the bot provides specific information/data (when there should be none), or fails to recognize the empty result, return "failure"
- If the response is empty/error without explanation, return "failure"
"""
        else:
            # Use LLM to check if response contains ground_truth semantically
            system_msg = """You are an evaluator checking if a bot's response contains the reference answer.
Your task is to determine if the bot's response semantically contains the information in the reference answer.

The response doesn't need to match exactly - it should contain the same information, even if expressed differently.

Format your response as:
Thoughts: <Your reasoning about whether the response contains the reference answer>
Status: "success" or "failure"
"""

            prompt = f"""Bot's Response: {response}

Reference Answer: {ground_truth if ground_truth else "N/A"}

Does the bot's response contain the information from the reference answer?
- If yes, return "success"
- If no or if the response is empty/error, return "failure"
"""

        print(f"Evaluating response: {response}")
        print(f"Against ground truth: {ground_truth}")

        msg_str, _ = lm_client.one_step_chat(prompt, system_msg=system_msg)
        msg_dict = {
            "thoughts": extract_content(msg_str, "Thoughts:"),
            "status": extract_content(msg_str, "Status:").replace('"', ""),
        }

        print('======================= Evaluation Result =======================')
        print(f"Evaluation result: {msg_dict['status']}")
        print(f"Thoughts: {msg_dict['thoughts']}")
        print('===============================================================')

        return msg_dict, msg_str, prompt


    def eval_vision(self, info, client):
        assert client == "gpt-4v" or client == "gpt-4o"
        action_history = ""
        for idx, act in enumerate(info["actions"]):
            action_history += f"{idx+1}: {act}\n"
        prompt, sys_msg = build_vision_eval_prompt(
            info["intent"], info["response"], action_history
        )
        img = info["images"][-1]
        
        lm_client = self.lm_clients[client]
        msg_str, _ = lm_client.one_step_chat(
            text=prompt, image=img, system_msg=sys_msg
        )
        del info["images"]
        msg_dict = {
            "thoughts": extract_content(msg_str, "Thoughts:"),
            "status": extract_content(msg_str, "Status:").replace('"', ""),
        }
        return msg_dict, msg_str, prompt
