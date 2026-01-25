import os
import json
import time
import random
import numpy as np
import statistics
from typing import Tuple, List, Dict, Any
import src.utils as utils
from .memory_agent import MemoryAgent
from .our_agent import OurAgent
from .naive_agent import NaiveAgent
from .awm_agent import AWMAgent
from .env import JerichoEnv

class StateNode:
    def __init__(self, state, reward=0.0):
        self.state = state
        self.reward = reward 
        self.response = ""

class GameEvaluator:
    
    def __init__(self, args):
        self.args = args
        self.base_output_path = os.path.join(self.args.output_path, self.args.game_name)
        os.makedirs(self.base_output_path, exist_ok=True)
        if self.args.seed is None:
            self.args.seed = random.randint(0, 100000)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        self.args.rom_path = os.path.join(self.args.rom_path, utils.game_file(self.args.game_name))
    
    def run_evaluation(self) -> Dict[str, Any]:

        print(f"\n--- STARTING EVALUATION ---")
        print(f"Game: {self.args.game_name}, Agent LLM Model: {self.args.llm_model}")
        print(f"Agent Type: {self.args.agent_type}, Runs for statistics: {self.args.eval_runs}")
        print(f"Base Seed for evaluation session: {self.args.seed}")
        
        model_slug = self.args.llm_model.replace('/', '_').replace('\\', '_')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(self.base_output_path, self.args.agent_type, model_slug, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
                
        print(f"Episode logs and summary will be in: {self.log_dir}")
        
        if self.args.agent_type == 'memory':
            agent = MemoryAgent(self.args, guiding_prompt="")
        elif self.args.agent_type == 'our':
            agent = OurAgent(self.args, guiding_prompt="")
        elif self.args.agent_type == 'naive':
            agent = NaiveAgent(self.args, guiding_prompt="")
        elif self.args.agent_type == 'awm':
            agent = AWMAgent(self.args, guiding_prompt="")
        else:
            raise ValueError(f"Unknown agent type: {self.args.agent_type}.")
        scores = []
        for i in range(self.args.eval_runs):
            print(f"\nStarting game run {i+1}/{self.args.eval_runs}...")
            current_score, _ = self.run_game_episode(agent, episode_num=i)
            scores.append(current_score)
            print(f"Run {i+1}/{self.args.eval_runs} finished. Score: {current_score}")
            
            average_score = statistics.mean(scores)
            stdev_score = statistics.stdev(scores) if len(scores) > 1 else 0
            
            print(f"\n--- EVALUATION SUMMARY ({self.args.agent_type} method) ---")
            print(f"Agent LLM Model: {self.args.llm_model}")
            print(f"Number of final evaluation runs: {self.args.eval_runs}")
            print(f"Individual Scores: {scores}")
            print(f"Average Score: {average_score:.2f}")
            print(f"Standard Deviation: {stdev_score:.2f}")
            
            model_slug = self.args.llm_model.replace('/', '_').replace('\\', '_')
            summary_file_name = f"evaluation_summary_{self.args.agent_type}_{model_slug}_{timestamp}.json"
            summary_path = os.path.join(self.log_dir, summary_file_name)
            
            summary_data = {
                "individual_scores": scores,
                "average_score": average_score,
                "stdev_score": stdev_score,
                "timestamp_evaluation_session_start": timestamp,
                "timestamp_summary_created": time.strftime("%Y%m%d-%H%M%S"),
                "all_args": vars(self.args),
                "success": True
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=4)

        return summary_data


    def run_game_episode(self, agent, episode_num: int = 0):
        self.args.llm_model_name_for_log = self.args.llm_model.replace('/', '_') # for logging
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file_name = f"episode_{episode_num:03d}_{timestamp}.txt"
        log_file_path = os.path.join(self.log_dir, log_file_name)
        os.makedirs(self.log_dir, exist_ok=True)

        episode_specific_seed = self.args.seed + episode_num if self.args.seed is not None else random.randint(0, 1000000)

        env = JerichoEnv(rom_path=self.args.rom_path,
                        seed=episode_specific_seed, 
                        step_limit=self.args.env_step_limit)

        ob, info = env.reset()
        score = info['score']
        
        with open(log_file_path, 'w', encoding='utf-8') as output:
            output.write(f"Game: {self.args.game_name}, Model: {self.args.llm_model}, Episode: {episode_num}, Seed for episode: {episode_specific_seed}\n")
            agent.start_episode()
            output.write(f"Guiding prompt: {agent.guiding_prompt}\n")

            for step in range(self.args.env_step_limit):
                state_node = StateNode(state=ob)

                # Add missing action for technician interaction
                if 'technician' in ob.lower() and 'copier technician' not in ob.lower():
                    if 'ask technician about gate' not in info['valid']:
                        info['valid'].append('ask technician about gate')

                action, raw_llm_output = agent.generate_action(state_node, info=info)

                output.write(f"==========\n")
                output.write(f"[STEP] {step}\n")
                output.write(f"----------\n")
                output.write(f"[OBS] {ob}\n")
                output.write(f"----------\n")
                output.write(f"[INV] {info.get('inv', 'N/A')}\n")
                valid_actions = info.get('valid', []) if self.args.use_valid_actions else None
                if valid_actions:
                    output.write(f"[VALID_ACTIONS] {valid_actions}\n")
                output.write(f"[RAW_LLM_OUTPUT] {raw_llm_output}\n")
                output.write(f"[CHOSEN_ACTION] {action}\n")
                output.write(f"----------\n")
                prev_ob = ob
                ob, reward, done, info = env.step(action)
                score = info['score']
                output.write(f"[REWARD] {reward}\n")
                output.write(f"[CUM_REWARD] {score}\n")  
                output.write(f"----------\n")

                # Feed reward and score back into the agent for cross-episode memory
                try:
                    agent.update_game_history_reward(reward, score)
                except Exception:
                    pass

                prev_ob_short = prev_ob[:70].replace('\n', ' ').ljust(70)
                print(f"[STEP {step}] | {prev_ob_short} || Action: {action.ljust(25)} || Reward: {reward}, Score: {score}")
                
                if done:
                    output.write(f"\nGame finished at step {step}. Final score: {score}\n")
                    break
            if not done:
                output.write(f"\nStep limit ({self.args.env_step_limit}) reached. Final score: {score}\n")
            agent.end_episode(state=ob, score=score)
            
        return score, log_file_path
