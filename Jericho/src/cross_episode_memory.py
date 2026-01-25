import os
import json
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from .utils import generate_history_summary, generate_trajectory_summary, calculate_summary_similarity, evaluate_step_scores_with_llm, generate_trajectory_context_for_vector, generate_prompt_recommendation
from .prompt_update_with_history import generate_prompt_with_history
from .openai_helpers import get_embedding_with_retries

try:
    import faiss
except ImportError:
    print("Warning: faiss not installed. Please install with: pip install faiss-cpu")
    faiss = None


class CrossEpisodeMemory:
    def __init__(self, base_dir: str, gamma: float = 0.95, llm_model: str = "gpt-4", eval_llm_model: str = None, vector_dim: int = 1536, embedding_model: str = "text-embedding-ada-002"):
        self.base_dir = base_dir
        self.gamma = gamma
        self.llm_model = llm_model
        self.eval_llm_model = eval_llm_model or llm_model  # Use eval_llm_model if provided, otherwise use llm_model
        self.vector_dim = vector_dim
        self.embedding_model = embedding_model
        os.makedirs(self.base_dir, exist_ok=True)
        self.episodes_path = os.path.join(self.base_dir, 'episodes.jsonl')
        self.episodes_abstract_path = os.path.join(self.base_dir, 'episode_abstract.jsonl')
        self.prompt_history_path = os.path.join(self.base_dir, 'prompt_history.jsonl')
        self.summary_cache = {}  # Cache for episode summaries
        self.step_summaries = []  # Store step summaries during episode
        self.step_context_cache = []  # Cache for trajectory_context and current_env_info for each step

        # Initialize episode counter (will be set properly in add_episode)
        self.current_episode_number = 0

        # Initialize vector database
        self._init_vector_database()

    def _init_vector_database(self):
        """Initialize dual Faiss vector databases (history + state) and OpenAI embedding."""
        # Paths for dual indexes
        self.history_index_path = os.path.join(self.base_dir, 'history_vectors.index')
        self.state_index_path = os.path.join(self.base_dir, 'state_vectors.index')
        self.step_metadata_path = os.path.join(self.base_dir, 'step_metadata.pkl')

        # Check if OpenAI embedding is available (skip test for now to avoid API call during init)
        self.encoder_available = True
        print(f"Using OpenAI embedding model: {self.embedding_model} (dimension: {self.vector_dim})")

        # Initialize or load dual Faiss indexes
        if faiss is not None and self.encoder_available:
            # History index (for historical summary)
            if os.path.exists(self.history_index_path):
                self.history_index = faiss.read_index(self.history_index_path)
            else:
                self.history_index = faiss.IndexFlatIP(self.vector_dim)

            # State index (for current state)
            if os.path.exists(self.state_index_path):
                self.state_index = faiss.read_index(self.state_index_path)
            else:
                self.state_index = faiss.IndexFlatIP(self.vector_dim)

            # Load metadata
            if os.path.exists(self.step_metadata_path):
                with open(self.step_metadata_path, 'rb') as f:
                    self.step_metadata = pickle.load(f)
            else:
                self.step_metadata = []
        else:
            self.history_index = None
            self.state_index = None
            self.step_metadata = []
            if faiss is None:
                print("Warning: Faiss not available, vector database disabled")
            elif not self.encoder_available:
                print("Warning: OpenAI embedding not available, vector database disabled")
    
    def _save_vector_database(self):
        """Save the dual Faiss indexes and metadata to disk."""
        if self.history_index is not None:
            faiss.write_index(self.history_index, self.history_index_path)
        if self.state_index is not None:
            faiss.write_index(self.state_index, self.state_index_path)
        if self.step_metadata:
            with open(self.step_metadata_path, 'wb') as f:
                pickle.dump(self.step_metadata, f)
    
    def clear_vector_database(self, save_to_disk: bool = True):
        """
        Clear all vectors and metadata from the dual vector databases, and optionally delete prompt history.

        Args:
            save_to_disk: If True, also delete the saved files on disk (including prompt history)
        """
        print("Clearing dual vector database and prompt history...")

        history_count = 0
        state_count = 0

        if self.history_index is not None:
            history_count = self.history_index.ntotal
            self.history_index.reset()
            print(f"✓ History index cleared (was {history_count} vectors)")

        if self.state_index is not None:
            state_count = self.state_index.ntotal
            self.state_index.reset()
            print(f"✓ State index cleared (was {state_count} vectors)")

        # Clear metadata
        cleared_count = len(self.step_metadata)
        self.step_metadata = []
        print(f"✓ Metadata cleared ({cleared_count} entries removed)")

        if save_to_disk:
            # Remove saved files
            if os.path.exists(self.history_index_path):
                os.remove(self.history_index_path)
                print(f"✓ Deleted {self.history_index_path}")

            if os.path.exists(self.state_index_path):
                os.remove(self.state_index_path)
                print(f"✓ Deleted {self.state_index_path}")

            if os.path.exists(self.step_metadata_path):
                os.remove(self.step_metadata_path)
                print(f"✓ Deleted {self.step_metadata_path}")

            # Delete prompt history
            if os.path.exists(self.prompt_history_path):
                os.remove(self.prompt_history_path)
                print(f"✓ Deleted {self.prompt_history_path}")

            print("✓ Vector database and prompt history files deleted from disk")
        else:
            # Save empty database
            self._save_vector_database()
            print("✓ Empty database saved to disk")

        print("Vector database and prompt history cleared successfully!")
    
    def get_vector_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dual vector databases.

        Returns:
            Dictionary containing database statistics
        """
        stats = {
            'total_history_vectors': 0,
            'total_state_vectors': 0,
            'total_metadata_entries': len(self.step_metadata),
            'vector_dimension': self.vector_dim,
            'database_available': self.history_index is not None and self.state_index is not None and self.encoder_available,
            'history_index_exists': os.path.exists(self.history_index_path),
            'state_index_exists': os.path.exists(self.state_index_path),
            'metadata_file_exists': os.path.exists(self.step_metadata_path)
        }

        if self.history_index is not None:
            stats['total_history_vectors'] = self.history_index.ntotal

        if self.state_index is not None:
            stats['total_state_vectors'] = self.state_index.ntotal

        if stats['history_index_exists']:
            stats['history_index_size'] = os.path.getsize(self.history_index_path)

        if stats['state_index_exists']:
            stats['state_index_size'] = os.path.getsize(self.state_index_path)

        if stats['metadata_file_exists']:
            stats['metadata_file_size'] = os.path.getsize(self.step_metadata_path)

        return stats
    
    def _encode_trajectory_context(self, states: List[str], actions: List[str], current_step: int, current_summary: str, info=None) -> Optional[np.ndarray]:
        """Encode trajectory context as dual vectors (history + state) using LLM summarization.

        Args:
            states: List of all states in the episode
            actions: List of all actions in the episode
            current_step: Current step index
            current_summary: Summary of history up to previous step
            info: Game environment info dictionary

        Returns:
            Tuple of (trajectory_context, current_env_info, history_vector, state_vector) or (None, None, None, None) if encoder is not available
        """
        if not self.encoder_available:
            return None, None, None, None

        # First generate earlier summary using our updated function
        # Create game_history from states and actions for the summary
        if current_step == 40:
            stop = 1
        earlier_summary = current_summary

        # Use LLM to generate optimized trajectory context
        trajectory_context, current_env_info = generate_trajectory_context_for_vector(
            states=states,
            actions=actions,
            earlier_summary=earlier_summary,
            current_step=current_step,
            episode_number=self.current_episode_number,
            llm_model=self.llm_model,
            temperature=0.8,
            max_tokens=1000,
            info=info
        )



        # Encode history summary separately (if available)
        history_vector = None
        if current_summary:
            history_vector = get_embedding_with_retries(current_summary, model=self.embedding_model)
            if history_vector is None:
                print("Warning: Failed to get history embedding, using zero vector")
                history_vector = np.zeros(self.vector_dim, dtype=np.float32)
        else:
            # No history yet (first step), use zero vector
            history_vector = np.zeros(self.vector_dim, dtype=np.float32)

        # Encode current state separately
        current_state_text = f"step {current_step}: State: {current_env_info}"
        state_vector = get_embedding_with_retries(current_state_text, model=self.embedding_model)

        if state_vector is None:
            print("Failed to get state embedding")
            return None, None, None, None

        # Normalize vectors for cosine similarity
        history_vector = history_vector / (np.linalg.norm(history_vector) + 1e-8)
        state_vector = state_vector / (np.linalg.norm(state_vector) + 1e-8)

        return trajectory_context, current_env_info, history_vector, state_vector
    
    def _store_step_in_vector_db(self, states: List[str], actions: List[str], step_index: int, episode_data: Dict[str, Any]):
        """Store a step in the dual vector databases (history + state).

        Args:
            states: List of all states in the episode
            actions: List of all actions in the episode
            step_index: Index of the current step
            episode_data: Data for the entire episode
        """
        if self.history_index is None or self.state_index is None or not self.encoder_available:
            return

        # Use cached trajectory context and vectors if available
        trajectory_context, current_env_info, history_vector, state_vector = self.step_context_cache[step_index]

        # Get step data from episode
        steps = episode_data.get('steps', [])
        if step_index >= len(steps):
            return
        step_data = steps[step_index]

        # Extract future rewards from current step to end of episode
        future_rewards = []

        for u in range(step_index, len(steps)):
            future_rewards.append(steps[u].get('llm_step_score', 0) if steps[u].get('llm_step_score', 0) != 0 else steps[u].get('reward', 0))

        # Prepare step metadata
        step_metadata = {
            'episode_timestamp': episode_data.get('timestamp'),
            'episode_number': self.current_episode_number,  # Add episode number
            'step_data': step_data.copy(),  # All step information
            'episode_final_score': episode_data.get('final_score'),
            'episode_success': episode_data.get('success'),
            'trajectory_context': trajectory_context,
            'current_env_info': current_env_info,  # Store current environment info
            'future_rewards': future_rewards  # Store reward sequence for flexible discounting
        }

        # Add vectors to dual Faiss indexes
        self.history_index.add(history_vector.reshape(1, -1))
        self.state_index.add(state_vector.reshape(1, -1))

        # Store metadata
        self.step_metadata.append(step_metadata)

        print(f"Stored step {step_index} in dual vector database (history: {self.history_index.ntotal}, state: {self.state_index.ntotal})")

    def search_similar_steps(self, query_states: List[str], query_actions: List[str], k: int = 5, current_summary: str = None) -> List[Dict[str, Any]]:
        """Search for similar steps in the dual vector database.

        Args:
            query_states: List of states in the query
            query_actions: List of actions in the query trajectory
            k: Number of similar steps to retrieve
            current_summary: Summary of history up to previous step

        Returns:
            List of similar steps with similarity scores
        """

        # Encode the query trajectory as dual vectors
        trajectory_context, current_env_info, query_hist_vec, query_state_vec = self._encode_trajectory_context(
            query_states, query_actions, len(query_states) - 1, current_summary
        )

        self.step_context_cache.append((trajectory_context, current_env_info, query_hist_vec, query_state_vec))

        if query_hist_vec is None or query_state_vec is None:
            return []
        if self.history_index is None or self.state_index is None or not self.encoder_available:
            return []
        if self.history_index.ntotal == 0 or self.state_index.ntotal == 0:
            return []

        # Search in both indexes
        hist_scores, hist_indices = self.history_index.search(query_hist_vec.reshape(1, -1), k)
        state_scores, state_indices = self.state_index.search(query_state_vec.reshape(1, -1), k)

        # Combine scores (weighted: 0.25 history + 0.75 state)
        results = []
        for i in range(min(k, len(hist_indices[0]))):
            hist_idx = hist_indices[0][i]
            state_idx = state_indices[0][i]

            if hist_idx == state_idx and hist_idx < len(self.step_metadata):
                # Same index found in both - ideal match
                combined_score = 0.25 * hist_scores[0][i] + 0.75 * state_scores[0][i]
                result = self.step_metadata[hist_idx].copy()
                result['similarity_score'] = float(combined_score)
                result['hist_sim'] = float(hist_scores[0][i])
                result['state_sim'] = float(state_scores[0][i])
                result['rank'] = i + 1
                results.append(result)

        return results

    def retrieve_similar_with_vector(self, game_history: List[Dict[str, Any]], current_state: str, current_summary: str, k: int = 3, r: float = 0.7, info=None) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar trajectory segments using dual vector similarity search (history + state).

        Args:
            game_history: Current game history
            current_state: The current state to append to the trajectory
            current_summary: Summary of history up to previous step
            k: Number of similar trajectories to retrieve
            r: Similarity threshold (cosine similarity score)
            info: Game environment info dictionary

        Returns:
            List of similar trajectories with similarity scores
        """
        # Extract states and actions from current game history
        query_states = [entry.get('state', '') for entry in game_history]
        query_actions = [entry.get('action', '') for entry in game_history]

        # Add current state
        query_states.append(current_state)

        # Calculate dynamic similarity threshold based on step count
        step_count = len(game_history)
        max_steps = 20
        # Gradually decrease threshold from r to r-0.1 as step count increases
        if step_count >= max_steps:
            dynamic_threshold = r - 0.1
        else:
            # Linear interpolation: from r (at step 0) to r-0.1 (at max_steps)
            threshold_decrease = 0.1 * (step_count / max_steps)
            dynamic_threshold = r - threshold_decrease

        print(f"Step count: {step_count}, Dynamic threshold: {dynamic_threshold:.3f}")

        # Encode current trajectory as dual vectors
        trajectory_context, current_env_info, query_history_vec, query_state_vec = self._encode_trajectory_context(
            query_states, query_actions, len(query_states) - 1, current_summary, info=info
        )


        query_current_tokens = self._tokenize(current_env_info)
        query_history_tokens = self._tokenize(trajectory_context)
        self.step_context_cache.append((trajectory_context, current_env_info, query_history_vec, query_state_vec))

        if self.history_index is None or self.state_index is None or not self.encoder_available:
            print("Dual vector database not available")
            return []
        if self.history_index.ntotal == 0 or self.state_index.ntotal == 0:
            print("Dual vector database is empty")
            return []
        if query_history_vec is None or query_state_vec is None:
            print("Failed to encode query trajectory")
            return []

        # Search in both indexes with limited candidates for efficiency
        # Phase 1: Recall top candidates using vector similarity (fast)
        # Use 10x more candidates than final k to ensure good coverage
        recall_size = min(max(k * 10, 100), self.history_index.ntotal)

        history_scores, history_indices = self.history_index.search(query_history_vec.reshape(1, -1), recall_size)
        state_scores, state_indices = self.state_index.search(query_state_vec.reshape(1, -1), recall_size)

        # Combine dual vector scores with weighted average
        # Create a mapping of index to combined score
        combined_scores = {}
        for hist_score, hist_idx in zip(history_scores[0], history_indices[0]):
            if hist_idx >= len(self.step_metadata):
                continue
            combined_scores[hist_idx] = {'history': float(hist_score), 'state': 0.0}

        for state_score, state_idx in zip(state_scores[0], state_indices[0]):
            if state_idx >= len(self.step_metadata):
                continue
            if state_idx not in combined_scores:
                combined_scores[state_idx] = {'history': 0.0, 'state': float(state_score)}
            else:
                combined_scores[state_idx]['state'] = float(state_score)

        # Process results and separate by similarity thresholds
        high_similarity_trajectories = []  # similarity > 0.98
        medium_similarity_trajectories = []  # similarity > dynamic_threshold but <= 0.98
        all_trajectories = []  # all trajectories for fallback

        for idx, scores_dict in combined_scores.items():
            hist_sim = scores_dict['history']
            state_sim = scores_dict['state']


            metadata = self.step_metadata[idx]
            step_data = metadata.get('step_data', {})
            history_current_state = metadata.get('current_env_info', {})
            history_past_states = metadata.get('trajectory_context', [])
            current_tokens = self._tokenize(history_current_state)
            history_tokens = self._tokenize(history_past_states)
            sim2 = self._jaccard(query_current_tokens, current_tokens, ngram=4)
            sim1 = self._jaccard(query_history_tokens, history_tokens, ngram=1)
            similarity = sim1*0.3 + sim2*0.7

            # Calculate discounted return using stored future rewards and current gamma
            future_rewards = metadata.get('future_rewards', [])
            discounted_return = 0.0
            for u, step_reward in enumerate(future_rewards):
                llm_step_score = step_data.get('llm_step_score', 0)
                discount = self.gamma ** u
                discounted_return += discount * (step_reward if step_reward != 0 else llm_step_score)
                # discounted_return += discount * (step_reward + llm_step_score)

            result = {
                'step': step_data,
                'trajectory_context': metadata.get('trajectory_context', ''),
                'current_env_info': metadata.get('current_env_info', ''),
                'action': step_data.get('action', ''),
                'hist_sim': hist_sim,  # History similarity score
                'state_sim': state_sim,  # State similarity score
                'jaccard_sim2': sim2,  # Jaccard similarity
                'jaccard_sim1': sim1,  # Jaccard similarity
                'similarity': similarity,  # Final similarity (0.3*sim1 + 0.7*sim2)
                'reward': step_data.get('reward', 0),
                'llm_step_score': step_data.get('llm_step_score', 0),
                'discounted_reward': discounted_return,
                'episode_final_score': metadata.get('episode_final_score', 0),
                'episode_number': metadata.get('episode_number', 0),
                'step_index': step_data.get('step_num', 0)
            }

            # Store all trajectories for potential fallback
            all_trajectories.append((similarity, discounted_return, result))

            # Filter by dynamic threshold for normal operation
            if similarity < dynamic_threshold:
                continue
            # if hist_sim < 0.98:
            #     continue
            if similarity > 0.98:
                high_similarity_trajectories.append((similarity, discounted_return, result))
            else:
                medium_similarity_trajectories.append((similarity, discounted_return, result))

        # Sort both lists by similarity first, then by discounted reward
        high_similarity_trajectories.sort(key=lambda x: (x[0], x[1]), reverse=True)
        medium_similarity_trajectories.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Combine results: first all high similarity (>0.98), then top k from medium similarity if needed
        filtered_trajectories = high_similarity_trajectories[:]
        if len(filtered_trajectories) < k:
            remaining_needed = k - len(filtered_trajectories)
            filtered_trajectories.extend(medium_similarity_trajectories[:remaining_needed])
        
        # Debug output
        if filtered_trajectories:
            print("\n=== Vector-based Retrieval ===")
            print(f"Current trajectory context:\n{trajectory_context}")
            print(f"Current info:\n{current_env_info}")

            bottom_k = min(5, len(filtered_trajectories))
            print(f"\nBottom {bottom_k} matches:")
            for i in range(bottom_k):
                trajectory_data = filtered_trajectories[-i][2]
                # print(f"\n------------------------------------------ Match {i+1} ------------------------------------------")
                # print(f"history info: ")
                # print(trajectory_data['trajectory_context'])
                # print("current info: ")
                # print(trajectory_data['current_env_info'])
                # print(f"hist_sim: {trajectory_data['hist_sim']:.4f}")
                # print(f"jaccard_sim1: {trajectory_data['jaccard_sim1']:.4f}")
                # print(f"jaccard_sim2: {trajectory_data['jaccard_sim2']:.4f}")
                # print(f"Similarity: {trajectory_data['similarity']:.4f}")
                # print(f"Action: {trajectory_data['action']}")
                # print(f"Reward: {trajectory_data['reward']}") if trajectory_data['reward'] != 0 else print(f"Reward: {trajectory_data['llm_step_score']}")            
            print("="*50)
        else:
            # If no trajectories pass the threshold, show top 5 by similarity
            print("\n=== Vector-based Retrieval (No matches above threshold) ===")
            print(f"Current trajectory context:\n{trajectory_context}")
            print(f"Current info:\n{current_env_info}")
            # print(f"Dynamic threshold: {dynamic_threshold:.3f}")

            if all_trajectories:
                # Sort all trajectories by similarity
                all_trajectories.sort(key=lambda x: x[0], reverse=True)

                top_k = min(1, len(all_trajectories))
                print(f"\nTop {top_k} similarity matches (below threshold):")
                for i in range(top_k):
                    trajectory_data = all_trajectories[i][2]
                    # print(f"\n********************************************** Match {i+1} **********************************************")
                    # print(f"history info: ")
                    # print(trajectory_data['trajectory_context'])
                    # print(f"current info: ")
                    # print(trajectory_data['current_env_info'])
                    # print(f"hist_sim: {trajectory_data['hist_sim']:.4f}")
                    # print(f"jaccard_sim1: {trajectory_data['jaccard_sim1']:.4f}")
                    # print(f"jaccard_sim2: {trajectory_data['jaccard_sim2']:.4f}")
                    # print(f"Similarity: {trajectory_data['similarity']:.4f} (below threshold)")
                    # print(f"Action: {trajectory_data['action']}")
                    # print(f"Reward: {trajectory_data['reward']}") if trajectory_data['reward'] != 0 else print(f"Reward: {trajectory_data['llm_step_score']}")
                print("*"*50)
            else:
                print("No trajectories found in database.")
        
        # Return results: all high similarity (>0.98) + additional medium similarity if needed to reach k
        return filtered_trajectories

    # -------------------- Persistence helpers --------------------
    def _append_jsonl(self, path: str, obj: Dict[str, Any]):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    # -------------------- Episode storage --------------------
    def add_episode(self,
                   game_history: List[Dict[str, Any]],
                   state: str,
                   final_score: float = 0,
                   success: bool = False):
        """
        Store a complete episode with all steps.

        Args:
            game_history: List of game history entries containing state, action, score, reward, etc.
            final_score: Final score of the episode
            success: Whether the episode was successful
        """
        # Set current episode number (1-based)
        existing_episodes = self.load_episodes()
        self.current_episode_number = self.current_episode_number + 1

        # Extract data from game_history
        states = [entry.get('state', '') for entry in game_history]
        actions = [entry.get('action', '') for entry in game_history]
        scores = [entry.get('score', 0) for entry in game_history]
        rewards = [entry.get('reward', 0) for entry in game_history]

        # Calculate delta scores
        delta_scores = []
        prev_score = 0
        for entry in game_history:
            current_score = entry.get('score', 0) or 0
            delta = current_score - prev_score
            delta_scores.append(delta)
            prev_score = current_score

        num_steps = len(states) - 1 if success and len(states) > 0 else len(states)

        episode = {
            'timestamp': os.path.basename(self.base_dir),  # Use directory name as timestamp
            'num_steps': len(states),
            'final_score': final_score,
            'success': success,
            'steps': [],
        }

        print(f"Evaluating step scores for episode with {len(game_history)} steps using model: {self.eval_llm_model}...")
        step_scores, step_reasonings = evaluate_step_scores_with_llm(
                game_history=game_history,
                state=state,
                final_score=final_score,
                success=success,
                llm_model=self.eval_llm_model,
                temperature=0.8
            )
        print(f"Generated step scores: {step_scores}")
        print(f"Generated step reasonings: {len(step_reasonings)} entries")

        # Build steps list
        for i in range(num_steps):
            step = {
                'step_num': i,
                'state': states[i] if i < len(states) else '',
                'action': actions[i] if i < len(actions) else None,
                'step_summary': self.step_summaries[i] if i < len(self.step_summaries) else '',
                'reward': rewards[i],
                'score': scores[i],
                'delta_score': delta_scores[i],
                'llm_step_score': step_scores[i] if i < len(step_scores) else 0,
                'llm_reasoning': step_reasonings[i] if i < len(step_reasonings) else 'No reasoning available'
            }
            episode['steps'].append(step)

        if not success and episode['steps']:
            episode['steps'][-1]['reward'] = -10

        # Store all steps in vector database after episode is complete
        for i in range(num_steps):
            self._store_step_in_vector_db(states, actions, i, episode)

        self._append_jsonl(self.episodes_path, episode)
        
        # Save vector database to disk
        self._save_vector_database()
        self.step_context_cache = []
        # Generate and save abstract episode with summaries
        # self._save_abstract_episode(game_history, final_score, success)

    def _save_abstract_episode(self, game_history: List[Dict[str, Any]], final_score: float, success: bool):
        """
        Save abstract episode using stored step summaries.

        Args:
            game_history: List of game history entries
            final_score: Final score of the episode
            success: Whether the episode was successful
        """
        if not game_history:
            return

        abstract_episode = {
            'timestamp': os.path.basename(self.base_dir),
            'num_steps': len(game_history),
            'final_score': final_score,
            'success': success,
            'steps': []
        }

        # Use stored summaries for each step
        for i in range(len(game_history)):
            # Use stored summary if available, otherwise use fallback
            if i < len(self.step_summaries):
                summary = self.step_summaries[i]
            else:
                summary = f"Step {i}: [State: no summary available]"

            # Create abstract step with summary instead of full state
            abstract_step = {
                'step_num': i,
                'summary': summary,
                'action': game_history[i].get('action', ''),
                'reward': game_history[i].get('reward', 0),
                'score': game_history[i].get('score', 0)
            }

            # Add delta score if available
            if i == 0:
                abstract_step['delta_score'] = game_history[i].get('score', 0) or 0
            else:
                prev_score = game_history[i-1].get('score', 0) or 0
                current_score = game_history[i].get('score', 0) or 0
                abstract_step['delta_score'] = current_score - prev_score

            abstract_episode['steps'].append(abstract_step)

        # Save abstract episode
        self._append_jsonl(self.episodes_abstract_path, abstract_episode)

        # Clear step summaries for next episode
        self.step_summaries = []

    def load_episodes(self) -> List[Dict[str, Any]]:
        """Load all stored episodes."""
        return self._load_jsonl(self.episodes_path)

    def load_abstract_episodes(self) -> List[Dict[str, Any]]:
        """Load all stored abstract episodes with summaries."""
        return self._load_jsonl(self.episodes_abstract_path)

    # -------------------- Similarity retrieval --------------------
    def _tokenize(self, text: str) -> List[str]:
        return [t for t in (text or '').lower().replace('\n', ' ').split() if t.isalpha() or t.isalnum()]

    # Convert tokens to n-grams
    def _get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        if len(tokens) < n:
            return tokens  # Return original tokens if too short for n-grams
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram_str = ' '.join(tokens[i:i+n])
            ngrams.append(ngram_str)
        return ngrams
    def _jaccard(self, a_tokens: List[str], b_tokens: List[str], ngram: int = 3) -> float:

        # Get n-grams for both token lists
        a_ngrams = self._get_ngrams(a_tokens, ngram)
        b_ngrams = self._get_ngrams(b_tokens, ngram)

        # Use multiset (Counter) to preserve frequency information
        from collections import Counter
        counter_a = Counter(a_ngrams)
        counter_b = Counter(b_ngrams)

        if not counter_a or not counter_b:
            return 0.0

        # Intersection: min count for each element
        inter = sum((counter_a & counter_b).values())
        # Union: max count for each element
        union = sum((counter_a | counter_b).values())

        return inter / union if union > 0 else 0.0

    def retrieve_similar_with_summary(self, game_history, current_state: str, current_summary: str, k: int = 3, r=0.9, use_summary: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar trajectory segments using LLM-generated summaries.
        
        Args:
            game_history: Current game history
            current_state: The current state to append to the trajectory
            k: Number of similar trajectories to retrieve
            r: Similarity threshold
            use_summary: Whether to use LLM summaries (True) or traditional method (False)
        
        Returns:
            List of similar trajectories with similarity scores
        """
        if use_summary:
            return self._retrieve_with_summary(current_summary, k, r)
        else:
            return self._retrieve_with_tokens(game_history, current_state, k, r)
    
    # def _retrieve_with_summary(self, current_summary, k: int = 3, r=0.9) -> List[Dict[str, Any]]:
    #     """
    #     Retrieve similar trajectories using LLM-generated summaries.
    #     """

    #     self.step_summaries.append(current_summary)
        
    #     episodes = self.load_episodes()
    #     scored_trajectories = []
        
    #     for episode in episodes:
    #         steps = episode.get('steps', [])
    #         if not steps:
    #             continue
            
    #         # Select all steps starting from step 1
    #         # Build game history for this window
    #         window_history = []
    #         for end_idx in range(len(steps)): 
    #             step = steps[end_idx]
    #             window_history.append({
    #                     'state': step.get('state', ''),
    #                     'action': step.get('action', ''),
    #                     'score': step.get('score'),
    #                     'reward': step.get('reward')
    #                 })
                
    #             # Use stored step_summary if available, otherwise generate new one
    #             if 'step_summary' in step and step['step_summary']:
    #                 summary = step['step_summary']
    #             else:
    #                 # Fallback to generating summary if not stored
    #                 summary = generate_trajectory_summary(
    #                         game_history=window_history,
    #                         llm_model=self.llm_model,
    #                         temperature=0.8,
    #                         max_tokens=500
    #                     )
                
    #             # print("=============window history=============")
    #             # print(window_history)
    #             # print(f"============summary================")
    #             # print(summary)
    #             # Calculate similarity between summaries using LLM
    #             sim = calculate_summary_similarity(
    #                 current_summary, 
    #                 summary,
    #                 llm_model=self.llm_model,
    #                 temperature=0.3
    #             )
    #             # print(sim)
                
    #             # Calculate discounted return from the END of the window
    #             discounted_return = 0.0
    #             for u in range(end_idx, len(steps)):
    #                 step_reward = steps[u].get('reward', 0)
    #                 llm_step_score = steps[u].get('llm_step_score', 0)
    #                 if step_reward == 0:
    #                     step_reward = llm_step_score
    #                 discount = self.gamma ** (u - end_idx)
    #                 discounted_return += discount * step_reward
                
    #             result = {
    #                 'step': steps[end_idx],
    #                 'trajectory_summary': summary,
    #                 'action': steps[end_idx].get('action', ''),
    #                 'similarity': sim,
    #                 'reward': steps[end_idx].get('reward', 0),
    #                 'discounted_reward': discounted_return,
    #                 'episode_final_score': episode.get('final_score', 0),
    #                 'window_end_idx': end_idx
    #             }
    #             scored_trajectories.append((sim, discounted_return, result))
        
    #     # Sort by similarity first, then by discounted reward
    #     scored_trajectories.sort(key=lambda x: (x[0], x[1]), reverse=True)
    #     filtered_trajectories = [traj for traj in scored_trajectories if traj[0] >= r]
        
    #     # Debug output
    #     if scored_trajectories:
    #         print("\n=== Summary-based Retrieval ===")
    #         print(f"Current trajectory summary:\n{current_summary}")
            
    #         top_k = min(5, len(scored_trajectories))
    #         print(f"\nTop {top_k} matches:")
    #         for i in range(top_k):
    #             trajectory_data = scored_trajectories[i][2]
    #             print(f"\n--- Match {i+1} ---")
    #             print(f"Summary: {trajectory_data['trajectory_summary']}")
    #             print(f"Similarity: {trajectory_data['similarity']:.4f}")
    #             print(f"Action: {trajectory_data['action']}")
    #             print(f"Discounted reward: {trajectory_data['discounted_reward']:.4f}")
    #         print("="*50)
        
    #     return filtered_trajectories[:k]
    
    # def _retrieve_with_tokens(self, game_history, current_state: str, k: int = 3, r=0.9) -> List[Dict[str, Any]]:
    #     """
    #     Original token-based retrieval method (renamed from retrieve_similar).
    #     """
    #     # Original implementation
    #     query_states = [entry['state'] for entry in game_history]
    #     query_actions = [entry['action'] for entry in game_history]
        
    #     query_states.append(current_state)
        
    #     if not query_actions:
    #         query_actions = []
        
    #     episodes = self.load_episodes()
    #     scored_trajectories = []
        
    #     # Use all query states and actions (no sliding window)
    #     query_states_window = query_states
    #     query_actions_window = query_actions
        
    #     query_trajectory = ""
    #     for i, state in enumerate(query_states_window):
    #         query_trajectory += f"STATE: {state}\n"
    #         if i < len(query_actions_window) and query_actions_window[i]:
    #             query_trajectory += f"ACTION: {query_actions_window[i]}\n"
    #     query_trajectory_tokens = self._tokenize(query_trajectory)
        
    #     for episode in episodes:
    #         steps = episode.get('steps', [])
    #         if not steps:
    #             continue
            
    #         for end_idx in range(len(steps)):
    #             # Always start from step 0 (first step)
    #             window_start = 0
    #             window_size = end_idx + 1
                
    #             trajectory = ""
    #             for i in range(window_start, end_idx + 1):
    #                 step = steps[i]
    #                 trajectory += f"STATE: {step.get('state', '')}\n"
    #                 if step.get('action') and i < end_idx:
    #                     trajectory += f"ACTION: {step.get('action', '')}\n"
                
    #             trajectory_tokens = self._tokenize(trajectory)
    #             sim = self._jaccard(query_trajectory_tokens, trajectory_tokens)
                
    #             discounted_return = 0.0
    #             for u in range(end_idx, len(steps)):
    #                 step_reward = steps[u].get('reward', 0)
    #                 discount = self.gamma ** (u - end_idx)
    #                 discounted_return += discount * step_reward
                
    #             result = {
    #                 'step': steps[end_idx],
    #                 'trajectory_length': window_size,
    #                 'trajectory': trajectory,
    #                 'action': steps[end_idx].get('action', ''),
    #                 'similarity': sim,
    #                 'reward': steps[end_idx].get('reward', 0),
    #                 'discounted_reward': discounted_return,
    #                 'episode_final_score': episode.get('final_score', 0),
    #                 'window_start_idx': window_start,
    #                 'window_end_idx': end_idx
    #             }
    #             scored_trajectories.append((sim, discounted_return, result))
        
    #     scored_trajectories.sort(key=lambda x: (x[0], x[1]), reverse=True)
    #     filtered_trajectories = [traj for traj in scored_trajectories if traj[0] > r]
        
    #     if scored_trajectories:
    #         print("==========================")
    #         print(query_trajectory)
    #         print("**************************")
    #         print(scored_trajectories[0][2]['trajectory'])
    #         print(scored_trajectories[0][2]['action'])
    #         print(scored_trajectories[0][2]['similarity'])
    #         print(scored_trajectories[0][2]['reward'])
    #         print(scored_trajectories[0][2]['discounted_reward'])
        
    #     return filtered_trajectories[:k]
    
    def retrieve_similar(self, game_history, current_state: str, current_summary: str = None, k: int = 3, r=0.9, use_vector: bool = True, info=None) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar trajectory segments from all episodes.
        Can use either LLM-generated summaries or vector embeddings for similarity matching.

        Args:
            game_history: Current game history
            current_state: The current state to append to the trajectory
            current_summary: Current trajectory summary (required if use_vector=False)
            k: Number of similar trajectories to retrieve
            r: Similarity threshold
            use_vector: If True, use vector-based similarity; if False, use summary-based similarity
            info: Game environment info dictionary

        Returns:
            List of similar trajectories with similarity scores
        """
        print(f"retrieve similar (method: {'vector' if use_vector else 'summary'})")

        if use_vector:
            # Use vector-based retrieval
            return self.retrieve_similar_with_vector(
                game_history=game_history,
                current_state=current_state,
                current_summary=current_summary,
                k=k,
                r=r,
                info=info
            )
        else:
            # Use summary-based retrieval (original behavior)
            if current_summary is None:
                print("Warning: current_summary is required for summary-based retrieval")
                return []
            
            return self.retrieve_similar_with_summary(
                game_history=game_history,
                current_state=current_state,
                current_summary=current_summary,
                k=k,
                r=r,
                use_summary=True  # Use LLM summaries
            )

    # -------------------- Prompt management --------------------
    def generate_prompt_update(self, game_history: List[Dict[str, Any]], final_score: float, success: bool, current_prompt: str, use_history: bool = True) -> Dict[str, Any]:
        """
        Generate a prompt update recommendation based on the episode trajectory.

        Args:
            game_history: Complete game history from the episode
            final_score: Final score achieved
            success: Whether the episode was successful
            current_prompt: The current guiding prompt being used
            use_history: If True, use the new history-based prompt generation (default: True)

        Returns:
            dict: Contains 'recommended_prompt', 'reasoning', and 'key_insights'
        """
        print(f"\n{'='*60}")
        print("GENERATING PROMPT UPDATE RECOMMENDATION")
        if use_history:
            print("Using HISTORY-BASED prompt generation (comparing with top episodes)")
        else:
            print("Using SINGLE-EPISODE prompt generation (legacy mode)")
        print(f"{'='*60}")

        # Use the new history-based prompt generation if enabled
        if use_history:
            recommendation = generate_prompt_with_history(
                game_history=game_history,
                final_score=final_score,
                success=success,
                current_prompt=current_prompt,
                episodes_jsonl_path=self.episodes_path,
                llm_model=self.llm_model,
                temperature=0.8,
                max_tokens=3000,
                top_k_episodes=3
            )
        else:
            # Legacy single-episode mode
            recommendation = generate_prompt_recommendation(
                game_history=game_history,
                final_score=final_score,
                success=success,
                current_prompt=current_prompt,
                llm_model=self.llm_model,
                temperature=0.8,
                max_tokens=2000
            )

        # Save to prompt history
        prompt_entry = {
            'episode_number': self.current_episode_number,
            'timestamp': os.path.basename(self.base_dir),
            'final_score': final_score,
            'success': success,
            'previous_prompt': current_prompt,
            'recommended_prompt': recommendation.get('recommended_prompt', current_prompt),
            'reasoning': recommendation.get('reasoning', ''),
            'key_insights': recommendation.get('key_insights', [])
        }
        self._append_jsonl(self.prompt_history_path, prompt_entry)

        print(f"\n{'='*60}")
        print("PROMPT UPDATE RECOMMENDATION")
        print(f"{'='*60}")
        print(f"Previous Prompt: {current_prompt}")
        print(f"\nRecommended Prompt: {recommendation.get('recommended_prompt', current_prompt)}")
        print(f"\nReasoning: {recommendation.get('reasoning', '')}")
        print(f"\nKey Insights:")
        for insight in recommendation.get('key_insights', []):
            print(f"  - {insight}")
        print(f"{'='*60}\n")

        return recommendation

    def get_prompt_history(self) -> List[Dict[str, Any]]:
        """
        Load the complete prompt history.

        Returns:
            List of prompt history entries
        """
        return self._load_jsonl(self.prompt_history_path)

    def save_guiding_prompt(self, guiding_prompt: str, episode_number: int = None):
        """
        Save the guiding prompt for the current episode to prompt_history.jsonl.
        This adds a 'current_guiding_prompt' field to track what prompt was used for each episode.

        Args:
            guiding_prompt: The guiding prompt to save
            episode_number: Episode number (optional, uses current_episode_number if not provided)
        """
        if episode_number is None:
            episode_number = self.current_episode_number

        # Load existing history to check if we need to add a new entry or update
        prompt_history = self._load_jsonl(self.prompt_history_path)

        # Check if there's already an entry for this episode
        updated = False
        for entry in prompt_history:
            if entry.get('episode_number') == episode_number:
                # Update existing entry
                entry['current_guiding_prompt'] = guiding_prompt
                updated = True
                break

        if not updated:
            # Create new entry for this episode (will be updated by generate_prompt_update later)
            prompt_entry = {
                'episode_number': episode_number,
                'timestamp': os.path.basename(self.base_dir),
                'current_guiding_prompt': guiding_prompt
            }
            self._append_jsonl(self.prompt_history_path, prompt_entry)
            print(f"Saved guiding prompt for episode {episode_number}")
        else:
            # Rewrite the file with updated entries
            with open(self.prompt_history_path, 'w', encoding='utf-8') as f:
                for entry in prompt_history:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Updated guiding prompt for episode {episode_number}")

    def load_latest_guiding_prompt(self) -> Optional[str]:
        """
        Load the most recent guiding prompt from prompt_history.jsonl.
        Tries to load 'recommended_prompt' first (from generate_prompt_update),
        falls back to 'current_guiding_prompt' if not available.

        Returns:
            str: The most recent guiding prompt, or None if no history exists
        """
        prompts_history = self._load_jsonl(self.prompt_history_path)
        if prompts_history:
            latest_entry = prompts_history[-1]
            # Try recommended_prompt first (updated prompt from LLM)
            latest_prompt = latest_entry.get('recommended_prompt')
            if not latest_prompt:
                # Fall back to current_guiding_prompt (prompt used in episode)
                latest_prompt = latest_entry.get('current_guiding_prompt')

            if latest_prompt:
                print(f"\n{'='*60}")
                print("LOADED GUIDING PROMPT FROM HISTORY")
                print(f"{'='*60}")
                print(f"Episode: {latest_entry.get('episode_number', 'Unknown')}")
                print(f"Prompt: {latest_prompt}")
                print(f"{'='*60}\n")
                return latest_prompt
        return None
