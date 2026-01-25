import os
import json
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from .utils import calculate_summary_similarity, evaluate_step_scores_with_llm, summarize_trajectory_context, normalize_action, normalize_url
from .openai_helpers import get_embedding_with_retries

try:
    import faiss
except ImportError:
    print("Warning: faiss not installed. Please install with: pip install faiss-cpu")
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Warning: rank-bm25 not installed. Please install with: pip install rank-bm25")
    BM25Okapi = None


class CrossEpisodeMemory:
    def __init__(self, base_dir: str, gamma: float = 0.95, llm_model: str = "gpt-4", llm_extract: str = None, vector_dim: int = 1536, embedding_model: str = "text-embedding-ada-002", task_similarity_threshold: float = 0.27):
        self.llm_extract = llm_extract or llm_model
        print(f"Initializing CrossEpisodeMemory in {base_dir} with gamma={gamma}, llm_model={llm_model}, llm_extract={self.llm_extract}, vector_dim={vector_dim}, embedding_model={embedding_model}, task_similarity_threshold={task_similarity_threshold}")
        self.base_dir = base_dir
        self.gamma = gamma
        self.llm_model = llm_model
        self.vector_dim = vector_dim
        self.embedding_model = embedding_model
        self.task_similarity_threshold = task_similarity_threshold
        os.makedirs(self.base_dir, exist_ok=True)
        self.episodes_path = os.path.join(self.base_dir, 'episodes.jsonl')
        self.episodes_abstract_path = os.path.join(self.base_dir, 'episode_abstract.jsonl')
        self.step_summaries = []  # Store step summaries during episode
        self.step_context_cache = []  # Cache for trajectory_context and current_env_info for each step

        # Initialize episode counter (will be set properly in add_episode)
        self.current_episode_number = 0

        # Initialize vector database
        self._init_vector_database()

    def _init_vector_database(self):
        """Initialize BM25 index (replacing Faiss vector database)."""
        self.history_index_path = os.path.join(self.base_dir, 'history_vectors.index')
        self.state_index_path = os.path.join(self.base_dir, 'state_vectors.index')
        self.task_index_path = os.path.join(self.base_dir, 'task_vectors.index')
        self.step_metadata_path = os.path.join(self.base_dir, 'step_metadata.pkl')
        self.bm25_index_path = os.path.join(self.base_dir, 'bm25_index.pkl')

        # Initialize BM25 index
        self.bm25_index = None
        self.bm25_corpus = []  # Store tokenized documents for BM25

        # Check if OpenAI embedding is available (kept for compatibility, but not used)
        self.encoder_available = False  # Disable embedding to save costs
        print(f"Using BM25 for retrieval (vector embedding disabled to save costs)")

        # Load metadata and BM25 index from disk if exists
        if os.path.exists(self.step_metadata_path):
            with open(self.step_metadata_path, 'rb') as f:
                self.step_metadata = pickle.load(f)

            # Rebuild BM25 index from metadata
            if BM25Okapi is not None and self.step_metadata:
                print(f"Rebuilding BM25 index from {len(self.step_metadata)} documents...")
                self.bm25_corpus = []
                for metadata in self.step_metadata:
                    trajectory_context = metadata.get('trajectory_context', '')
                    # Tokenize for BM25
                    tokens = trajectory_context.split()
                    self.bm25_corpus.append(tokens)

                self.bm25_index = BM25Okapi(self.bm25_corpus)
                print(f"BM25 index ready with {len(self.bm25_corpus)} documents")
        else:
            self.step_metadata = []

        # Keep Faiss indices as None (not used)
        self.history_index = None
        self.state_index = None
        self.task_index = None
    
    def _save_vector_database(self):
        """Save metadata to disk (BM25 index is rebuilt on load)."""
        # Only save metadata, BM25 index will be rebuilt from metadata on next load
        with open(self.step_metadata_path, 'wb') as f:
            pickle.dump(self.step_metadata, f)
        # Note: No need to save Faiss indices anymore
    
    def clear_vector_database(self, save_to_disk: bool = True):
        """
        Clear all vectors and metadata from the vector database.
        
        Args:
            save_to_disk: If True, also delete the saved files on disk
        """
        print("Clearing vector database...")

        history_count = 0
        state_count = 0
        task_count = 0

        if self.history_index is not None:
            history_count = self.history_index.ntotal
            self.history_index.reset()
            print(f"✓ History index cleared (was {history_count} vectors)")

        if self.state_index is not None:
            state_count = self.state_index.ntotal
            self.state_index.reset()
            print(f"✓ State index cleared (was {state_count} vectors)")

        if self.task_index is not None:
            task_count = self.task_index.ntotal
            self.task_index.reset()
            print(f"✓ Task index cleared (was {task_count} vectors)")

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

            if os.path.exists(self.task_index_path):
                os.remove(self.task_index_path)
                print(f"✓ Deleted {self.task_index_path}")

            if os.path.exists(self.step_metadata_path):
                os.remove(self.step_metadata_path)
                print(f"✓ Deleted {self.step_metadata_path}")
            
            print("✓ Vector database files deleted from disk")
        else:
            # Save empty database
            self._save_vector_database()
            print("✓ Empty database saved to disk")
        
        print("Vector database cleared successfully!")
    
    def get_vector_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary containing database statistics
        """
        stats = {
            'total_history_vectors': 0,
            'total_state_vectors': 0,
            'total_metadata_entries': len(self.step_metadata),
            'vector_dimension': self.vector_dim,
            'database_available': self.history_index is not None and self.state_index is not None and self.encoder_available,
            'history_index_file_exists': os.path.exists(self.history_index_path),
            'state_index_file_exists': os.path.exists(self.state_index_path),
            'metadata_file_exists': os.path.exists(self.step_metadata_path)
        }
        
        if self.history_index is not None:
            stats['total_history_vectors'] = self.history_index.ntotal
        
        if self.state_index is not None:
            stats['total_state_vectors'] = self.state_index.ntotal
        
        if stats['history_index_file_exists']:
            stats['history_index_file_size'] = os.path.getsize(self.history_index_path)
        
        if stats['state_index_file_exists']:
            stats['state_index_file_size'] = os.path.getsize(self.state_index_path)
        
        if stats['metadata_file_exists']:
            stats['metadata_file_size'] = os.path.getsize(self.step_metadata_path)
        
        return stats
    
    def _encode_trajectory_context(self, states: List[str], actions: List[str], current_step: int, current_summary: str = None, task_goal: str = None, urls: List[str] = None, screenshots_dir: str = None):
        """Encode trajectory context from step 0 to current_step as vectors.

        Generates three vectors: history vector (from trajectory context), state vector (from current state),
        and task vector (from task goal). Also normalizes the URL.

        Args:
            states: List of all states in the episode
            actions: List of all actions in the episode
            current_step: Current step index
            current_summary: Summary of current step (optional, for compatibility)
            task_goal: Task goal/instruction (e.g., "Search for laptop and add to cart")
            urls: List of URLs for each step (optional)
            screenshots_dir: Directory containing screenshots (optional, for LLM-based trajectory extraction)

        Returns:
            Tuple of (trajectory_context, current_env_info, normalized_url) for BM25
            Note: No vectors needed anymore
        """
        # BM25 doesn't need embeddings, so we skip the encoder check
        # if not self.encoder_available:
        #     return None, None, None

        # Validate current_step
        if current_step < 0 or current_step >= len(states):
            print(f"Invalid current_step: {current_step}")
            return None, None, None

        # Get current state as current_env_info (raw state, for metadata storage only)
        current_env_info = states[current_step] if states[current_step] else ""

        # Get current URL (normalized)
        from .utils import normalize_url
        current_url = urls[current_step] if urls and current_step < len(urls) else None
        current_normalized_url = normalize_url(current_url) if current_url else None

        # Build trajectory context from normalized historical actions
        # ONLY include actions on the SAME normalized URL
        if current_step > 0 and current_normalized_url:
            from .utils import normalize_action, extract_effective_trajectory_context
            normalized_actions = []

            # Find the first step with the same normalized URL
            for i in range(current_step):
                step_url = urls[i] if urls and i < len(urls) else None
                step_normalized_url = normalize_url(step_url) if step_url else None

                # Only include actions on the same normalized URL
                if step_normalized_url == current_normalized_url:
                    action = actions[i]
                    state = states[i] if i < len(states) else ""
                    try:
                        normalized_data = normalize_action(action, state)
                        normalized_action = normalized_data.get('normalized_action', action)
                        normalized_actions.append(normalized_action)
                    except Exception as e:
                        print(f"Warning: Failed to normalize action at step {i}: {e}")
                        normalized_actions.append(action)

            # Use LLM-based extraction if screenshots are available
            # Note: We pre-save screenshot_step_N.png in generate_action(), so we can use current_step
            trajectory_context = extract_effective_trajectory_context(
                    normalized_actions=normalized_actions,
                    screenshots_dir=screenshots_dir,
                    current_step=current_step,  # Use current step's screenshot
                    llm_model=self.llm_extract,  # Use the model specified by --llm_extract parameter
                    temperature=0.3,
                    max_tokens=1000
                )

        else:
            trajectory_context = "No action"

        # BM25 doesn't need embeddings - just return the text
        print(current_url)
        print(current_normalized_url)
        print(trajectory_context)
        return trajectory_context, current_env_info, current_normalized_url
    
    def _store_step_in_vector_db(self, states: List[str], actions: List[str], step_index: int, episode_data: Dict[str, Any]):
        """Store a step in BM25 index (no vector embedding needed).

        Args:
            states: List of all states in the episode
            actions: List of all actions in the episode
            step_index: Index of the current step
            episode_data: Data for the entire episode
        """
        # Use cached trajectory context and normalized_url (no vectors needed)
        if step_index >= len(self.step_context_cache):
            return

        # For BM25, we only need trajectory_context and normalized_url (no vectors)
        cache_entry = self.step_context_cache[step_index]
        if len(cache_entry) == 6:
            # Old format with vectors
            trajectory_context, current_env_info, _, _, _, normalized_url = cache_entry
        else:
            # New format without vectors
            trajectory_context, current_env_info, normalized_url = cache_entry

        # Get step data from episode
        steps = episode_data.get('steps', [])
        if step_index >= len(steps):
            return
        step_data = steps[step_index]

        # Extract future rewards from current step to end of episode
        future_rewards = []
        for u in range(step_index, len(steps)):
            future_rewards.append(steps[u].get('llm_step_score', 0))

        # Prepare step metadata
        step_url = step_data.get('url')

        step_metadata = {
            'episode_timestamp': episode_data.get('timestamp'),
            'episode_number': self.current_episode_number,
            'task_goal': episode_data.get('task_goal', ''),
            'step_data': step_data.copy(),
            'episode_final_score': episode_data.get('final_score'),
            'episode_success': episode_data.get('success'),
            'trajectory_context': trajectory_context,
            'current_env_info': current_env_info,
            'future_rewards': future_rewards,
            'step_summary': step_data.get('step_summary', ''),
            'url': step_url,
            'normalized_url': normalized_url
        }

        # Store metadata
        self.step_metadata.append(step_metadata)

        # Add to BM25 corpus
        tokens = trajectory_context.split()
        self.bm25_corpus.append(tokens)

        # Rebuild BM25 index (fast operation)
        if BM25Okapi is not None:
            self.bm25_index = BM25Okapi(self.bm25_corpus)

        print(f"Stored step {step_index} in BM25 index (total docs: {len(self.step_metadata)})")

    def search_similar_steps(self, query_states: List[str], query_actions: List[str], k: int = 5, current_summary: str = None, history_weight: float = 0.3, state_weight: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar steps in the vector database using both history and state vectors.
        
        Args:
            query_states: List of states in the query trajectory
            query_actions: List of actions in the query trajectory
            k: Number of similar steps to retrieve
            current_summary: Summary of current step
            history_weight: Weight for history vector similarity (default: 0.3)
            state_weight: Weight for state vector similarity (default: 0.7)
            
        Returns:
            List of similar steps with similarity scores
        """
        
        # Encode the query trajectory
        trajectory_context, current_env_info, history_vector, state_vector = self._encode_trajectory_context(query_states, query_actions, len(query_states) - 1, current_summary)
        
        # Store both vectors in cache
        self.step_context_cache.append((trajectory_context, current_env_info, history_vector, state_vector))
        
        # Check if vectors are valid
        if history_vector is None or state_vector is None:
            return []
        
        if self.history_index is None or self.state_index is None or not self.encoder_available:
            return []
        
        if self.history_index.ntotal == 0 or self.state_index.ntotal == 0:
            return []
        
        # Search in both indices
        history_scores, history_indices = self.history_index.search(history_vector.reshape(1, -1), k)
        state_scores, state_indices = self.state_index.search(state_vector.reshape(1, -1), k)
        
        # Combine results with weighted similarity
        combined_results = {}
        
        # Process history results
        for score, idx in zip(history_scores[0], history_indices[0]):
            if idx < len(self.step_metadata):
                if idx not in combined_results:
                    combined_results[idx] = {'history_score': 0.0, 'state_score': 0.0}
                combined_results[idx]['history_score'] = float(score)
        
        # Process state results
        for score, idx in zip(state_scores[0], state_indices[0]):
            if idx < len(self.step_metadata):
                if idx not in combined_results:
                    combined_results[idx] = {'history_score': 0.0, 'state_score': 0.0}
                combined_results[idx]['state_score'] = float(score)
        
        # Calculate weighted similarity and prepare results
        results = []
        for idx, scores in combined_results.items():
            weighted_similarity = history_weight * scores['history_score'] + state_weight * scores['state_score']
            result = self.step_metadata[idx].copy()
            result['history_similarity'] = scores['history_score']
            result['state_similarity'] = scores['state_score']
            result['similarity_score'] = weighted_similarity
            results.append((weighted_similarity, result))
        
        # Sort by weighted similarity
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k results
        final_results = []
        for i, (_, result) in enumerate(results[:k]):
            result['rank'] = i + 1
            final_results.append(result)
        
        return final_results

    def retrieve_similar_with_vector(self, game_history: List[Dict[str, Any]], current_state: str, current_summary: str = None, task_goal: str = None, current_url: str = None, k: int = 3, r: float = 0.7, task_weight: float = 0.3, history_weight: float = 0.5, state_weight: float = 0.5, screenshots_dir: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar trajectory segments using task filtering + TWO-FACTOR similarity:
        - Task filtering: Jaccard similarity (n-gram=1, threshold=0.27) to filter same-task trajectories
        - History vector: Trajectory context similarity
        - State vector: Current state similarity

        Args:
            game_history: Current game history
            current_state: The current state to append to the trajectory
            current_summary: Summary of current trajectory
            task_goal: Task goal/instruction
            k: Number of similar trajectories to retrieve
            r: Similarity threshold (cosine similarity score)
            task_weight: Weight for task similarity (not used, kept for compatibility)
            history_weight: Weight for history similarity (default: 0.5)
            state_weight: Weight for state similarity (default: 0.5)
            screenshots_dir: Directory containing screenshots (optional, for LLM-based trajectory extraction)

        Returns:
            List of similar trajectories with similarity scores
        """
        # if self.faiss_index is None or not self.encoder_available or self.faiss_index.ntotal == 0:
        #     print("Vector database not available or empty")
        #     return []

        # Extract states, actions, and URLs from current game history
        query_states = [entry.get('state', '') for entry in game_history]
        query_actions = [entry.get('action', '') for entry in game_history]
        query_urls = [entry.get('url') for entry in game_history]

        # Use provided current_url or fallback to game_history
        if current_url is None:
            current_url = game_history[-1].get('url', 'N/A') if game_history else 'N/A'

        # Add current state and URL
        query_states.append(current_state)
        query_urls.append(current_url)

        # Normalize current action for similarity comparison
        current_action = query_actions[-1] if query_actions else None
        current_action_normalized = None
        if current_action and current_state:
            try:
                current_action_normalized = normalize_action(current_action, current_state)
            except Exception as e:
                print(f"Warning: Failed to normalize current action '{current_action}': {e}")

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

        # Encode current trajectory (no vectors needed for BM25)
        trajectory_context, current_env_info, normalized_url = self._encode_trajectory_context(
            query_states, query_actions, len(query_states) - 1, current_summary, task_goal=task_goal, urls=query_urls, screenshots_dir=screenshots_dir
        )

        if trajectory_context is None:
            print("Failed to encode trajectory context")
            return []

        self.step_summaries.append(trajectory_context)

        # Cache without vectors
        self.step_context_cache.append((trajectory_context, current_env_info, normalized_url))
            
        current_task_tokens = self._tokenize(task_goal) if task_goal else set()

        # Direct filtering approach (no BM25 recall stage)
        if len(self.step_metadata) == 0:
            print("Database is empty")
            return []

        # Step 1: Filter by normalized URL (most restrictive)
        # print(f"\n[STEP 1] Filtering by URL: {normalized_url}")
        url_filtered_indices = []
        for idx, metadata in enumerate(self.step_metadata):
            if metadata.get('normalized_url') == normalized_url:
                url_filtered_indices.append(idx)

        # print(f"  Result: {len(self.step_metadata)} total → {len(url_filtered_indices)} with matching URL")

        if not url_filtered_indices:
            # print("  No matching URL found in database")
            return []

        # Step 2: Filter by task similarity
        # print(f"\n[STEP 2] Filtering by task similarity (threshold > 0.27)")
        task_filtered_indices = []
        for idx in url_filtered_indices:
            metadata = self.step_metadata[idx]
            stored_task_goal = metadata.get('task_goal', '')
            stored_task_tokens = self._tokenize(stored_task_goal)
            task_sim = self._jaccard(current_task_tokens, stored_task_tokens, ngram=1)

            if task_sim > self.task_similarity_threshold:
                task_filtered_indices.append(idx)

        # print(f"  Result: {len(url_filtered_indices)} with URL → {len(task_filtered_indices)} with task_sim > 0.27")

        if not task_filtered_indices:
            # print("  No matching tasks found")
            return []

        # Step 3: Calculate similarity for all filtered candidates
        # print(f"\n[STEP 3] Calculating trajectory similarity for {len(task_filtered_indices)} candidates")

        # Process results and filter by threshold
        filtered_trajectories = []
        all_trajectories = []

        for idx in task_filtered_indices:
            metadata = self.step_metadata[idx]

            # Re-calculate task similarity (already done above, but needed for result)
            stored_task_goal = metadata.get('task_goal', '')
            stored_task_tokens = self._tokenize(stored_task_goal)
            task_sim = self._jaccard(current_task_tokens, stored_task_tokens, ngram=1)

            # Calculate history similarity using Jaccard similarity on trajectory context
            stored_trajectory_context = metadata.get('trajectory_context', '')
            # Tokenize trajectory contexts
            current_tokens = trajectory_context.split()
            stored_tokens = stored_trajectory_context.split()
            history_sim = self._jaccard(current_tokens, stored_tokens, ngram=1)

            # Get step data from metadata (needed for action similarity)
            step_data = metadata.get('step_data', {})

            # Calculate action similarity (bid-independent)
            if current_action_normalized and step_data:
                stored_action = step_data.get('action', '')
                stored_normalized_action = step_data.get('action_metadata')

                # If stored action doesn't have normalized version, normalize it now
                if not stored_normalized_action and stored_action:
                    stored_state = step_data.get('state', '')
                    try:
                        stored_normalized_action = normalize_action(stored_action, stored_state)
                    except:
                        stored_normalized_action = None

            # Since we already filtered by URL in Step 1, state similarity is 1
            state_sim = 1.0

            similarity = 0.7 * history_sim + 0.3 * task_sim

            # Calculate discounted return using stored future rewards and current gamma
            future_rewards = metadata.get('future_rewards', [])
            discounted_return = 0.0
            for u, step_reward in enumerate(future_rewards):
                llm_step_score = step_data.get('llm_step_score', 0)
                # Ensure llm_step_score is not None
                if llm_step_score is None:
                    llm_step_score = 0
                discount = self.gamma ** u
                # print(f"Step {u}: reward={step_reward}, llm_step_score={llm_step_score}, discount={discount:.4f}")
                discounted_return += discount * (step_reward if (step_reward != 0 and step_reward != None) else llm_step_score)

            result = {
                'step': step_data,
                'trajectory_context': metadata.get('trajectory_context', ''),
                'current_env_info': metadata.get('current_env_info', ''),
                'step_summary': metadata.get('step_summary', ''),
                'action': step_data.get('action', ''),
                'normalized_action': step_data.get('normalized_action', ''),
                'task_similarity': task_sim,
                'history_similarity': history_sim,
                'state_similarity': state_sim,
                'similarity': similarity,
                'reward': step_data.get('reward', 0),
                'llm_step_score': step_data.get('llm_step_score', 0),
                'discounted_reward': discounted_return,
                'episode_final_score': metadata.get('episode_final_score', 0),
                'episode_number': metadata.get('episode_number', 0),
                'step_index': step_data.get('step_num', 0),
                'task_goal': metadata.get('task_goal', ''),
                'url': metadata.get('url'),
                'normalized_url': metadata.get('normalized_url')
            }

            # Store all trajectories for potential fallback
            all_trajectories.append((similarity, discounted_return, result))

            # Filter by dynamic threshold
            if similarity >= dynamic_threshold:
                filtered_trajectories.append((similarity, discounted_return, result))

        # Sort filtered trajectories by similarity first, then by discounted reward
        filtered_trajectories.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Limit to top k results
        # filtered_trajectories = filtered_trajectories[:k]

        # Print filtering summary
        # print(f"\n[STEP 4] Filtering by similarity threshold")
        # print(f"  Candidates before threshold: {len(all_trajectories)}")
        # print(f"  Dynamic threshold: {dynamic_threshold:.3f}")
        # print(f"  Candidates after threshold: {len(filtered_trajectories)}")
        # if filtered_trajectories:
        #     top_sim = filtered_trajectories[0][0]
        #     print(f"  Top similarity: {top_sim:.4f}")


        # # Filter database entries by current normalized URL and calculate similarities
        # matching_entries = []
        # for idx, metadata in enumerate(self.step_metadata):
        #     if metadata.get('normalized_url') == normalized_url:
        #         # Calculate similarity with query
        #         stored_trajectory_context = metadata.get('trajectory_context', '')
        #         stored_tokens = stored_trajectory_context.split()
        #         current_tokens = trajectory_context.split()
        #         similarity = self._jaccard(current_tokens, stored_tokens, ngram=1)

        #         matching_entries.append((idx, metadata, similarity))

        # print("\n" + "="*100)
        # print(f"DATABASE ENTRIES FOR CURRENT URL: {normalized_url}")
        # print(f"Total matching entries: {len(matching_entries)}")
        # print("="*100)

        # if matching_entries:
        #     # Sort by similarity for better readability
        #     matching_entries_sorted = sorted(matching_entries, key=lambda x: x[2], reverse=True)

        #     for idx, metadata, similarity in matching_entries_sorted:
        #         step_data = metadata.get('step_data', {})
        #         print(f"\n[Entry #{idx}] Similarity: {similarity:.4f}")
        #         print(f"  Stored Task: {metadata.get('task_goal', 'N/A')}")
        #         print(f"  Trajectory Context: {metadata.get('trajectory_context', 'N/A')}")
        #         print(f"  URL (Original): {metadata.get('url', 'N/A')}")
        #         print(f"  URL (Normalized): {metadata.get('normalized_url', 'N/A')}")
        #         print(f"  Historical Action: {step_data.get('normalized_action', step_data.get('action', 'N/A'))}")
        #         print(f"  Historical Reward: {step_data.get('llm_step_score', 'N/A')}")
        # else:
        #     print("\nNo entries found for this URL in database.")

        print("\n" + "="*100)
        print("CURRENT QUERY")
        print("="*100)
        print(f"Current Task: {task_goal}")
        print(f"Current Trajectory: {trajectory_context}")
        print(f"Current URL (Original): {current_url}")
        print(f"Current URL (Normalized): {normalized_url}")
        print(f"Similarity Threshold: {dynamic_threshold:.3f}")

        print("\n" + "="*100)
        print("RETRIEVAL RESULTS")
        print("="*100)

        if filtered_trajectories:
            print(f"Found {len(filtered_trajectories)} matches above threshold")
            print("-"*100)

            # Show all matches
            for i, (sim, reward, trajectory_data) in enumerate(filtered_trajectories):
                print(f"\n[Match {i+1}] Similarity: {trajectory_data['similarity']:.4f}")
                print(f"  Stored Task: {trajectory_data.get('task_goal')}")
                print(f"  Trajectory Context: {trajectory_data.get('trajectory_context')}")
                print(f"  URL (Original): {trajectory_data.get('url')}")
                print(f"  URL (Normalized): {trajectory_data.get('normalized_url')}")
                print(f"  History Similarity: {trajectory_data['history_similarity']:.4f}")
                print(f"  Historical Action: {trajectory_data.get('normalized_action', trajectory_data['action'])}")
                print(f"  Historical Reward: {trajectory_data['llm_step_score']}")
        else:
            print("No matches above threshold")

            if all_trajectories:
                # Show why top matches were filtered out
                all_trajectories.sort(key=lambda x: x[0], reverse=True)
                top_k = min(5, len(all_trajectories))
                print(f"\nTop {top_k} candidates (below threshold):")
                print("-"*100)

                for i in range(top_k):
                    sim, reward, trajectory_data = all_trajectories[i]
                    print(f"\n[Candidate {i+1}] Similarity: {trajectory_data['similarity']:.4f} < {dynamic_threshold:.3f}")
                    print(f"  Stored Task: {trajectory_data.get('task_goal', 'N/A')}")
                    print(f"  Trajectory Context: {trajectory_data.get('trajectory_context')}")
                    print(f"  URL (Original): {trajectory_data.get('url')}")
                    print(f"  URL (Normalized): {trajectory_data.get('normalized_url')}")
                    print(f"  History Similarity: {trajectory_data['history_similarity']:.4f}")
                    print(f"  Historical Action: {trajectory_data.get('normalized_action', trajectory_data['action'])}")
                    print(f"  Historical Reward: {trajectory_data['llm_step_score']}")
            else:
                print("\nNo entries in database at all.")

        print("="*100)
        
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
                   success: bool = False,
                   llm_analysis: str = None,
                   task_goal: str = None,
                   user_instruction: str = None,
                   screenshots_dir: str = None):
        """
        Store a complete episode with all steps.

        Args:
            game_history: List of game history entries containing state, action, score, reward, etc.
            state: Current state
            final_score: Final score of the episode
            success: Whether the episode was successful
            llm_analysis: LLM analysis of the episode result
            task_goal: Task goal/instruction for this episode
            user_instruction: Custom user instruction for scoring actions (optional)
        """
        # Set current episode number (1-based)
        existing_episodes = self.load_episodes()
        self.current_episode_number = self.current_episode_number + 1

        # Extract data from game_history
        states = [entry.get('state', '') for entry in game_history]
        actions = [entry.get('action', '') for entry in game_history]
        scores = [entry.get('score', 0) for entry in game_history]
        rewards = [entry.get('reward', 0) for entry in game_history]
        urls = [entry.get('url') or None for entry in game_history]

        # Calculate delta scores
        delta_scores = []
        prev_score = 0
        for entry in game_history:
            current_score = entry.get('score', 0) or 0
            delta = current_score - prev_score
            delta_scores.append(delta)
            prev_score = current_score

        # Save all steps including the last one (e.g., send_msg_to_user action)
        num_steps = len(states)

        episode = {
            'timestamp': os.path.basename(self.base_dir),  # Use directory name as timestamp
            'num_steps': len(states),
            'final_score': final_score,
            'success': success,
            'llm_analysis': llm_analysis,
            'task_goal': task_goal if task_goal else '',
            'steps': [],
        }

        # Use LLM to score each step
        print(f"Evaluating step scores for episode with {len(game_history)} steps...")
        print(f"llm_model: {self.llm_extract}")  # Use llm_extract for step evaluation
        step_scores, step_reasonings = evaluate_step_scores_with_llm(
                game_history=game_history,
                state=state,
                final_score=final_score,
                success=success,
                llm_analysis=llm_analysis,
                llm_model=self.llm_extract,  # Changed from self.llm_model to self.llm_extract
                temperature=0.3,
                user_instruction=user_instruction,
                task_goal=task_goal,
                screenshots_dir=screenshots_dir
            )
        print(f"Generated step scores: {step_scores}")
        print(f"Generated step reasonings: {len(step_reasonings)} entries")

        # Build steps list
        # Build steps list
        for i in range(num_steps):
            action_str = actions[i] if i < len(actions) else None
            state_str = states[i] if i < len(states) else ''

            # Normalize action with semantic element description
            normalized_action_data = None
            if action_str:
                try:
                    normalized_action_data = normalize_action(action_str, state_str)
                except Exception as e:
                    print(f"Warning: Failed to normalize action '{action_str}': {e}")
                    normalized_action_data = {
                        'raw_action': action_str,
                        'normalized_action': action_str,
                        'semantic_action': action_str
                    }

            step = {
                'step_num': i,
                'state': state_str,
                'action': action_str,
                'normalized_action': normalized_action_data.get('normalized_action') if normalized_action_data else None,
                'semantic_action': normalized_action_data.get('semantic_action') if normalized_action_data else None,
                'action_metadata': normalized_action_data,  # Store full metadata for debugging
                'step_summary': self.step_summaries[i] if i < len(self.step_summaries) else '',
                'reward': rewards[i],
                'score': scores[i],
                'delta_score': delta_scores[i],
                'llm_step_score': step_scores[i] if i < len(step_scores) else 0,
                'llm_reasoning': step_reasonings[i] if i < len(step_reasonings) else 'No reasoning available',
                'url': urls[i] if i < len(urls) else None  # Always save URL (even if None)
            }
            episode['steps'].append(step)

        if not success and episode['steps']:
            episode['steps'][-1]['reward'] = -10

        # step_context_cache is already built during inference by retrieve_similar_with_vector
        # No need to rebuild - just use the cached data directly
        print(f"[INFO] Using existing step_context_cache with {len(self.step_context_cache)} entries")

        # Store all steps in vector database after episode is complete
        for i in range(num_steps):
            self._store_step_in_vector_db(states, actions, i, episode)

        self._append_jsonl(self.episodes_path, episode)
        
        # Save vector database to disk
        self._save_vector_database()
        self.step_context_cache = []
        self.step_summaries = []
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
        
        # Convert to sets
        set_a, set_b = set(a_ngrams), set(b_ngrams)
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
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
    #                         temperature=0.3,
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
            
    #         # Show top 5 matching results
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
    
    def retrieve_similar(self, game_history, current_state: str, current_summary: str = None, task_goal: str = None, current_url: str = None, k: int = 3, r=0.9, use_vector: bool = True, screenshots_dir: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar trajectory segments from all episodes.
        Can use either LLM-generated summaries or vector embeddings for similarity matching.

        Args:
            game_history: Current game history
            current_state: The current state to append to the trajectory
            current_summary: Current trajectory summary (required if use_vector=False)
            task_goal: Task goal/instruction
            current_url: Current page URL
            normalized_url: Normalized URL for similarity calculation
            k: Number of similar trajectories to retrieve
            r: Similarity threshold
            use_vector: If True, use vector-based similarity; if False, use summary-based similarity
            screenshots_dir: Directory containing screenshots for LLM-based trajectory extraction

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
                task_goal=task_goal,
                current_url=current_url,
                k=k,
                r=r,
                screenshots_dir=screenshots_dir
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

    def __getstate__(self):
        """Custom serialization for multiprocessing support."""
        # Create a copy of the object's state
        state = self.__dict__.copy()

        # Remove FAISS indices (not serializable)
        state['history_index'] = None
        state['state_index'] = None
        state['task_index'] = None

        # Note: We keep paths so we can reload indices in __setstate__
        return state

    def __setstate__(self, state):
        """Custom deserialization for multiprocessing support."""
        # Restore the object's state
        self.__dict__.update(state)

        # Reload FAISS indices from disk
        if faiss is not None and self.encoder_available:
            try:
                if os.path.exists(self.history_index_path):
                    self.history_index = faiss.read_index(self.history_index_path)
                else:
                    self.history_index = faiss.IndexFlatL2(self.vector_dim)

                if os.path.exists(self.state_index_path):
                    self.state_index = faiss.read_index(self.state_index_path)
                else:
                    self.state_index = faiss.IndexFlatL2(self.vector_dim)

                if os.path.exists(self.task_index_path):
                    self.task_index = faiss.read_index(self.task_index_path)
                else:
                    self.task_index = faiss.IndexFlatL2(self.vector_dim)

            except Exception as e:
                print(f"Warning: Failed to reload FAISS indices: {e}")
                self.history_index = faiss.IndexFlatL2(self.vector_dim)
                self.state_index = faiss.IndexFlatL2(self.vector_dim)
                self.task_index = faiss.IndexFlatL2(self.vector_dim)
        else:
            self.history_index = None
            self.state_index = None
            self.task_index = None
    