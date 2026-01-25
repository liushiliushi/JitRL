# JitRL: Cross-Episode Memory for LLM Agents

JitRL is a research framework for building LLM-based agents with **cross-episode memory** capabilities. It enables agents to learn from past experiences and improve their performance over multiple episodes.

## Overview

This project contains two main components:

| Component | Domain | Description |
|-----------|--------|-------------|
| **Jericho** | Text Adventure Games | LLM agents for interactive fiction games (Zork, etc.) |
| **WebArena** | Web Automation | LLM agents for browser-based tasks using BrowserGym |

Both components share a common architecture featuring:
- **Cross-Episode Memory**: Store and retrieve relevant experiences from past episodes
- **LLM-based Decision Making**: Use large language models for action generation
- **Multiple Agent Types**: Support for various agent architectures
- **Automatic Evaluation**: Built-in evaluation systems for measuring performance

## Project Structure

```
JitRL/
├── Jericho/                    # Text adventure game agents
│   ├── main.py                 # Entry point
│   ├── src/
│   │   ├── memory_agent.py     # Memory-augmented agent
│   │   ├── our_agent.py        # UCB tree search agent
│   │   ├── naive_agent.py      # Simple baseline agent
│   │   ├── awm_agent.py        # Agent Workflow Memory agent
│   │   ├── cross_episode_memory.py  # Cross-episode memory implementation
│   │   ├── evaluation.py       # Game evaluation
│   │   ├── env.py              # Jericho environment wrapper
│   │   └── utils.py            # LLM utilities
│   └── jericho-games/          # Game ROM files (not included)
│
├── WebArena/                   # Web automation agents
│   ├── run.py                  # Main entry point
│   ├── memory_agents/
│   │   ├── memory_agent.py     # BrowserGym memory agent
│   │   ├── dynamic_prompting.py # Dynamic prompt generation
│   │   └── utils/              # Utilities (LLM helpers, memory, etc.)
│   ├── autoeval/               # Automatic evaluation system
│   │   ├── evaluator.py        # LLM-based evaluator
│   │   ├── enhanced_evaluator.py # Rule-based evaluator
│   │   └── live_evaluator.py   # Real-time evaluation
│   ├── config_files/           # Task configurations (812 tasks)
│   ├── config_files_lite/      # WebArena-Lite (165 tasks)
│   └── workflow_utils.py       # Shared workflow utilities
│
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- OpenAI API key (or OpenRouter API key for other models)

### Jericho Setup

```bash
cd Jericho

# Install dependencies
pip install jericho openai tiktoken numpy python-dotenv

# Download game ROMs (not included in repo)
# Place ROM files in jericho-games/ directory
```

### WebArena Setup

```bash
cd WebArena

# Install dependencies
pip install -r requirements.txt

# Install BrowserGym
pip install browsergym-core browsergym-experiments

# Set up environment variables
cp env_setup.txt.example env_setup.txt
# Edit env_setup.txt with your API keys and WebArena URLs
```

## Usage

### Jericho (Text Adventure Games)

```bash
cd Jericho

# Run with memory agent (default)
python main.py --game_name zork1 --agent_type memory --eval_runs 10

# Run with UCB tree search agent
python main.py --game_name library --agent_type our --eval_runs 50

# Available agent types: memory, our, naive, awm
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--game_name` | `library` | Game to play (zork1, zork3, library, etc.) |
| `--agent_type` | `memory` | Agent type: memory, our, naive, awm |
| `--eval_runs` | `50` | Number of episodes to run |
| `--llm_model` | `gemini-2.5-flash` | LLM model to use |
| `--env_step_limit` | `50` | Max steps per episode |
| `--enable_cross_mem` | `True` | Enable cross-episode memory |

### WebArena (Web Automation)

```bash
cd WebArena

# Run a single task
python run.py --task webarena.0 --agent_type memory

# Run multiple tasks
python run.py --task webarena.0 --test_case_start 0 --test_case_end 10

# Run with repeated episodes (for learning)
python run.py --task webarena.0 --repeat_runs 5
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | Required | Task name (e.g., webarena.0) |
| `--agent_type` | `memory` | Agent type: memory, evotest, reference, reflexion |
| `--repeat_runs` | `1` | Number of repeated runs per task |
| `--config_dir` | `config_files` | Config directory (config_files or config_files_lite) |
| `--enable_cross_mem` | `True` | Enable cross-episode memory |
| `--headless` | `True` | Run browser in headless mode |

## Agent Types

### Jericho Agents

| Agent | Description |
|-------|-------------|
| `memory` | Memory-augmented agent with cross-episode learning |
| `our` | UCB tree search with evolutionary prompt optimization |
| `naive` | Simple LLM agent without memory |
| `awm` | Agent Workflow Memory (AWM) baseline |

### WebArena Agents

| Agent | Description |
|-------|-------------|
| `memory` | BrowserGym agent with cross-episode memory |
| `evotest` | Evolutionary testing agent |
| `reference` | Reference-guided agent |
| `reflexion` | Reflexion-based agent with self-reflection |

## Cross-Episode Memory

The core innovation of JitRL is the **Cross-Episode Memory** system:

1. **Experience Storage**: After each episode, successful action sequences and their outcomes are stored
2. **Similarity Matching**: When facing a new state, the agent retrieves similar past experiences
3. **Guided Decision Making**: Retrieved experiences inform the agent's action selection
4. **Continuous Learning**: Memory accumulates over episodes, improving performance over time

```python
# Example: Memory retrieval in action
similar_experiences = memory.retrieve(
    current_state=state,
    top_k=10,
    similarity_threshold=0.95
)
```

## Evaluation

### Jericho Evaluation

Games are evaluated by final score. Results are saved to `output/{game_name}/{agent_type}/`.

### WebArena Evaluation

WebArena supports multiple evaluation methods:

```bash
# Evaluate a completed trajectory
python -m autoeval.evaluate_trajectory --result_dir results/webarena.0

# Evaluation types:
# - string_match: Exact/fuzzy string matching
# - url_match: URL pattern matching
# - program_html: DOM content verification
```

## Environment Variables

Create a `.env` file or `env_setup.txt` with:

```bash
# For OpenRouter (recommended)
OPENAI_API_KEY=your_openrouter_api_key

# For direct OpenAI API (embeddings)
OPENAI_API_KEY2=your_openai_api_key

# WebArena URLs (if running locally)
WA_SHOPPING=http://localhost:7770
WA_SHOPPING_ADMIN=http://localhost:7780/admin
WA_REDDIT=http://localhost:9999
WA_GITLAB=http://localhost:8023
WA_MAP=http://localhost:3000
```

## Supported LLM Models

JitRL supports any model available through OpenRouter:

- **OpenAI**: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
- **Anthropic**: claude-3.5-sonnet, claude-3-opus
- **Google**: gemini-2.5-flash, gemini-2.5-pro
- **Others**: llama, mistral, etc.

## Citation

If you use JitRL in your research, please cite:

```bibtex
@misc{jitrl2024,
  title={JitRL: Cross-Episode Memory for LLM Agents},
  year={2024}
}
```

## License

This project is for research purposes.

## Acknowledgments

- [Jericho](https://github.com/microsoft/jericho) - Text adventure game framework
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) - Web automation framework
- [WebArena](https://webarena.dev/) - Web agent benchmark
