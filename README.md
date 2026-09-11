# Agent State Architecture: From Naive ReAct to Explicit State

For daily progress of this repo, check out the articles that i post at X (https://x.com/vihaaan17)

An educational repository designed to teach the fundamentals of **Agent State Management** from scratch, moving from an implicit string-gluing loop to a production-grade four-layered explicit state architecture.

---

## The Core Concept: What is "Agent State"?

When running an agent, "state" is often treated loosely as a single prompt string. In reality, state breaks down into **four distinct categories**, each with its own lifecycle, persistence requirements, and failure modes:

| State Layer | Definition & Scope | Where it Physically Lives | Lifecycle |
| :--- | :--- | :--- | :--- |
| **1. Working State** | Temporary values that only exist within a single turn (e.g. parsed tool arguments, regex matches, step tokens). | Local in-memory dict | Reset to `{}` at the start of **every turn**. Never persisted. |
| **2. Episode State** | Everything accumulated across one full run/trajectory (conversation turns, tool execution history, partial plans). | In-memory session store / Redis | Keyed by `session_id`. Survives turns; intentionally discarded or archived when the episode ends. |
| **3. Persistent State** | Durable data that survives across multiple user sessions (e.g. user preferences, total tasks completed, account flags). | Durable database (SQLite / Postgres) | Loaded **once** at session start; written back explicitly when a persistent field changes. |
| **4. Environment Beliefs** | The agent's cached, local view of the outside world (e.g. file contents, DB rows, external API responses). | Local cache with timestamps | **Not** the real source of truth! Contains `cached_at` timestamps to detect and prevent stale belief drift. |

---

## Repository Structure

```
agent-state/
├── React_loop.py               # Part 1: The Naive ReAct Loop (Implicit String State)
├── explicit_state_agent.py     # Part 2: The Explicit 4-Layer Architecture (Diffable & Replayable)
├── report.txt                  # External environment file (simulated operational database)
├── agent_persistent.db         # Durable SQLite database (Persistent State)
├── state_logs/                 # Turn-by-turn serialized JSON snapshots
├── .env.example                # Environment template for GEMINI_API_KEY
└── README.md                   # Architecture and learning documentation
```

---

## Part 1: The Problem with Implicit State (`React_loop.py`)

In the naive implementation, everything is squashed into a single growing string:

```python
history = ""
for turn in range(max_turns):
    prompt = history + "\nWhat's next?"
    response = call_llm(prompt)
    tool_result = maybe_call_tool(response)
    history += f"\n{response}\n{tool_result}"
```

### Why this breaks down:
1. **Un-debuggable**: You cannot answer *"What did the agent believe at turn 2?"* without writing custom regex to re-parse the raw string.
2. **Ephemeral Leakage**: Working state (e.g. intermediate reasoning or temporary tool parsing errors) gets permanently glued into the history.
3. **No True Persistence**: User preferences or long-term task records are lost when the script ends.
4. **Stale Environment**: External file contents get frozen in the prompt string with zero indication of when they were fetched or if they are stale.

---

## Part 2: The Explicit Four-Layer Architecture (`explicit_state_agent.py`)

In the explicit architecture, we replace the string with a typed, centralized state object:

```python
state = {
    "working": {},             # Reset every turn
    "episode": {
        "session_id": "trajectory_01",
        "question": "...",
        "messages": [],        # Structured list of turns
        "tool_outputs": [],    # Tool logs with timestamps
        "plan": None,
    },
    "persistent": {},          # Loaded from SQLite once at start
    "environment_beliefs": {}, # Timestamped snapshots of external facts
}
```

### Execution Lifecycle:
```
1. Reset Working State: state["working"] = {"turn": turn, ...}
2. Log Pre-Decision Snapshot: Serializes state to state_logs/
3. Render Context Window: Formats prompt strictly from explicit state
4. Decide Action: Calls LLM -> populates state["working"]["raw_llm_response"]
5. Execute Tool: Runs tool -> records timestamped observation into state["environment_beliefs"]
6. Update State: Appends to episode["messages"], saves to EpisodeStore and SQLite
7. Compute State Diff: Displays exact fields changed during this turn
```

---

## Quick Start

### 1. Configure Environment
Create `.env` and set your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Run the Naive Implicit Loop
```bash
python React_loop.py
```
*Observe how everything is glued into one giant string at the end.*

### 3. Run the Explicit State Agent
```bash
python explicit_state_agent.py
```
*Observe real-time state diffs, clean working state resets, and inspect the JSON snapshots in `state_logs/`.*
