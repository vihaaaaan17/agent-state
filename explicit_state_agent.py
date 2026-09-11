"""
explicit_state_agent.py
================================================================================
Architecture: Explicit Four-Layer Agent State Architecture

Implements the four-layered agent state taxonomy:
1. Working State: Local ephemeral values for one turn (cleared every turn).
2. Episode State: Structured trajectory history (messages, tools, plan).
3. Persistent State: Cross-session durable data (SQLite / Postgres).
4. Environment Beliefs: Cached external observations with freshness timestamps.

Provides full state serialization, turn-by-turn diffing, and deterministic replay.
Interacts with React_loop.py by importing its tools and LLM caller.
================================================================================
"""

import json
import os
import re
import sqlite3
import time
import copy
from typing import Any, Dict, Optional

# Interact with the foundational baseline without modifying it
import React_loop

# Staleness threshold for environment beliefs (drift guard)
STALE_THRESHOLD_SECONDS = 7200

# ------------------------------------------------------------------------------
# 1. PHYSICAL STORAGE BACKENDS FOR EACH STATE CATEGORY
# ------------------------------------------------------------------------------

class PersistentDatabase:
    """
    Physical backend for PERSISTENT STATE.
    In production, this is Postgres. Here, we use SQLite (agent_persistent.db).
    Loaded ONCE at session start; written back explicitly when persistent fields change.
    """
    def __init__(self, db_path: str = "agent_persistent.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persistent_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at REAL
                )
            """)
            conn.commit()

    def load_all(self) -> Dict[str, Any]:
        """Loads all persistent records into a dictionary."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT key, value FROM persistent_state")
            rows = cursor.fetchall()
            return {k: json.loads(v) for k, v in rows}

    def save_key(self, key: str, value: Any):
        """Persists a single key-value pair to durable storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO persistent_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            conn.commit()


class EpisodeStore:
    """
    Physical backend for EPISODE STATE.
    In production, this is Redis. Here, we use an in-memory dictionary keyed by session_id.
    Survives across steps in the same trajectory, can be purged when episode ends.
    """
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def get_episode(self, session_id: str) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self._store.get(session_id))

    def save_episode(self, session_id: str, data: Dict[str, Any]):
        self._store[session_id] = copy.deepcopy(data)

    def discard_episode(self, session_id: str):
        self._store.pop(session_id, None)


# ------------------------------------------------------------------------------
# 2. THE EXPLICIT FOUR-LAYER STATE SCHEMA
# ------------------------------------------------------------------------------

def initialize_state(
    session_id: str,
    question: str,
    persistent_db: PersistentDatabase,
    episode_store: EpisodeStore,
    existing_beliefs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Initializes the explicit 4-category state object.
    Can inherit existing_beliefs from a prior run to demonstrate cache pick-up!
    """
    # 1. Load persistent state from DB (or initialize defaults if first run)
    saved_persistent = persistent_db.load_all()
    if "user_preferences" not in saved_persistent:
        default_prefs = {"currency": "USD", "verbosity": "concise", "auto_confirm": True}
        persistent_db.save_key("user_preferences", default_prefs)
        persistent_db.save_key("total_tasks_completed", 0)
        saved_persistent = persistent_db.load_all()

    # 2. Check for existing episode state (e.g. resuming trajectory) or start fresh
    existing_episode = episode_store.get_episode(session_id)
    if not existing_episode:
        episode_data = {
            "session_id": session_id,
            "question": question,
            "messages": [],         # Chronological list of structured turns
            "tool_outputs": [],     # Structured history of all tool executions
            "plan": None,           # High-level plan if agent forms one
            "step_count": 0,
        }
        episode_store.save_episode(session_id, episode_data)
    else:
        episode_data = existing_episode

    # 3. Assemble the full four-way state object
    state = {
        # Category 1: Working State (local, cleared every step)
        "working": {},

        # Category 2: Episode State (survives across steps in trajectory, lives in episode_store)
        "episode": episode_data,

        # Category 3: Persistent State (durable across sessions, lives in persistent_db)
        "persistent": saved_persistent,

        # Category 4: Environment Beliefs (agent's cached snapshot of outside world with timestamps)
        "environment_beliefs": copy.deepcopy(existing_beliefs) if existing_beliefs else {},
    }

    return state


# ------------------------------------------------------------------------------
# 3. STATE LOGGING, DIFFING, AND TRAJECTORY REPLAY
# ------------------------------------------------------------------------------

LOGS_DIR = "state_logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def log_state(step: int, state: Dict[str, Any]):
    """
    Serializes the complete state object as JSON to disk and logs a structured summary.
    This enables turn-by-turn diffing and exact trajectory replay from any point.
    """
    session_id = state["episode"]["session_id"]
    filename = os.path.join(LOGS_DIR, f"{session_id}_turn{step:02d}.json")

    # Save complete raw snapshot
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    # Print clean readable console inspection
    print(f"\n--- [STATE LOG: Turn {step}] ---", flush=True)
    print(f"  * Working State:             {list(state['working'].keys())}", flush=True)
    print(f"  * Episode Messages Count:    {len(state['episode']['messages'])}", flush=True)
    print(f"  * Cached Environment Beliefs: {list(state['environment_beliefs'].keys())}", flush=True)
    print(f"  * Persistent Tasks Done:     {state['persistent'].get('total_tasks_completed', 0)}", flush=True)
    print(f"  [Snapshot saved to {filename}]", flush=True)


def print_state_diff(step: int, before: Dict[str, Any], after: Dict[str, Any]):
    """Prints what fields changed in the explicit state object during this step."""
    print(f"\n>>> [STATE DIFF FOR STEP {step}]:", flush=True)
    
    # Check working changes
    if before.get("working") != after.get("working"):
        print(f"  + working: {list(after.get('working', {}).keys())}", flush=True)
        
    # Check episode changes
    msg_diff = len(after["episode"]["messages"]) - len(before["episode"]["messages"])
    if msg_diff > 0:
        print(f"  + episode.messages: +{msg_diff} new turn(s)", flush=True)
    tool_diff = len(after["episode"]["tool_outputs"]) - len(before["episode"]["tool_outputs"])
    if tool_diff > 0:
        print(f"  + episode.tool_outputs: +{tool_diff} new output(s)", flush=True)

    # Check environment beliefs changes
    new_beliefs = set(after["environment_beliefs"].keys()) - set(before["environment_beliefs"].keys())
    if new_beliefs:
        print(f"  + environment_beliefs added: {list(new_beliefs)}", flush=True)
    for k, v in after["environment_beliefs"].items():
        if k in before["environment_beliefs"]:
            if before["environment_beliefs"][k] != v:
                print(f"  ~ environment_beliefs['{k}'] updated (cached_at: {v['cached_at']:.1f})", flush=True)

    # Check persistent changes
    for k, v in after["persistent"].items():
        if k not in before["persistent"] or before["persistent"][k] != v:
            print(f"  ~ persistent['{k}'] updated: {v}", flush=True)


# ------------------------------------------------------------------------------
# THE CONTEXT WINDOW BRIDGE: Render Prompt from Explicit State
# ------------------------------------------------------------------------------

def render_prompt_from_state(state: Dict[str, Any]) -> str:
    """
    Renders the prompt sent to the LLM from the explicit state object.
    The model ONLY sees what is explicitly rendered into its context window here.
    """
    prompt_parts = [React_loop.SYSTEM_PROMPT.strip()]

    # Inject Persistent context
    prefs = state["persistent"].get("user_preferences", {})
    prompt_parts.append(f"\nUser Preferences: Currency={prefs.get('currency', 'USD')}, Tone={prefs.get('verbosity', 'concise')}")

    # Inject Cached Environment Beliefs if present
    if state.get("environment_beliefs"):
        beliefs_lines = ["\n[Known Cached Environment Beliefs]:"]
        for filename, b in state["environment_beliefs"].items():
            age_sec = int(time.time() - b["cached_at"])
            beliefs_lines.append(f"- File '{filename}' (cached {age_sec}s ago, {b['full_length']} chars):\n{b['content_snapshot']}")
        prompt_parts.append("\n".join(beliefs_lines))

    # Inject Question
    prompt_parts.append(f"\nQuestion: {state['episode']['question']}")

    # Inject Episode History (turn-by-turn)
    for msg in state["episode"]["messages"]:
        if msg["role"] == "assistant":
            prompt_parts.append(f"\n{msg['content']}")
        elif msg["role"] == "observation":
            prompt_parts.append(f"\n{msg['content']}")

    prompt_parts.append("\nWhat's next?")
    return "\n".join(prompt_parts)


# ------------------------------------------------------------------------------
# REASONING & EXECUTION CYCLE (Step 4)
# ------------------------------------------------------------------------------

def decide_next_action(state: Dict[str, Any]) -> str:
    """
    Uses current explicit state to render the prompt, invokes Gemini,
    and stores intermediate outputs into working state.
    """
    rendered_prompt = render_prompt_from_state(state)
    state["working"]["rendered_prompt_tokens_est"] = len(rendered_prompt.split())

    # Call LLM via React_loop module
    llm_response = React_loop.call_llm(rendered_prompt)
    state["working"]["raw_llm_response"] = llm_response

    return llm_response


def execute_action(response_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses and executes the action.
    Notice how environment observations are also cached with timestamps
    into state['environment_beliefs']!
    """
    execution_result = {
        "is_finish": False,
        "tool_called": None,
        "tool_arg": None,
        "observation": None,
        "final_answer": None,
    }

    # Check if final answer
    if "Answer:" in response_text:
        match_ans = re.search(r"Answer:\s*(.*)", response_text, re.DOTALL)
        answer = match_ans.group(1).strip() if match_ans else response_text
        execution_result["is_finish"] = True
        execution_result["final_answer"] = answer
        return execution_result

    # Check for tool call
    match = re.search(r"Action:\s*(\w+):\s*(.*)", response_text, re.IGNORECASE)
    if match:
        tool_name = match.group(1).strip()
        tool_arg = match.group(2).strip().split("\n")[0].replace("PAUSE", "").strip()

        execution_result["tool_called"] = tool_name
        execution_result["tool_arg"] = tool_arg

        # Execute using React_loop's tools with Drift & Cache Checking
        print(f"  [Executing Tool: {tool_name}(arg='{tool_arg}')]", flush=True)
        tool_fn = React_loop.AVAILABLE_TOOLS.get(tool_name)
        if not tool_fn:
            obs = f"Observation: Tool '{tool_name}' not found."
        else:
            now = time.time()
            # Check if this tool touches the external environment
            if tool_name == "get_file_contents":
                cached_belief = state["environment_beliefs"].get(tool_arg)
                
                # DRIFT GUARD: Is belief cached and still fresh (< STALE_THRESHOLD_SECONDS)?
                if cached_belief and (now - cached_belief["cached_at"] <= STALE_THRESHOLD_SECONDS):
                    age = now - cached_belief["cached_at"]
                    print(f"  >>> [CACHE HIT] Using fresh cached belief for '{tool_arg}' ({age:.1f}s old, no disk re-read)", flush=True)
                    raw_output = cached_belief["full_content"]
                    obs = f"Observation (cached {age:.1f}s ago): {raw_output}"
                else:
                    if cached_belief:
                        age = now - cached_belief["cached_at"]
                        print(f"  >>> [DRIFT GUARD / CACHE STALE] Belief for '{tool_arg}' is {age:.1f}s old (> {STALE_THRESHOLD_SECONDS}s). Re-fetching from disk...", flush=True)
                    else:
                        print(f"  >>> [CACHE MISS] No cached belief for '{tool_arg}'. Reading from disk...", flush=True)

                    raw_output = tool_fn(tool_arg)
                    obs = f"Observation: {raw_output}"

                    # Update Environment Belief with new freshness timestamp!
                    state["environment_beliefs"][tool_arg] = {
                        "content_snapshot": raw_output[:250] + ("..." if len(raw_output) > 250 else ""),
                        "full_content": raw_output,
                        "full_length": len(raw_output),
                        "cached_at": now,
                        "source": "disk_read",
                    }
            else:
                raw_output = tool_fn(tool_arg)
                obs = f"Observation: {raw_output}"

        execution_result["observation"] = obs

    return execution_result


def update_state(
    state: Dict[str, Any],
    action_thought: str,
    result: Dict[str, Any],
    persistent_db: PersistentDatabase,
    episode_store: EpisodeStore,
) -> Dict[str, Any]:
    """
    Updates the four state categories explicitly after the action has run.
    """
    # 1. Update Working State (temporary bookkeeping for this step)
    state["working"]["tool_called"] = result["tool_called"]
    state["working"]["is_finished"] = result["is_finish"]

    # 2. Update Episode State (messages and tool outputs)
    state["episode"]["step_count"] += 1
    state["episode"]["messages"].append({
        "step": state["episode"]["step_count"],
        "role": "assistant",
        "content": action_thought,
    })

    if result["observation"]:
        state["episode"]["messages"].append({
            "step": state["episode"]["step_count"],
            "role": "observation",
            "content": result["observation"],
        })
        state["episode"]["tool_outputs"].append({
            "tool": result["tool_called"],
            "arg": result["tool_arg"],
            "output": result["observation"],
            "timestamp": time.time(),
        })

    # 3. Update Persistent State if task completed
    if result["is_finish"]:
        tasks_done = state["persistent"].get("total_tasks_completed", 0) + 1
        state["persistent"]["total_tasks_completed"] = tasks_done
        persistent_db.save_key("total_tasks_completed", tasks_done)
        persistent_db.save_key("last_completed_session", state["episode"]["session_id"])

    # 4. Save episode to episode store
    episode_store.save_episode(state["episode"]["session_id"], state["episode"])

    return state


# ------------------------------------------------------------------------------
# 4. STRUCTURED RE-ACT EXECUTION CYCLE
# ------------------------------------------------------------------------------

def run_explicit_loop(
    question: str,
    max_steps: int = 6,
    session_id: str = "session_001",
    existing_beliefs: Optional[Dict[str, Any]] = None,
):
    """
    Runs the ReAct loop where all state transitions happen via the single explicit state object.
    Can inherit existing_beliefs from a previous run to demonstrate cache hits and drift detection.
    """
    print("\n" + "=" * 75, flush=True)
    print(f"RUNNING EXPLICIT FOUR-LAYER STATE AGENT", flush=True)
    print(f"Session ID: {session_id}", flush=True)
    print(f"Question:   {question}", flush=True)
    if existing_beliefs:
        print(f"Inherited Environment Beliefs: {list(existing_beliefs.keys())}", flush=True)
    print("=" * 75, flush=True)

    # Initialize physical backends
    persistent_db = PersistentDatabase("agent_persistent.db")
    episode_store = EpisodeStore()

    # Initialize explicit 4-layer state object
    state = initialize_state(session_id, question, persistent_db, episode_store, existing_beliefs=existing_beliefs)

    for step in range(1, max_steps + 1):
        print(f"\n==================== [STEP {step}] ====================", flush=True)

        # Working state is cleared at the start of every single step!
        state["working"] = {
            "current_step": step,
            "step_start_time": time.time(),
        }

        # Keep state_before snapshot for diff calculation only
        state_before = copy.deepcopy(state)

        # 1. Decide next action
        action_thought = decide_next_action(state)
        print(f"\n[Model Thought & Action]:\n{action_thought}\n", flush=True)

        # 2. Execute action
        exec_result = execute_action(action_thought, state)
        if exec_result["observation"]:
            print(f"\n[Tool Observation]:\n{exec_result['observation']}\n", flush=True)

        # 3. Update state
        state = update_state(state, action_thought, exec_result, persistent_db, episode_store)

        # 4. Log state after update & print diff
        log_state(step, state)
        print_state_diff(step, state_before, state)

        if exec_result["is_finish"]:
            print(f"\n>>> Agent finished at Step {step}!", flush=True)
            print(f">>> FINAL ANSWER:\n{exec_result['final_answer']}", flush=True)
            break

    print("\n" + "=" * 75, flush=True)
    print("EXPLICIT STATE REPLAY & INSPECTION SUMMARY:")
    print("=" * 75, flush=True)
    print(f"1. Working State: Cleared at every step (Currently holds: {list(state['working'].keys())})")
    print(f"2. Episode State: Structured turns saved in episode store ({len(state['episode']['messages'])} messages)")
    print(f"3. Persistent State: Stored in SQLite (Tasks completed: {state['persistent'].get('total_tasks_completed')})")
    print(f"4. Environment Beliefs: Timestamped cached snapshots of outside world:")
    for path, belief in state["environment_beliefs"].items():
        age = time.time() - belief["cached_at"]
        print(f"     - '{path}': age={age:.1f}s, length={belief['full_length']} chars (fresh: {age < STALE_THRESHOLD_SECONDS})")
    print(f"\nAll step snapshots are logged as JSON in './{LOGS_DIR}/' for diffing & replay.")
    print("=" * 75, flush=True)

    return state


if __name__ == "__main__":
    print("\n" + "#" * 75)
    print("EXERCISE: Testing Environment Belief Caching and Drift Detection")
    print("#" * 75)

    # Question 1: Cold start (Cache Miss - reads report.txt from disk)
    q1 = "What is the remaining budget in Engineering, and how much is it as a percentage of their total allocated budget?"
    state_after_q1 = run_explicit_loop(q1, max_steps=5, session_id="trajectory_01")

    # Question 2: Follow-up question in the same session (Within 60s -> CACHE HIT!)
    q2 = "What is the total headcount in Engineering according to the report?"
    print("\n\n" + "#" * 75)
    print("FOLLOW-UP QUESTION: Demonstrating Cache Hit from Environment Beliefs")
    print("#" * 75)
    state_after_q2 = run_explicit_loop(
        q2,
        max_steps=5,
        session_id="trajectory_01_followup",
        existing_beliefs=state_after_q1["environment_beliefs"]
    )
