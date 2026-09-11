import os
import re
import sys
from dotenv import load_dotenv
from google import genai

# -------------------------------------------------------------
# 0. API & Client Initialization
# -------------------------------------------------------------
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment or .env file. Please check .env.")

client = genai.Client(api_key=api_key)

# -------------------------------------------------------------
# 1. Simple Tools (Interactions with External World)
# -------------------------------------------------------------
def get_file_contents(filename: str) -> str:
    """Read contents of a file from disk."""
    filename = filename.strip().strip("'\"")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        return f"Error reading file '{filename}': {e}"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    expression = expression.strip().strip("'\"")
    try:
        allowed_chars = set("0123456789+-*/(). %")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in math expression."
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating '{expression}': {e}"

AVAILABLE_TOOLS = {
    "get_file_contents": get_file_contents,
    "calculate": calculate,
}

# -------------------------------------------------------------
# 2. ReAct System Prompt
# -------------------------------------------------------------
SYSTEM_PROMPT = """You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.

Use Action: <tool_name>: <arguments>
Followed strictly by PAUSE on a new line.

Available tools:
- get_file_contents: filename
- calculate: math_expression

Example format:
Question: How many servers are in US-East?
Thought: I need to check report.txt to see the list of servers.
Action: get_file_contents: report.txt
PAUSE

Observation: [Section 1: Infrastructure & Fleet Health] ...

Thought: I see 4 servers listed in US-East: srv-useast-01, 02, 03, 04.
Answer: There are 4 servers in US-East.
"""

# -------------------------------------------------------------
# 3. LLM Caller and Tool Dispatcher
# -------------------------------------------------------------
import time

def call_llm(prompt: str, model_name: str = "gemini-3.6-flash") -> str:
    """Call Gemini API with real-time logging, timing, and error reporting."""
    print(f"  [Calling Gemini ({model_name})...", end="", flush=True)
    t0 = time.time()
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        elapsed = time.time() - t0
        print(f" done in {elapsed:.1f}s]", flush=True)
        return response.text.strip()
    except Exception as e:
        elapsed = time.time() - t0
        print(f" failed in {elapsed:.1f}s: {e}]", flush=True)
        
        # Fallback to gemini-3.5-flash-lite
        fallback_model = "gemini-3.5-flash-lite"
        print(f"  [Attempting fallback with ({fallback_model})...", end="", flush=True)
        t1 = time.time()
        try:
            response = client.models.generate_content(
                model=fallback_model,
                contents=prompt,
            )
            print(f" done in {time.time() - t1:.1f}s]", flush=True)
            return response.text.strip()
        except Exception as e2:
            print(f" fallback failed: {e2}]", flush=True)
            raise RuntimeError(f"All Gemini model calls failed:\nPrimary ({model_name}): {e}\nFallback ({fallback_model}): {e2}") from e


def maybe_call_tool(response_text: str) -> str:
    """Parses 'Action: tool_name: arg' and executes it, returning 'Observation: ...'"""
    match = re.search(r"Action:\s*(\w+):\s*(.*)", response_text, re.IGNORECASE)
    if not match:
        return ""
    
    tool_name = match.group(1).strip()
    tool_arg = match.group(2).strip()

    # Strip trailing PAUSE or extra newlines if the model included it
    tool_arg = tool_arg.split("\n")[0].replace("PAUSE", "").strip()

    print(f"  [Executing Tool: {tool_name}(arg='{tool_arg}')]", flush=True)

    tool_fn = AVAILABLE_TOOLS.get(tool_name)
    if not tool_fn:
        err = f"Observation: Tool '{tool_name}' not found. Available tools: {list(AVAILABLE_TOOLS.keys())}"
        print(f"  [Tool Error]: {err}", flush=True)
        return err
    
    result = tool_fn(tool_arg)
    return f"Observation: {result}"

# -------------------------------------------------------------
# 4. Naive ReAct Loop (Implicit State via String Concatenation)
# -------------------------------------------------------------
def run_naive_loop(question: str, max_steps: int = 10) -> str:
    """Runs the naive ReAct loop where everything is concatenated into a single history string."""
    print("\n" + "=" * 70, flush=True)
    print(f"QUESTION: {question}", flush=True)
    print("=" * 70, flush=True)

    # Implicit state: everything lives in this growing string!
    history = SYSTEM_PROMPT + f"\nQuestion: {question}"

    for step in range(1, max_steps + 1):
        print(f"\n--- [STEP {step}] ---", flush=True)
        prompt = history + "\nWhat's next?"
        
        # 1. LLM call
        try:
            response = call_llm(prompt)
        except Exception as err:
            print(f"\n[CRITICAL ERROR AT STEP {step}]: {err}", flush=True)
            break

        print(f"\n[Model Output]:\n{response}\n", flush=True)

        # 2. Check if agent finished
        if "Answer:" in response:
            print(f">>> Agent finished at Step {step}!", flush=True)
            history += f"\n{response}"
            break

        # 3. Tool execution
        tool_result = maybe_call_tool(response)
        if tool_result:
            print(f"[Tool Observation]:\n{tool_result}\n", flush=True)
        
        # 4. Implicit State update (string gluing)
        history += f"\n{response}\n{tool_result}"

    print("\n" + "=" * 70, flush=True)
    print("FINAL ACCUMULATED 'HISTORY' STRING (IMPLICIT STATE):", flush=True)
    print("=" * 70, flush=True)
    print(history, flush=True)
    print("=" * 70, flush=True)
    # print("THE PROBLEM WITH IMPLICIT STATE:", flush=True)
    # print("1. Working State (e.g. current step's regex match, tool args) is lost or glued into history.", flush=True)
    # print("2. Episode State (turn-by-turn messages, tool calls) is one unparseable text blob.", flush=True)
    # print("3. Persistent State (task status, session memory) doesn't exist.", flush=True)
    # print("4. Environment State (report.txt snapshot) is stale and un-timestamped.", flush=True)
    # print("=" * 70, flush=True)

    return history

if __name__ == "__main__":
    test_question = "What is the remaining budget in Engineering, and how much is it as a percentage of their total allocated budget?"
    run_naive_loop(test_question, max_steps=6)

