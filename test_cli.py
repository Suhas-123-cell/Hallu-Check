
import os
import sys
from typing import Optional
import time

try:
    import requests
except ImportError:
    print("Error: 'requests' module is missing. Please install it:")
    print("pip install requests")
    sys.exit(1)

API_URL       = "http://127.0.0.1:8000/generate"
SUMMARIZE_URL = "http://127.0.0.1:8000/summarize"

# Summarize after every N completed turns
SUMMARIZE_EVERY_N = 2

# ~4 chars per token; warn at 3000 tokens, hard-stop at 5000 tokens
_CONTEXT_WARN_CHARS  = 12_000
_CONTEXT_LIMIT_CHARS = 20_000

VERDICT_ICONS = {
    "SUPPORTED":          "+",
    "CONTRADICTED":       "x",
    "UNVERIFIABLE":       "?",
    "NO_CLAIM":           "-",
    "HONEST_UNCERTAINTY": "~",
}
VERDICT_COLORS = {
    "SUPPORTED":          "\033[92m",
    "CONTRADICTED":       "\033[91m",
    "UNVERIFIABLE":       "\033[93m",
    "NO_CLAIM":           "\033[90m",
    "HONEST_UNCERTAINTY": "\033[96m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def _truncate(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "…" if len(text) > max_len else text


def _history_chars(history: list) -> int:
    return sum(len(m.get("content", "")) for m in history)


def _estimated_tokens(history: list) -> int:
    return _history_chars(history) // 4


def print_header():
    print("  hallu-correct v3.0 - claim-level hallucination detection")
    print(f"  {DIM}type 'exit' or 'quit' to stop.{RESET}\n")


def print_claim_verdicts(verdicts: list):
    if not verdicts:
        print(f"\n  {DIM}no factual claims detected.{RESET}")
        return

    print(f"\n  {BOLD}claim-by-claim analysis:{RESET}")
    print(f"  {'─' * 56}")

    for v in verdicts:
        verdict    = v.get("verdict", "UNKNOWN")
        icon       = VERDICT_ICONS.get(verdict, "?")
        color      = VERDICT_COLORS.get(verdict, "")
        claim      = _truncate(v.get("claim", ""), 100)
        evidence   = _truncate(v.get("evidence", ""), 100)
        reasoning  = _truncate(v.get("reasoning", ""), 120)
        confidence = v.get("confidence", 0.0)

        print(f"  {color}{icon} [{verdict}]{RESET} (conf: {confidence:.0%})")
        print(f"     claim:    {claim}")
        if evidence and verdict != "NO_CLAIM":
            print(f"     {DIM}evidence: {evidence}{RESET}")
        if reasoning:
            print(f"     {DIM}reasoning: {reasoning}{RESET}")
        print()


def print_hallucination_score(score: float, detected: bool):
    bar_len = 30
    filled  = int(score * bar_len)
    empty   = bar_len - filled

    if score < 0.3:
        bar_color = "\033[92m"
    elif score < 0.6:
        bar_color = "\033[93m"
    else:
        bar_color = "\033[91m"

    bar    = f"{bar_color}{'█' * filled}{'░' * empty}{RESET}"
    status = (
        f"{BOLD}\033[91mhallucination detected{RESET}"
        if detected else
        f"{BOLD}\033[92mno hallucination{RESET}"
    )
    print(f"  hallucination score: [{bar}] {score:.2f}")
    print(f"     status: {status}")


def _read_query_interactive() -> Optional[str]:
    print(f"\n  {BOLD}enter your query (type or paste, then press ctrl+d to submit):{RESET}")
    try:
        query = sys.stdin.read().strip()
    except KeyboardInterrupt:
        print("\n  exiting...")
        return None

    try:
        sys.stdin = open("/dev/tty", "r")
    except OSError:
        pass

    return query


def _call_summarize(history: list) -> Optional[str]:
    """Call /summarize and return the summary string, or None on failure."""
    try:
        resp = requests.post(SUMMARIZE_URL, json={"history": history}, timeout=120)
        resp.raise_for_status()
        return resp.json().get("summary", "").strip()
    except Exception as e:
        print(f"\n  {DIM}[summarization failed: {e}]{RESET}")
        return None


def _process_query(query: str, history: list) -> Optional[dict]:
    """Post to /generate and return the parsed JSON dict, or None on error."""
    print(f"\n  processing pipeline (this may take 10-60 seconds)...")
    start = time.time()
    response = None
    try:
        response = requests.post(
            API_URL,
            json={"query": query, "conversation_history": history},
            timeout=300,
        )
        response.raise_for_status()
        data    = response.json()
        elapsed = time.time() - start

        if data.get("cache_hit"):
            print(f"\n  {BOLD}\033[96m⚡ answered from cache{RESET} ({elapsed:.2f}s)")
        else:
            print(f"\n  pipeline completed in {elapsed:.1f}s")

        category = data.get("query_category", "UNKNOWN")
        print(f"\n  route: {BOLD}{category}{RESET}")

        print(f"\n  {'─' * 56}")
        print(f"  {BOLD}qwen's original output:{RESET}")
        for line in data["llm_output"].split("\n"):
            print(f"     {line}")

        print(f"\n  {'─' * 56}")
        rag_out     = data["rag_output"]
        rag_preview = rag_out[:400] + "…" if len(rag_out) > 400 else rag_out
        print(f"  {BOLD}rag context (pageindex):{RESET}")
        for line in rag_preview.split("\n")[:8]:
            print(f"     {DIM}{line}{RESET}")

        print(f"\n  {'─' * 56}")
        bs = data["alignment_score"]
        print(f"  {BOLD}alignment score:{RESET}")
        print(f"     F1: {bs['f1']:.4f}  |  Precision: {bs['precision']:.4f}  |  Recall: {bs['recall']:.4f}")

        print(f"\n  {'─' * 56}")
        print_claim_verdicts(data.get("claim_verdicts", []))

        print(f"  {'─' * 56}")
        print_hallucination_score(
            data.get("hallucination_score", 0.0),
            data.get("hallucination_detected", False),
        )

        hall_summary = data.get("hallucination_summary", "")
        if hall_summary:
            print(f"\n  summary: {hall_summary}")

        print(f"\n  {'─' * 56}")
        print(f"  {BOLD}final answer:{RESET}")
        for line in data["final_answer"].split("\n"):
            print(f"     {line}")

        print(f"\n{'═' * 64}\n")
        return data

    except requests.exceptions.ConnectionError:
        print("\n  error: cannot connect to the api.")
        print("  make sure the server is running:")
        print("    source .venv/bin/activate && uvicorn main:app --reload")
    except requests.exceptions.Timeout:
        print("\n  error: the request timed out.")
    except requests.exceptions.HTTPError as err:
        print(f"\n  api error: {err}")
        if response:
            print(f"  details: {response.text[:500]}")
    except Exception as e:
        print(f"\n  unexpected error: {e}")
    return None


def main():
    # ── --file flag for large inputs ──────────────────────────────────────────
    if len(sys.argv) >= 3 and sys.argv[1] == "--file":
        filepath = sys.argv[2]
        if not os.path.isfile(filepath):
            print(f"  file not found: {filepath}")
            sys.exit(1)
        with open(filepath) as f:
            query = f.read().strip()
        if not query:
            print("  file is empty.")
            sys.exit(1)
        print(f"  read {len(query)} chars from {filepath}")
        _process_query(query, [])
        return

    print_header()

    conversation_history: list = []   # [{role, content}, ...]
    turn_count = 0                    # completed Q-A pairs

    while True:
        # ── Context limit check ───────────────────────────────────────────────
        hist_chars = _history_chars(conversation_history)
        if hist_chars >= _CONTEXT_LIMIT_CHARS:
            print(f"\n  {BOLD}\033[91m⚠ context limit reached (~{_estimated_tokens(conversation_history)} tokens).{RESET}")
            print(f"  {BOLD}start a new session to continue.{RESET}\n")
            break

        if hist_chars >= _CONTEXT_WARN_CHARS:
            print(f"\n  {DIM}[approaching context limit — ~{_estimated_tokens(conversation_history)} tokens used]{RESET}")

        query = _read_query_interactive()
        if query is None:
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("  goodbye!")
            break

        data = _process_query(query, conversation_history)
        if data is None:
            continue

        # ── Append this turn to history ───────────────────────────────────────
        conversation_history.append({"role": "user",      "content": query})
        conversation_history.append({"role": "assistant", "content": data["final_answer"]})
        turn_count += 1

        # ── Summarize every SUMMARIZE_EVERY_N turns ───────────────────────────
        if turn_count % SUMMARIZE_EVERY_N == 0:
            print(f"  {DIM}[summarizing {turn_count} turns via llama…]{RESET}")
            summary_text = _call_summarize(conversation_history)
            if summary_text:
                conversation_history = [
                    {"role": "system", "content": f"[Conversation summary]: {summary_text}"}
                ]
                print(f"  {DIM}[session compressed → {summary_text[:120]}…]{RESET}\n")
            else:
                print(f"  {DIM}[summarization skipped — keeping full history]{RESET}\n")


if __name__ == "__main__":
    main()
