
import os
import sys
from typing import Optional
import time

try:
    import requests  # type: ignore[import-untyped]
except ImportError:
    print("Error: 'requests' module is missing. Please install it:")
    print("pip install requests")
    sys.exit(1)

API_URL = "http://127.0.0.1:8000/generate"

# ── Verdict display config 
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
BOLD = "\033[1m"
DIM = "\033[2m"


def _truncate(text: str, max_len: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "…" if len(text) > max_len else text


def print_header():
    print("  hallu-correct v3.0 - claim-level hallucination detection")
    
    print(f"  {DIM}type 'exit' or 'quit' to stop.{RESET}\n")


def print_claim_verdicts(verdicts: list):
    if not verdicts:
        print(f"\n  {DIM}no factual claims detected.{RESET}")
        return

    print(f"\n  {BOLD}claim-by-claim analysis:{RESET}")
    print(f"  {'─' * 56}")

    for i, v in enumerate(verdicts, 1):
        verdict = v.get("verdict", "UNKNOWN")
        icon = VERDICT_ICONS.get(verdict, "?")
        color = VERDICT_COLORS.get(verdict, "")
        claim = _truncate(v.get("claim", ""), 100)
        evidence = _truncate(v.get("evidence", ""), 100)
        reasoning = _truncate(v.get("reasoning", ""), 120)
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
    filled = int(score * bar_len)
    empty = bar_len - filled

    if score < 0.3:
        bar_color = "\033[92m"  
    elif score < 0.6:
        bar_color = "\033[93m"  
    else:
        bar_color = "\033[91m"  

    bar = f"{bar_color}{'█' * filled}{'░' * empty}{RESET}"
    status = f"{BOLD}\033[91mhallucination detected{RESET}" if detected else f"{BOLD}\033[92mno hallucination{RESET}"

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


def main():
    # ── Handle --file flag for very large inputs ─────────────────────────
    if len(sys.argv) >= 3 and sys.argv[1] == "--file":
        filepath = sys.argv[2]
        if not os.path.isfile(filepath):
            print(f"  file not found: {filepath}")
            sys.exit(1)
        with open(filepath, "r") as f:
            query = f.read().strip()
        if not query:
            print("  file is empty.")
            sys.exit(1)
        print(f"  read {len(query)} chars from {filepath}")
        _process_query(query)
        return

    print_header()

    while True:
        query = _read_query_interactive()

        if query is None:
            break  # Ctrl+C → exit

        if not query:
            continue

        if query.lower() in ("exit", "quit"):
            print("  goodbye!")
            break

        _process_query(query)


def _process_query(query: str) -> None:
    print(f"\n  processing pipeline (this may take 10-60 seconds)...")
    start_time = time.time()
    response = None
    try:
        response = requests.post(API_URL, json={"query": query}, timeout=300)
        response.raise_for_status()
        data = response.json()
        elapsed = time.time() - start_time

        if elapsed > 60:
            print(f"\n  pipeline took {elapsed:.1f}s (longer than usual).")
        else:
            print(f"\n  pipeline completed in {elapsed:.1f}s")

        # ── Gatekeeper Classification ─────────────────────────────────
        category = data.get("query_category", "UNKNOWN")
        print(f"\n  route: {BOLD}{category}{RESET}")

        # ── Qwen's Original Output ─────────────────────────────────────
        print(f"\n  {'─' * 56}")
        print(f"  {BOLD}qwen's original output:{RESET}")
        llm_out = data["llm_output"]
        for line in llm_out.split("\n"):
            print(f"     {line}")

        # ── RAG Ground Truth ───────────────────────────────────────────
        print(f"\n  {'─' * 56}")
        rag_out = data["rag_output"]
        rag_preview = rag_out[:400] + "…" if len(rag_out) > 400 else rag_out
        print(f"  {BOLD}rag context (pageindex):{RESET}")
        for line in rag_preview.split("\n")[:8]:
            print(f"     {DIM}{line}{RESET}")

        # ── BERTScore ──────────────────────────────────────────────────
        print(f"\n  {'─' * 56}")
        bs = data["alignment_score"]
        print(f"  {BOLD}alignment score:{RESET}")
        print(f"     F1: {bs['f1']:.4f}  |  Precision: {bs['precision']:.4f}  |  Recall: {bs['recall']:.4f}")

        # ── Claim-Level Verdicts ───────────────────────────────────────
        print(f"\n  {'─' * 56}")
        print_claim_verdicts(data.get("claim_verdicts", []))

        # ── Hallucination Score ────────────────────────────────────────
        print(f"  {'─' * 56}")
        print_hallucination_score(
            data.get("hallucination_score", 0.0),
            data.get("hallucination_detected", False),
        )

        # ── Summary ───────────────────────────────────────────────────
        summary = data.get("hallucination_summary", "")
        if summary:
            print(f"\n  summary: {summary}")

        # ── Final Answer ───────────────────────────────────────────────
        print(f"\n  {'─' * 56}")
        print(f"  {BOLD}final answer:{RESET}")
        for line in data["final_answer"].split("\n"):
            print(f"     {line}")

        print(f"\n{'═' * 64}\n")

    except requests.exceptions.ConnectionError:
        print(f"\n  error: cannot connect to the api.")
        print("  please make sure the server is running:")
        print("    source .venv/bin/activate")
        print("    uvicorn main:app --reload")
    except requests.exceptions.Timeout:
        print(f"\n  error: the request timed out.")
    except requests.exceptions.HTTPError as err:
        print(f"\n  api error: {err}")
        if response:
            print(f"  details: {response.text[:500]}")
    except Exception as e:
        print(f"\n  unexpected error: {e}")


if __name__ == "__main__":
    main()
