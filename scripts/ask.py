"""Test the graph from the terminal (without FastAPI).

  python -m scripts.ask "How do I declare a path parameter?"   # a single question
  python -m scripts.ask --chat                                  # chat with follow-up questions (memory test)
  python -m scripts.ask --graph                                 # Mermaid diagram of the graph (for the README)
"""
import sys

from app.graph import get_graph, run_query


def show(result: dict) -> None:
    print("\n=== ANSWER ===")
    print(result["answer"])
    print(f"\nstatus: {result['status']}   grounded: {result.get('grounded')}   retries: {result['retry_count']}")
    if result.get("sources"):
        print("\n=== SOURCES ===")
        for s in result["sources"]:
            where = s["url"] or s["source"]
            print(f"  [{s['id']}] {s['source']} | {s['section']}  (score {s['score']})\n      {where}")
    print("\n=== TRACE (what the graph did) ===")
    for line in result["trace"]:
        print(line if line.startswith(" ") else f"  - {line}")


def chat() -> None:
    history: list[dict] = []
    print("Chat mode. You can ask follow-up questions ('and how do I make that optional?'). Quit: Ctrl+C or 'exit'.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit", ""}:
            break
        result = run_query(q, history)
        show(result)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": result["answer"]}]


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
    elif args == ["--graph"]:
        print(get_graph().get_graph().draw_mermaid())
    elif args == ["--chat"]:
        chat()
    else:
        show(run_query(" ".join(args)))


if __name__ == "__main__":
    main()