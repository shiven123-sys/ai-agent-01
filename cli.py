"""
Command-line chat interface for the agent.

Usage:
    python cli.py
"""

import sys

from agent.agent import Agent
from agent import config


def main():
    print("=" * 60)
    print(" Agentic AI Assistant (CLI)")
    print(f" Provider: {config.LLM_PROVIDER}  |  Model: {config.MODEL}")
    print("=" * 60)

    if not config.is_configured():
        print("\n⚠️  Setup required:\n")
        print(config.setup_instructions())
        sys.exit(1)

    try:
        agent = Agent()
    except RuntimeError as e:
        print(f"\n⚠️  {e}")
        sys.exit(1)

    print("\nType your message. Commands: /reset to clear memory, /quit to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit"):
            print("Bye!")
            break
        if user_input.lower() == "/reset":
            agent.reset()
            print("(conversation memory cleared)\n")
            continue

        try:
            result = agent.chat(user_input)
        except Exception as e:
            print(f"\n⚠️  Error: {e}\n")
            continue

        for call in result["tool_calls"]:
            print(f"  🔧 used tool: {call['name']}({call['arguments']})")

        print(f"Assistant: {result['reply']}\n")


if __name__ == "__main__":
    main()
