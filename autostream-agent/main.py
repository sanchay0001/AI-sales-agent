import os
import sys
from dotenv import load_dotenv
from agent.graph import run_agent

load_dotenv()

# ─────────────────────────────────────────────
# Suppress noisy warnings for clean demo
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def print_banner():
    print("\n" + "=" * 55)
    print("       🎬  AutoStream AI Sales Agent  🎬")
    print("=" * 55)
    print("  AI-powered video editing for content creators")
    print("  Type 'quit' or 'exit' to end the conversation")
    print("=" * 55 + "\n")


def print_agent(response: str):
    print(f"\n🤖 AutoStream Agent:\n{response}\n")
    print("-" * 55)


def print_lead_status(result: dict):
    """Show lead collection progress during conversation."""
    name = result.get("lead_name")
    email = result.get("lead_email")
    platform = result.get("lead_platform")
    captured = result.get("lead_captured", False)

    # Only show status during lead collection
    if any([name, email, platform]) and not captured:
        collected = []
        if name:
            collected.append(f"name: {name}")
        if email:
            collected.append(f"email: {email}")
        if platform:
            collected.append(f"platform: {platform}")
        print(f"  📋 Lead info collected so far → {', '.join(collected)}")


def main():
    print_banner()

    history = []

    print("🤖 AutoStream Agent:")
    print("Hello! Welcome to AutoStream — AI-powered video editing")
    print("for content creators. How can I help you today?\n")
    print("-" * 55)

    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()

            # Exit conditions
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
                print("\n🤖 AutoStream Agent:")
                print("Thanks for chatting with AutoStream! Have a great day. 🎬\n")
                break

            # Run the agent
            response, history, result = run_agent(user_input, history)

            # Print agent response
            print_agent(response)

            # Show lead collection progress (helpful for demo)
            print_lead_status(result)

            # If lead captured, show success and optionally exit
            if result.get("lead_captured"):
                print("\n✅ Lead successfully captured in the system!")
                print("   The conversation can continue or type 'exit' to end.\n")

        except KeyboardInterrupt:
            print("\n\nSession ended. Goodbye! 👋\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()