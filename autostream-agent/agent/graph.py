import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from agent.intent import detect_intent
from agent.rag import retrieve_context
from agent.tools import mock_lead_capture

load_dotenv()

# ─────────────────────────────────────────────
# 1. STATE DEFINITION
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    user_message: str               # Latest user message
    history: list[dict]             # Full conversation history
    intent: str                     # Detected intent
    context: str                    # RAG retrieved context
    response: str                   # Agent response to send back
    lead_name: Optional[str]        # Collected lead name
    lead_email: Optional[str]       # Collected lead email
    lead_platform: Optional[str]    # Collected lead platform
    lead_captured: bool             # Whether lead has been captured


# ─────────────────────────────────────────────
# 2. LLM SETUP
# ─────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)


# ─────────────────────────────────────────────
# 3. HELPER — Format history for prompts
# ─────────────────────────────────────────────
def format_history(history: list[dict]) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history[-6:]:  # Keep last 6 turns for context
        lines.append(f"User: {turn['user']}")
        lines.append(f"Agent: {turn['agent']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 4. NODE — Classify Intent
# ─────────────────────────────────────────────
def node_classify_intent(state: AgentState) -> AgentState:
    history_text = format_history(state["history"])
    intent = detect_intent(state["user_message"], history_text)
    return {**state, "intent": intent}


# ─────────────────────────────────────────────
# 5. NODE — Handle Greeting
# ─────────────────────────────────────────────
def node_handle_greeting(state: AgentState) -> AgentState:
    prompt = PromptTemplate(
        input_variables=["message"],
        template="""
You are a friendly sales assistant for AutoStream, an AI-powered video editing SaaS for content creators.

The user said: {message}

Respond with a warm, brief greeting and mention you can help with:
- Pricing and plan information
- Product features
- Getting started

Keep it to 2-3 sentences max.
"""
    )
    chain = prompt | llm
    result = chain.invoke({"message": state["user_message"]})
    return {**state, "response": result.content.strip()}


# ─────────────────────────────────────────────
# 6. NODE — Handle Product Inquiry (RAG)
# ─────────────────────────────────────────────
def node_handle_inquiry(state: AgentState) -> AgentState:
    context = retrieve_context(state["user_message"])

    prompt = PromptTemplate(
        input_variables=["message", "context", "history"],
        template="""
You are a knowledgeable sales assistant for AutoStream, an AI-powered video editing SaaS.

Conversation history:
{history}

Knowledge base context:
{context}

User question: {message}

Answer the question accurately using ONLY the context provided.
Be helpful, concise and friendly.
At the end, subtly invite them to consider signing up if the information seems relevant.
"""
    )
    chain = prompt | llm
    result = chain.invoke({
        "message": state["user_message"],
        "context": context,
        "history": format_history(state["history"])
    })
    return {**state, "context": context, "response": result.content.strip()}


# ─────────────────────────────────────────────
# 7. NODE — Handle Lead Collection
# ─────────────────────────────────────────────
def node_handle_lead(state: AgentState) -> AgentState:

    # If lead already captured, don't ask again
    if state.get("lead_captured"):
        response = "You're already signed up! Our team will reach out to you shortly. 😊"
        return {**state, "response": response}

    # Extract any info from the current message
    updated_state = extract_lead_info(state)

    # Check what's still missing
    missing = []
    if not updated_state.get("lead_name"):
        missing.append("your full name")
    if not updated_state.get("lead_email"):
        missing.append("your email address")
    if not updated_state.get("lead_platform"):
        missing.append("your creator platform (e.g. YouTube, Instagram, TikTok)")

    # All collected — fire the tool
    if not missing:
        mock_lead_capture(
            updated_state["lead_name"],
            updated_state["lead_email"],
            updated_state["lead_platform"]
        )
        response = (
            f"🎉 You're all set! We've captured your details:\n"
            f"- Name: {updated_state['lead_name']}\n"
            f"- Email: {updated_state['lead_email']}\n"
            f"- Platform: {updated_state['lead_platform']}\n\n"
            f"Our team will reach out to you shortly to get you started on the Pro plan!"
        )
        return {**updated_state, "response": response, "lead_captured": True}

    # Ask for the next missing field
    if len(missing) == 3:
        # Nothing collected yet — show excitement and ask for name
        response = (
            "That's awesome! 🎬 I'd love to get you started with AutoStream.\n"
            "To set up your account, could you please share your full name?"
        )
    else:
        response = f"Thanks! Could you also share {missing[0]}?"

    return {**updated_state, "response": response}


# ─────────────────────────────────────────────
# 8. HELPER — Extract lead info from message
# ─────────────────────────────────────────────
def extract_lead_info(state: AgentState) -> AgentState:
    """Use LLM to extract name, email, platform from user message."""

    prompt = PromptTemplate(
        input_variables=["message", "history"],
        template="""
Extract lead information from the conversation if present.

Conversation history:
{history}

Latest message: {message}

Extract the following if mentioned anywhere in the conversation:
- name: person's full name (or first name)
- email: email address
- platform: content platform (YouTube, Instagram, TikTok, etc.)

Reply in this EXACT format (use null if not found):
name: <value or null>
email: <value or null>
platform: <value or null>
"""
    )
    chain = prompt | llm
    result = chain.invoke({
        "message": state["user_message"],
        "history": format_history(state["history"])
    })

    # Parse the LLM output
    lines = result.content.strip().split("\n")
    extracted = {}
    for line in lines:
        if ":" in line:
            key, _, value = line.partition(":")
            extracted[key.strip().lower()] = value.strip()

    # Only update state if new info found (don't overwrite existing)
    updated = {**state}
    if extracted.get("name") and extracted["name"] != "null" and not state.get("lead_name"):
        updated["lead_name"] = extracted["name"]
    if extracted.get("email") and extracted["email"] != "null" and not state.get("lead_email"):
        updated["lead_email"] = extracted["email"]
    if extracted.get("platform") and extracted["platform"] != "null" and not state.get("lead_platform"):
        updated["lead_platform"] = extracted["platform"]

    return updated


# ─────────────────────────────────────────────
# 9. ROUTER — Decide next node based on intent
# ─────────────────────────────────────────────
def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "product_inquiry")
    if intent == "casual_greeting":
        return "handle_greeting"
    elif intent == "high_intent_lead":
        return "handle_lead"
    else:
        return "handle_inquiry"


# ─────────────────────────────────────────────
# 10. BUILD THE GRAPH
# ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", node_classify_intent)
    graph.add_node("handle_greeting", node_handle_greeting)
    graph.add_node("handle_inquiry", node_handle_inquiry)
    graph.add_node("handle_lead", node_handle_lead)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional routing after intent classification
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_inquiry": "handle_inquiry",
            "handle_lead": "handle_lead"
        }
    )

    # All handler nodes end the graph
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_inquiry", END)
    graph.add_edge("handle_lead", END)

    return graph.compile()


# Compile once at import time
agent_graph = build_graph()


# ─────────────────────────────────────────────
# 11. MAIN RUN FUNCTION
# ─────────────────────────────────────────────
def run_agent(user_message: str, history: list[dict]) -> tuple[str, list[dict], dict]:
    """
    Run one turn of the agent.
    Returns: (response, updated_history, updated_lead_info)
    """

    # Build initial state
    state: AgentState = {
        "user_message": user_message,
        "history": history,
        "intent": "",
        "context": "",
        "response": "",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False
    }

    # Carry over lead info from history if available
    if history:
        last_state = history[-1].get("state", {})
        state["lead_name"] = last_state.get("lead_name")
        state["lead_email"] = last_state.get("lead_email")
        state["lead_platform"] = last_state.get("lead_platform")
        state["lead_captured"] = last_state.get("lead_captured", False)

    # Run the graph
    result = agent_graph.invoke(state)

    # Update history
    history.append({
        "user": user_message,
        "agent": result["response"],
        "state": {
            "lead_name": result.get("lead_name"),
            "lead_email": result.get("lead_email"),
            "lead_platform": result.get("lead_platform"),
            "lead_captured": result.get("lead_captured", False)
        }
    })

    return result["response"], history, result