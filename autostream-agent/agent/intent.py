import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0  # Keep it deterministic for classification
)

# Intent classification prompt
INTENT_PROMPT = PromptTemplate(
    input_variables=["message", "history"],
    template="""
You are an intent classifier for AutoStream, a SaaS video editing platform.

Conversation history:
{history}

Latest user message:
{message}

Classify the user's intent into EXACTLY one of these three categories:
1. casual_greeting     - User is just saying hi, hello, how are you, etc.
2. product_inquiry     - User is asking about features, pricing, plans, policies, or general questions
3. high_intent_lead    - User clearly wants to sign up, start a trial, buy a plan, or is ready to proceed

Rules:
- Reply with ONLY the category name, nothing else
- No explanation, no punctuation, just the category name
- If unsure between product_inquiry and high_intent_lead, choose product_inquiry

Category:
"""
)

def detect_intent(message: str, history: str = "") -> str:
    """
    Detect intent from user message.
    Returns one of: casual_greeting, product_inquiry, high_intent_lead
    """
    chain = INTENT_PROMPT | llm
    result = chain.invoke({
        "message": message,
        "history": history
    })

    intent = result.content.strip().lower()

    # Validate output — fallback to product_inquiry if unexpected
    valid_intents = ["casual_greeting", "product_inquiry", "high_intent_lead"]
    if intent not in valid_intents:
        intent = "product_inquiry"

    return intent