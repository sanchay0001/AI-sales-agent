# AutoStream AI Sales Agent 🎬

A conversational AI sales agent built for AutoStream, a fictional SaaS video editing platform. Built as part of the ServiceHive / Inflx internship assignment.

---

## Features

- 🧠 **Intent Detection** — Classifies user messages into greeting, product inquiry, or high-intent lead
- 📚 **RAG Pipeline** — Answers pricing and policy questions from a local knowledge base
- 🎯 **Lead Capture** — Collects name, email, and platform before triggering mock API
- 🔄 **State Management** — Retains memory across 5-6 conversation turns using LangGraph
- 🆓 **Fully Free Stack** — Uses Groq API (free tier) + local FAISS embeddings

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| LLM | Llama 3.1 8B via Groq API |
| Framework | LangChain + LangGraph |
| Vector Store | FAISS (local) |
| Embeddings | sentence-transformers (local) |
| Knowledge Base | Markdown file |

---

## Project Structure

```
autostream-agent/
├── knowledge_base/
│   └── autostream_kb.md     # Pricing, features, policies
├── agent/
│   ├── __init__.py
│   ├── rag.py               # RAG pipeline (embed + retrieve)
│   ├── intent.py            # Intent classification
│   ├── tools.py             # Mock lead capture tool
│   └── graph.py             # LangGraph state machine
├── main.py                  # Entry point / chat loop
├── .env                     # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the Agent
```bash
python main.py
```

---

## Architecture Explanation

This agent is built using **LangGraph**, a stateful graph framework built on top of LangChain. LangGraph was chosen over AutoGen because it provides explicit, deterministic control over conversation flow through a directed state graph — making it easier to manage multi-turn lead collection logic without unpredictable agent loops.

The agent state is a typed dictionary (`AgentState`) that persists across every turn of the conversation. It stores the full conversation history, detected intent, RAG-retrieved context, collected lead fields (name, email, platform), and a lead capture flag. This state is passed through the graph on every turn and updated at each node.

The graph has four nodes: `classify_intent` → routes to one of `handle_greeting`, `handle_inquiry`, or `handle_lead`. The RAG pipeline uses FAISS for local vector search with `sentence-transformers` embeddings to retrieve relevant chunks from the knowledge base. Lead collection is handled incrementally — the agent extracts available fields from each message and only triggers `mock_lead_capture()` once all three required fields are confirmed, preventing premature tool execution.

---

## WhatsApp Integration via Webhooks

To deploy this agent on WhatsApp, the following architecture would be used:

1. **Register a WhatsApp Business webhook** using the Meta Cloud API or Twilio WhatsApp API. This webhook receives an HTTP POST request every time a user sends a message.

2. **Build a Flask or FastAPI server** that exposes a `/webhook` endpoint. When a message arrives, the server extracts the sender ID and message text, then calls `run_agent(message, history)`.

3. **Persist conversation history** per user using a database like Redis or PostgreSQL, keyed by the WhatsApp sender ID. This replaces the in-memory history list used in the CLI version.

4. **Send the agent response back** to the user by calling the WhatsApp API's message sending endpoint with the sender ID and response text.

5. **Deploy the server** on a cloud platform like AWS, GCP, or Railway so the webhook URL is publicly accessible and always available.

This approach makes the agent fully production-ready for social-to-lead conversion on WhatsApp.

---

## Example Conversation

```
You: Hi there!
Agent: Hello! Welcome to AutoStream...

You: What is the price of the Pro plan?
Agent: Our Pro plan is $79/month with unlimited videos, 4K resolution...

You: I want to sign up for my YouTube channel
Agent: That's awesome! Could you share your full name?

You: Rahul Sharma
Agent: Thanks! Could you share your email address?

You: rahul@gmail.com
🎯 Lead captured successfully: Rahul Sharma, rahul@gmail.com, YouTube
Agent: 🎉 You're all set! Our team will reach out shortly...
```

---

## Evaluation Checklist

- ✅ Intent detection (greeting / inquiry / high-intent)
- ✅ RAG-powered knowledge retrieval
- ✅ Multi-turn state management
- ✅ Incremental lead field collection
- ✅ Tool called only after all fields collected
- ✅ Clean modular code structure