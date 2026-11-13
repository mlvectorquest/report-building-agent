# Report Building Agent (DocDacity Document Assistant)

A multi-agent document assistant built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/). It answers questions about, summarizes, and performs calculations on financial and healthcare documents (invoices, contracts, claims, etc.).

A user message is first classified by intent, then routed to a specialized agent:

| Agent | Purpose |
|-------|---------|
| **Q&A** | Answers specific questions about document content, with source citations |
| **Summarization** | Summarizes documents and extracts key points |
| **Calculation** | Reads document data and performs math using a calculator tool |

After each turn, a memory node updates the conversation summary and the list of active documents, and state persists across turns via a LangGraph checkpointer.

## Architecture

```
User input
   ↓
classify_intent
   ↓
qa_agent | summarization_agent | calculation_agent
   ↓
update_memory
   ↓
END
```

See the diagram at [starter/docs/langgraph_agent_architecture.png](starter/docs/langgraph_agent_architecture.png) and design rationale in [starter/DOCUMENTATION.md](starter/DOCUMENTATION.md).

## Project Structure

```
starter/
├── main.py               # Interactive CLI entry point
├── requirements.txt      # Python dependencies
├── DOCUMENTATION.md      # Implementation decisions and state/memory notes
├── README.md             # Original project/task instructions
├── docs/                 # Architecture diagram
└── src/
    ├── schemas.py        # Pydantic models (AnswerResponse, UserIntent, ...)
    ├── retrieval.py      # Simulated document retriever and sample documents
    ├── tools.py          # Calculator, document search/reader/statistics tools, ToolLogger
    ├── prompts.py        # Intent classification and per-agent system prompts
    ├── agent.py          # AgentState, graph nodes, and workflow creation
    └── assistant.py      # DocumentAssistant: sessions and message processing
```

## Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API key

### Installation

```bash
cd starter
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `starter/` containing your key:

```
OPENAI_API_KEY=your-key-here
```

### Running

```bash
python main.py
```

You will be asked for a user ID (default `demo_user`) and a session is started. Available commands:

- `/help` – show help
- `/docs` – list available documents
- `/quit` – exit

Example queries:

- What's the total amount in invoice INV-001?
- Summarize all contracts
- Calculate the sum of all invoice totals
- Find documents with amounts over $50,000

## Tools

- **calculator** – safely evaluates mathematical expressions
- **document_search** – keyword, type, and amount-based search
- **document_reader** – reads a document by ID
- **document_statistics** – aggregate statistics across documents

All tool calls are logged by `ToolLogger` to `./logs`.

## Built With

- [LangChain](https://python.langchain.com/) – LLM tooling and prompts
- [LangGraph](https://langchain-ai.github.io/langgraph/) – stateful agent workflow and checkpointing
- [OpenAI](https://platform.openai.com/) (`gpt-4o`) – language model
- [Pydantic](https://docs.pydantic.dev/) – structured output schemas
