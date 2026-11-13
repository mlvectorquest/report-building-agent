# Report Building Agent - Documentation

## Overview

This is a multi-agent document processing system built with LangChain and LangGraph. It classifies user intents and routes requests to specialized agents (Q&A, summarization, calculation).

## Architecture

```
User Input
    ↓
classify_intent (determine what type of request)
    ↓
Route to appropriate agent:
  • qa_agent (answer questions)
  • summarization_agent (summarize documents)
  • calculation_agent (perform calculations)
    ↓
update_memory (save conversation context)
    ↓
Response returned with sources, tools used, and confidence
```

## Implementation Decisions

### 1. Why LangGraph StateGraph?
- **Declarative workflow**: Graph structure is explicit and visualizable
- **State persistence**: MemorySaver checkpointer remembers past conversations
- **Conditional routing**: Routes to different agents based on intent type
- **Automatic message accumulation**: `add_messages` reducer combines conversation history

### 2. Why Pydantic Schemas?
- **Type safety**: LLM responses are guaranteed to match expected format
- **Automatic validation**: `field_validator` enforces rules (e.g., confidence 0-1)
- **No manual parsing**: No regex or string extraction needed
- **JSON serialization**: Schemas convert cleanly to JSON for logging/storage

### 3. Why Intent Classification Node?
- **Routing accuracy**: System knows which agent to use before executing
- **Confidence scoring**: Tracks certainty in classification
- **Auditability**: Records what the system thought the user wanted

### 4. Why MemorySaver Checkpointer?
- **Multi-turn conversations**: State persists across messages
- **Session isolation**: Different users don't interfere with each other
- **Simple in-memory**: No database setup needed for development

## How State and Memory Works

### AgentState Structure

```python
class AgentState(TypedDict):
    user_input: str                                    # Current user message
    messages: Annotated[List[BaseMessage], add_messages]  # Full conversation history
    intent: UserIntent                                 # Classified intent
    next_step: str                                     # Which node runs next
    conversation_summary: str                          # Summary of conversation
    active_documents: List[str]                        # Docs being discussed
    current_response: Dict                             # Agent's response
    tools_used: List[str]                              # Tools called this turn
    session_id: str                                    # Multi-turn session ID
    user_id: str                                       # User identifier
    actions_taken: Annotated[List[str], operator.add] # Nodes executed
```

### Message Accumulation with `add_messages` Reducer

Without reducer, you'd manually append messages:
```python
state["messages"] = state.get("messages", []) + [new_message]
```

With reducer, it happens automatically:
```python
messages: Annotated[List[BaseMessage], add_messages]
return {"messages": [new_message]}  # Automatically merged!
```

The reducer merges new messages into existing history.

### Action Tracking with `operator.add` Reducer

Without reducer:
```python
return {"actions_taken": ["qa_agent"]}  # Previous actions lost!
```

With reducer:
```python
actions_taken: Annotated[List[str], operator.add]
return {"actions_taken": ["qa_agent"]}  # Automatically accumulated!
```

Result: `["classify_intent", "qa_agent", "update_memory"]`

### State Persistence Flow

**Turn 1:**
1. User sends message
2. Workflow runs: classify_intent → qa_agent → update_memory → END
3. MemorySaver saves final state with thread_id = session_id
4. Response returned to user

**Turn 2 (same session):**
1. User sends new message with same session_id
2. MemorySaver loads previous state automatically
3. New user message added to existing message history
4. Workflow runs with full conversation context
5. Final state saved again

Result: Multi-turn conversation with full memory across turns.

## How Structured Outputs Are Enforced

### Pydantic Schema Validation

```python
class UserIntent(BaseModel):
    intent_type: Literal["qa", "summarization", "calculation", "unknown"]
    confidence: float  # Validated 0-1
    reasoning: str
```

**Validation happens automatically:**

1. **Type checking**: confidence must be a number
2. **Field constraints**: confidence must be between 0 and 1 (enforced by field_validator)
3. **Literal enforcement**: intent_type must be exactly one of the four allowed values
4. **Required fields**: All fields without defaults must be provided

**Example:**
```python
# INVALID - confidence > 1.0
UserIntent(intent_type="qa", confidence=1.5, reasoning="...")
# Raises: ValidationError: confidence must be less than or equal to 1

# VALID - all constraints met
UserIntent(intent_type="qa", confidence=0.95, reasoning="Clear question pattern")
# Success ✓
```

### Integration with LLM

```python
# Tell LLM to return structured output
structured_llm = llm.with_structured_output(UserIntent)

# LangChain injects schema into prompt
# LLM returns JSON matching the schema
# Pydantic validates and returns typed object
intent: UserIntent = structured_llm.invoke(prompt)

# Type-safe access
print(intent.intent_type)   # "qa"
print(intent.confidence)     # 0.95
```

### All Schemas with Validation

- **UserIntent**: intent_type (Literal), confidence (0-1), reasoning (string)
- **AnswerResponse**: question, answer, sources (list), confidence (0-1), timestamp (auto)
- **SummarizationResponse**: summary, key_points, document_ids
- **CalculationResponse**: expression, result, explanation, units
- **UpdateMemoryResponse**: summary, document_ids

All use Pydantic `field_validator` to enforce confidence bounds and type constraints.

## Example Conversations

### Example 1: Q&A Intent

```
USER: "What is the total amount in invoice INV-001?"

SYSTEM PROCESSING:
  classify_intent node:
    • Analyzes: "total" + "invoice" keywords → calculation query
    • Returns: intent_type="calculation", confidence=0.95
    • Sets next_step="calculation_agent"
  
  calculation_agent node:
    • Gets CALCULATION_SYSTEM_PROMPT ("Always use calculator tool...")
    • Uses document_reader("INV-001") tool to get invoice details
    • Uses calculator() tool to sum the amounts
    • LLM generates CalculationResponse with confidence 0.95
    • Returns: sources=["INV-001"], result=$22,000
  
  update_memory node:
    • Summarizes: "User asked for total in invoice INV-001, calculated $22,000"
    • Tracks: active_documents=["INV-001"]

RESPONSE:

{
  "success": true,
  "response": "The total amount in invoice INV-001 is $22,000.",
  "intent": {
    "intent_type": "calculation",
    "confidence": 0.95,
    "reasoning": "The user's input asks for the total amount in a specific invoice, which implies a need to retrieve or compute a numerical value. This aligns with the 'calculation' category, as it involves determining a total amount, potentially requiring arithmetic operations or accessing numerical data from the document."
  },
  "tools_used": ["document_reader", "calculator"],
  "sources": ["INV-001"],
  "actions_taken": ["classify_intent", "calculation_agent", "update_memory"],
  "summary": "The user inquired about the total amount in invoice INV-001. The invoice details were retrieved, showing a subtotal of $20,000 and a tax of $2,000. The total amount calculated was $22,000."
}


| Field | Value |
|-------|-------|
| **Success** | ✅ true |
| **Response** | The total amount in invoice INV-001 is $22,000. |
| **Intent Type** | calculation |
| **Confidence** | 95% |
| **Tools Used** | document_reader, calculator |
| **Sources** | INV-001 |
| **Actions Taken** | classify_intent → calculation_agent → update_memory |
| **Summary** | Successfully calculated invoice total ($20,000 subtotal + $2,000 tax = $22,000) |
```

### Example 2: Summarization Intent

```
USER: "Summarize all the contracts"

SYSTEM PROCESSING (Turn 2 - Multi-turn Context):
  classify_intent node:
    • Analyzes: "Summarize" keyword
    • Returns: intent_type="summarization", confidence=0.95
    • Sets next_step="summarization_agent"
  
  summarization_agent node:
    • Gets SUMMARIZATION_SYSTEM_PROMPT ("You are a summarization expert...")
    • Uses document_search("contracts") to find docs
    • Uses document_reader("CON-001") to get contract details
    • LLM generates SummarizationResponse with structured summary
    • Returns: sources=["CON-001"], comprehensive summary
  
  update_memory node:
    • Accumulates: active_documents=["INV-001", "CON-001"] (from both turns)
    • Tracks conversation spanning both Q&A and summarization

RESPONSE:

{
  "success": true,
  "response": "### Contract Summary\n\n**Document ID:** CON-001  \n**Title:** Service Agreement  \n**Date:** January 1, 2024  \n**Parties Involved:**  \n- **Provider:** DocDacity Solutions Inc.  \n- **Client:** Healthcare Partners LLC  \n\n**Services Provided:**  \n1. Document Processing Platform Access  \n2. 24/7 Technical Support  \n3. Monthly Data Analytics Reports  \n4. Compliance Monitoring  \n\n**Financial Details:**  \n- **Monthly Fee:** $15,000  \n- **Total Contract Value:** $180,000  \n\n**Duration:** 12 months  \n\n**Termination Clause:** Either party may terminate the agreement with 60 days written notice.  \n\nThis contract outlines the services and financial terms agreed upon between DocDacity Solutions Inc. and Healthcare Partners LLC for a duration of one year.",
  "intent": {
    "intent_type": "summarization",
    "confidence": 0.95,
    "reasoning": "The user's request to 'Summarize all the contracts' clearly falls under the 'summarization' category. They are asking for a condensed version or key points from the contracts, which does not involve any calculations or numerical analysis. The conversation history does not indicate any need for calculations related to this request, further supporting the classification as summarization."
  },
  "tools_used": ["document_reader", "calculator", "document_search", "document_reader"],
  "sources": ["INV-001", "CON-001"],
  "actions_taken": ["classify_intent", "calculation_agent", "update_memory", "classify_intent", "summarization_agent", "update_memory"],
  "summary": "The conversation involved calculating the total amount in invoice INV-001, which included consulting services, software development, and support & maintenance, totaling $20,000 with an additional 10% tax of $2,000, resulting in a total of $22,000. Additionally, a service agreement contract (CON-001) was summarized, detailing a 12-month agreement between DocDacity Solutions Inc. and Healthcare Partners LLC for document processing services, with a total contract value of $180,000 and a monthly fee of $15,000."
}

**CLI Output Display:**

🤖 Assistant: ### Contract Summary

**Document ID:** CON-001  
**Title:** Service Agreement  
**Date:** January 1, 2024  
**Parties Involved:**  
- **Provider:** DocDacity Solutions Inc.  
- **Client:** Healthcare Partners LLC  

**Services Provided:**  
1. Document Processing Platform Access  
2. 24/7 Technical Support  
3. Monthly Data Analytics Reports  
4. Compliance Monitoring  

**Financial Details:**  
- **Monthly Fee:** $15,000  
- **Total Contract Value:** $180,000  

**Duration:** 12 months  

**Termination Clause:** Either party may terminate the agreement with 60 days written notice.  

This contract outlines the services and financial terms agreed upon between DocDacity Solutions Inc. and Healthcare Partners LLC for a duration of one year.

INTENT: summarization

TOOLS USED: document_reader, calculator, document_search, document_reader

CONVERSATION SUMMARY: The conversation involved calculating the total amount in invoice INV-001, which included consulting services, software development, and support & maintenance, totaling $20,000 with an additional 10% tax of $2,000, resulting in a total of $22,000. Additionally, a service agreement contract (CON-001) was summarized, detailing a 12-month agreement between DocDacity Solutions Inc. and Healthcare Partners LLC for document processing services, with a total contract value of $180,000 and a monthly fee of $15,000.

| Field | Value |
|-------|-------|
| **Success** | ✅ true |
| **Intent Type** | summarization |
| **Confidence** | 95% |
| **Tools Used** | document_reader, calculator, document_search |
| **Sources** | INV-001, CON-001 |
| **Actions Taken** | classify_intent → calculation_agent → update_memory → classify_intent → summarization_agent → update_memory |
| **Context** | Multi-turn conversation preserving state across both queries |
```

### Example 3: Calculation Intent

```
USER: "Calculate the sum of invoices over $50,000"

SYSTEM PROCESSING:
  classify_intent node:
    • Analyzes: "Calculate" + "sum" keywords
    • Returns: intent_type="calculation", confidence=0.96
    • Sets next_step="calculation_agent"
  
  calculation_agent node:
    • Gets CALCULATION_SYSTEM_PROMPT ("Always use calculator tool...")
    • Uses document_search(query, comparison="over", amount=50000)
    • Finds: INV-002 ($69,300), INV-003 ($214,500)
    • Uses calculator("69300 + 214500") tool
    • LLM generates CalculationResponse
    • Returns: result=283800, units="USD"
  
  update_memory node:
    • Summarizes: "User calculated sum of high-value invoices"
    • Tracks: active_documents=["INV-002", "INV-003"]
  
  Tool usage logged to ./logs/tool_usage_*.json with:
    - tool_name: "calculator"
    - input: "69300 + 214500"
    - output: "Result: 283800"

RESPONSE:
{
  "success": true,
  "response": "The sum of invoices over $50,000 is $283,800.",
  "intent": {
    "intent_type": "calculation",
    "confidence": 0.95,
    "reasoning": "The user explicitly requests a calculation ('Calculate the sum of invoices over $50,000'), indicating a need for numerical computation. The context of the conversation history also supports this, as it involves financial document analysis and previous calculation tasks."
  },
  "tools_used": ["document_reader", "calculator", "document_search", "document_reader", "document_search", "document_reader", "document_reader", "calculator"],
  "sources": ["INV-001", "CON-001", "INV-003", "INV-002"],
  "actions_taken": ["classify_intent", "calculation_agent", "update_memory", "classify_intent", "summarization_agent", "update_memory", "classify_intent", "calculation_agent", "update_memory"],
  "summary": "The user requested the total amount for invoice INV-001, which was calculated to be $22,000. Subsequently, the user asked for a summary of all contracts, leading to the retrieval and summarization of a service agreement (CON-001) between DocDacity Solutions Inc. and Healthcare Partners LLC, valued at $180,000 for a 12-month duration. Finally, the user inquired about the sum of invoices over $50,000, resulting in a total of $283,800 from invoices INV-003 and INV-002."
}

**CLI Output Display:**

🤖 Assistant: The sum of invoices over $50,000 is $283,800.

INTENT: calculation

TOOLS USED: document_reader, calculator, document_search, document_reader, document_search, document_reader, document_reader, calculator

CONVERSATION SUMMARY: The user requested the total amount for invoice INV-001, which was calculated to be $22,000. Subsequently, the user asked for a summary of all contracts, leading to the retrieval and summarization of a service agreement (CON-001) between DocDacity Solutions Inc. and Healthcare Partners LLC, valued at $180,000 for a 12-month duration. Finally, the user inquired about the sum of invoices over $50,000, resulting in a total of $283,800 from invoices INV-003 and INV-002.

| Field | Value |
|-------|-------|
| **Success** | ✅ true |
| **Response** | The sum of invoices over $50,000 is $283,800. |
| **Intent Type** | calculation |
| **Confidence** | 95% |
| **Tools Used** | document_reader, calculator, document_search |
| **Sources** | INV-001, CON-001, INV-003, INV-002 |
| **Actions Taken** | classify_intent → calculation_agent → update_memory → classify_intent → summarization_agent → update_memory → classify_intent → calculation_agent → update_memory |
| **Context** | Multi-turn conversation: INV-001 query + CON-001 summary + invoices >$50k calculation |
```

### Example 4: Multi-Turn Conversation (Same Session)

```
TURN 1:
USER: "What is invoice INV-001?"

SYSTEM PROCESSING:
  classify_intent node:
    • Analyzes: "What is" + "invoice" keywords → qa query
    • Returns: intent_type="qa", confidence=0.85
    • Sets next_step="qa_agent"
  
  qa_agent node:
    • Gets QA_SYSTEM_PROMPT
    • Uses document_reader("INV-001") tool to get invoice details
    • LLM generates response with invoice information
    • Returns: sources=["INV-001"], detailed invoice breakdown
  
  update_memory node:
    • Summarizes: "User asked for details on invoice INV-001"
    • Tracks: active_documents=["INV-001"]

RESPONSE:

{
  "success": true,
  "response": "Invoice INV-001 includes the following details:\n\n- **Invoice Number:** 12345\n- **Date:** January 15, 2024\n- **Client:** Acme Corporation\n\n**Services Rendered:**\n- Consulting Services: $5,000\n- Software Development: $12,500\n- Support & Maintenance: $2,500\n\n**Financial Summary:**\n- **Subtotal:** $20,000\n- **Tax (10%):** $2,000\n- **Total Amount:** $22,000\n\n**Payment Terms:** Net 30 days\n\nThis invoice outlines the services provided to Acme Corporation, along with the associated costs and payment terms.",
  "intent": {
    "intent_type": "qa",
    "confidence": 0.85,
    "reasoning": "The user is asking for information about a specific invoice (INV-001). This request does not involve any calculations or require summarization of a document. It is a straightforward question about a document, which aligns with the 'qa' category."
  },
  "tools_used": ["document_reader"],
  "sources": ["INV-001"],
  "actions_taken": ["classify_intent", "qa_agent", "update_memory"],
  "summary": "Invoice INV-001 includes the following details:\n\n- **Invoice Number:** 12345\n- **Date:** January 15, 2024\n- **Client:** Acme Corporation\n\n**Services Rendered:**\n- Consulting Services: $5,000\n- Software Development: $12,500\n- Support & Maintenance: $2,500\n\n**Financial Summary:**\n- **Subtotal:** $20,000\n- **Tax (10%):** $2,000\n- **Total Amount:** $22,000\n\n**Payment Terms:** Net 30 days\n\nThis invoice outlines the services provided to Acme Corporation, along with the associated costs and payment terms."
}

**CLI Output Display:**

🤖 Assistant: Invoice INV-001 includes the following details:

- **Invoice Number:** 12345
- **Date:** January 15, 2024
- **Client:** Acme Corporation

**Services Rendered:**
- Consulting Services: $5,000
- Software Development: $12,500
- Support & Maintenance: $2,500

**Financial Summary:**
- **Subtotal:** $20,000
- **Tax (10%):** $2,000
- **Total Amount:** $22,000

**Payment Terms:** Net 30 days

This invoice outlines the services provided to Acme Corporation, along with the associated costs and payment terms.

INTENT: qa

TOOLS USED: document_reader

CONVERSATION SUMMARY: Invoice INV-001 includes the following details:

- **Invoice Number:** 12345
- **Date:** January 15, 2024
- **Client:** Acme Corporation

**Services Rendered:**
- Consulting Services: $5,000
- Software Development: $12,500
- Support & Maintenance: $2,500

**Financial Summary:**
- **Subtotal:** $20,000
- **Tax (10%):** $2,000
- **Total Amount:** $22,000

**Payment Terms:** Net 30 days

This invoice outlines the services provided to Acme Corporation, along with the associated costs and payment terms.

→ System returns invoice details
→ Saves session state with thread_id=session-123

TURN 2:
USER: "Summarize that invoice"

SYSTEM PROCESSING (Multi-turn Context):
  MemorySaver loads previous state with thread_id=session-123
  Message history now includes Turn 1 messages
  System knows "that invoice" refers to INV-001 from context
  
  classify_intent node:
    • Analyzes: "Summarize" keyword from previous context
    • Returns: intent_type="summarization", confidence=0.95
    • Sets next_step="summarization_agent"
  
  summarization_agent node:
    • Gets SUMMARIZATION_SYSTEM_PROMPT
    • Uses document_reader("INV-001") to get invoice details (from memory)
    • LLM generates SummarizationResponse with invoice summary
    • Returns: sources=["INV-001"], structured summary
  
  update_memory node:
    • Accumulates: active_documents=["INV-001"] (same document from Turn 1)
    • Tracks conversation history across both turns
    • Conversation memory preserved!

RESPONSE:

{
  "success": true,
  "response": "### Invoice Summary for INV-001\n\n**Invoice Number:** 12345  \n**Date Issued:** January 15, 2024  \n**Client:** Acme Corporation  \n\n**Services Provided:**\n- **Consulting Services:** $5,000\n- **Software Development:** $12,500\n- **Support & Maintenance:** $2,500\n\n**Financial Details:**\n- **Subtotal:** $20,000\n- **Tax (10%):** $2,000\n- **Total Amount Due:** $22,000\n\n**Payment Terms:** Net 30 days\n\nThis invoice details the services rendered to Acme Corporation, including consulting, software development, and support, with a total amount due of $22,000.",
  "intent": {
    "intent_type": "summarization",
    "confidence": 0.95,
    "reasoning": "The user explicitly requested a summary of an invoice, which aligns with the 'summarization' intent. There is no indication of a need for calculations or questions about the document's content, which would fall under 'qa' or 'calculation' intents. The request is clear and does not fall into the 'unknown' category."
  },
  "tools_used": ["document_reader", "calculator", "document_search", "document_reader", "document_search", "document_reader", "document_reader", "calculator"],
  "sources": ["INV-001"],
  "actions_taken": ["classify_intent", "calculation_agent", "update_memory", "classify_intent", "summarization_agent", "update_memory", "classify_intent", "calculation_agent", "update_memory", "classify_intent", "qa_agent", "update_memory", "classify_intent", "summarization_agent", "update_memory"],
  "summary": "Invoice INV-001 includes services provided to Acme Corporation, such as consulting services ($5,000), software development ($12,500), and support & maintenance ($2,500). The subtotal is $20,000, with a 10% tax amounting to $2,000, bringing the total to $22,000. Payment terms are net 30 days."
}

**CLI Output Display:**

🤖 Assistant: ### Invoice Summary for INV-001

**Invoice Number:** 12345  
**Date Issued:** January 15, 2024  
**Client:** Acme Corporation  

**Services Provided:**
- **Consulting Services:** $5,000
- **Software Development:** $12,500
- **Support & Maintenance:** $2,500

**Financial Details:**
- **Subtotal:** $20,000
- **Tax (10%):** $2,000
- **Total Amount Due:** $22,000

**Payment Terms:** Net 30 days

This invoice details the services rendered to Acme Corporation, including consulting, software development, and support, with a total amount due of $22,000.

INTENT: summarization

TOOLS USED: document_reader, calculator, document_search, document_reader, document_search, document_reader, document_reader, calculator

CONVERSATION SUMMARY: Invoice INV-001 includes services provided to Acme Corporation, such as consulting services ($5,000), software development ($12,500), and support & maintenance ($2,500). The subtotal is $20,000, with a 10% tax amounting to $2,000, bringing the total to $22,000. Payment terms are net 30 days.

→ MemorySaver saves updated state with full 2-turn conversation history

TURN 3:
USER: "Calculate total of all documents we discussed"

SYSTEM PROCESSING (Multi-turn Context with Full Memory):
  MemorySaver loads previous state with full 2-turn history
  Message history includes Turns 1 and 2 messages
  active_documents = ["INV-001"] from previous context
  
  classify_intent node:
    • Analyzes: "Calculate" + "total" + "all documents" keywords
    • Returns: intent_type="calculation", confidence=0.95
    • Sets next_step="calculation_agent"
  
  calculation_agent node:
    • Gets CALCULATION_SYSTEM_PROMPT
    • Uses document_reader to access INV-001, INV-002, INV-003, CON-001
    • Uses calculator to sum: $22,000 + $69,300 + $214,500 + $180,000
    • LLM generates CalculationResponse with final total
    • Returns: result=$485,800, all 4 documents as sources
  
  update_memory node:
    • Final summary: "User calculated total of all discussed documents"
    • Tracks: active_documents=["INV-001", "INV-003", "INV-002", "CON-001"]
    • Full conversation state saved with complete 3-turn history

RESPONSE:

{
  "success": true,
  "response": "The total of all the documents we discussed is $485,800.",
  "intent": {
    "intent_type": "calculation",
    "confidence": 0.95,
    "reasoning": "The user's request 'Calculate total of all documents we discussed' explicitly asks for a calculation of totals from previously discussed documents. The context of the conversation history shows multiple instances where calculations were performed on document data, reinforcing the user's intent to perform a mathematical operation. The use of the word 'calculate' and the context of discussing financial documents strongly indicate a calculation intent."
  },
  "tools_used": ["document_reader", "calculator", "document_search", "document_reader", "document_search", "document_reader", "document_reader", "calculator", "calculator"],
  "sources": ["INV-001", "INV-003", "INV-002", "CON-001"],
  "actions_taken": ["classify_intent", "calculation_agent", "update_memory", "classify_intent", "summarization_agent", "update_memory", "classify_intent", "calculation_agent", "update_memory", "classify_intent", "qa_agent", "update_memory", "classify_intent", "summarization_agent", "update_memory", "classify_intent", "calculation_agent", "update_memory"],
  "summary": "The conversation involved calculating the total amounts from various financial documents, including invoices and a service agreement. Key documents referenced were:\n- Invoice INV-001 with a total amount of $22,000.\n- Invoice INV-003 with a total amount of $214,500.\n- Invoice INV-002 with a total amount of $69,300.\n- Service Agreement CON-001 with a total contract value of $180,000.\n\nThe sum of invoices over $50,000 was calculated to be $283,800. The total of all discussed documents was calculated to be $485,800."
}

**CLI Output Display:**

🤖 Assistant: The total of all the documents we discussed is $485,800.

INTENT: calculation

TOOLS USED: document_reader, calculator, document_search, document_reader, document_search, document_reader, document_reader, calculator, calculator

CONVERSATION SUMMARY: The conversation involved calculating the total amounts from various financial documents, including invoices and a service agreement. Key documents referenced were:
- Invoice INV-001 with a total amount of $22,000.
- Invoice INV-003 with a total amount of $214,500.
- Invoice INV-002 with a total amount of $69,300.
- Service Agreement CON-001 with a total contract value of $180,000.

The sum of invoices over $50,000 was calculated to be $283,800. The total of all discussed documents was calculated to be $485,800.

→ Final state saved with complete multi-turn conversation history and all 4 documents tracked
```

## Key Features

1. **Intent Classification**: User requests automatically routed to right agent
2. **Structured Outputs**: All LLM responses validated by Pydantic schemas
3. **Tool Logging**: Every tool call logged to `./logs/` directory with timestamp
4. **Session Persistence**: Conversations remembered via `./sessions/` JSON files
5. **State Accumulation**: Actions and messages properly accumulated across turns
6. **Confidence Scoring**: All responses include confidence 0-1 score
7. **Source Tracking**: Q&A responses cite which documents were used
8. **Error Handling**: Safe eval for calculator, graceful error messages

## Files Structure

```
src/
  agent.py        - LangGraph workflow (5 nodes: classify, 3 agents, memory)
  assistant.py    - Session management and orchestration
  schemas.py      - Pydantic models (UserIntent, AnswerResponse, etc.)
  prompts.py      - Prompt templates for each agent type
  tools.py        - Tool implementations (calculator, document search/read)
  retrieval.py    - Document retrieval simulation

./logs/           - Tool usage logs (auto-created)
./sessions/       - Saved conversations (auto-created)
```
