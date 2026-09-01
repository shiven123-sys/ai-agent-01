# Agentic AI Assistant

An LLM-powered agent with real tool-calling — not a scripted chatbot. The
model decides *when* and *which* tool to call (calculator, weather,
Wikipedia, notes, unit conversion), executes it, and reasons over the
result before replying. Built on the OpenAI-compatible function-calling
API, works with **Groq (free)** or OpenAI.

## Why this is different from a basic chatbot

A basic chatbot just sends your message to an LLM and prints the reply.
This project implements the actual **agent loop**:

```
User message
   → LLM decides: answer directly, OR call one/more tools
   → if tool(s) called: execute locally → feed results back to LLM
   → LLM reasons over tool output → decides: answer, or call more tools
   → repeat until a final answer is produced (bounded by MAX_TOOL_ITERATIONS)
```

This is the same pattern used in production agent frameworks (LangChain
agents, OpenAI's Assistants API, etc.) — implemented here from scratch so
you understand exactly what's happening under the hood.

## Project structure

```
agentic-ai-assistant/
├── agent/
│   ├── config.py       # provider/model/API key config
│   ├── tools.py         # tool implementations + JSON schemas
│   └── agent.py          # the tool-calling loop
├── api/
│   └── app.py             # Flask web server
├── static/
│   └── index.html         # chat UI
├── tests/
│   └── test_tools.py      # pytest suite (no API key required to run)
├── notes_storage/         # where the add_note tool persists notes
├── cli.py                 # terminal chat interface
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free API key (no credit card needed)
This project defaults to **Groq**, which has a generous free tier and runs
open models (Llama 3.3) at very high speed.

1. Go to https://console.groq.com/keys
2. Sign up (free) and create an API key
3. Copy `.env.example` to `.env`
4. Paste your key into `.env`:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```

Prefer OpenAI instead? Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...`
in `.env` — everything else works unchanged, since both use the same
OpenAI-compatible SDK.

### 3. Run it

CLI:
```bash
python cli.py
```

Web UI:
```bash
python api/app.py
# open http://localhost:5000
```

## Example interaction

```
You: What's 15% tip on a $84 bill, and what's the weather in Mumbai right now?
  🔧 used tool: calculator({'expression': '84 * 0.15'})
  🔧 used tool: get_weather({'city': 'Mumbai'})
Assistant: A 15% tip on $84 is $12.60. Right now in Mumbai it's around
31°C with light wind.
```

The model chose two different tools in one turn, on its own, based on the
schemas — nothing in the code hardcodes "if message contains 'weather'".

## Tools included

| Tool | What it does | Needs internet? | Needs API key? |
|---|---|---|---|
| `calculator` | Safe arithmetic (AST-based, blocks code injection) | No | No |
| `get_current_datetime` | Current date/time, optional UTC offset | No | No |
| `get_weather` | Live weather via Open-Meteo | Yes | No |
| `search_wikipedia` | Short Wikipedia summary | Yes | No |
| `add_note` / `list_notes` | Persistent local notes (JSON file) | No | No |
| `convert_units` | km/miles, kg/lb, °C/°F, meters/feet | No | No |

## Running tests

```bash
pytest tests/ -v
```

All 8 tests run without needing an API key — they test the tools and config
logic in isolation. (Testing the agent loop itself requires a real key
since it calls a live LLM.)

## Extending this project

- **Add a new tool**: write the function in `agent/tools.py`, add its JSON
  schema to `TOOL_SCHEMAS`, register it in `TOOL_REGISTRY`. The agent picks
  it up automatically — no changes needed elsewhere.
- **Streaming responses**: swap `client.chat.completions.create(...)` for
  `stream=True` and adapt the Flask route to Server-Sent Events.
- **Multi-user web app**: replace the in-memory `_agents` dict in
  `api/app.py` with a real session store (Redis) for production use.
- **RAG**: add a `search_documents` tool backed by a vector store
  (e.g. FAISS/Chroma) over your own PDFs/docs.
- **Persistent conversation memory**: swap the in-memory `self.history`
  list in `agent.py` for a database-backed store.

## Notes

- The `notes_storage/` folder is created automatically on first use of
  `add_note`.
- `.env` is gitignored — never commit your API key.
