# Multi-Tool AI Research Agent

An autonomous AI agent built with LangChain that answers questions by deciding, on its own, which tools to call. It searches across Wikipedia, Arxiv, the live web (DuckDuckGo), and a custom RAG knowledge base, then synthesizes an answer. Powered by a Groq-hosted LLM with a Streamlit chat interface.

> **Resume summary:** Built a multi-tool LangChain agent (Groq LLM) that autonomously routes queries across Wikipedia, Arxiv, web search, and a FAISS-backed RAG tool, with tool-call validation retries, LaTeX rendering, and a Streamlit chat UI.

## Links

- **Live app:** https://multi-tool-ai-agent-ankit.streamlit.app/
- **GitHub repo:** https://github.com/Ankitkumar7217734/multi-tool-ai-agent

## Demo

_Add a screenshot or short GIF of the running app here._

```
[ screenshot: agent answering a question and showing which tools it called ]
```

## What it does

Given a question, the agent reasons about which of its tools can help, calls them, and combines the results into a single answer. It can chain multiple tool calls in one turn (for example, search Arxiv for a paper, then Wikipedia for background).

Two ways to run it:

- **Web app (`app.py`)** is a Streamlit chat interface backed by Groq. Uses Wikipedia, Arxiv, and DuckDuckGo as live tools, renders LaTeX math in answers, and auto-retries when the model emits an invalid tool call.
- **Notebook (`tools_agents.ipynb`)** is a step-by-step build of the same agent, including a LangSmith-docs RAG tool (FAISS + OpenAI embeddings) and pulling a prompt from LangSmith Hub.

## Tools the agent can call

| Tool | Source | Purpose |
|------|--------|---------|
| Wikipedia | langchain_community + wikipedia | Encyclopedic background |
| Arxiv | arxiv.Client() (direct HTTPS) | Scientific papers |
| Web search | langchain_community + ddgs | Current / live information |
| Calculator | stdlib (ast + math) | Exact arithmetic, safe (no eval/exec) |
| Current date/time | stdlib (datetime) | Resolves 'today', 'now', current year |
| Stock price | yfinance | Latest closing price for a ticker (optional) |
| LangSmith docs (RAG) | FAISS + OpenAI embeddings | Answers from a private doc set |

## Architecture

```
User question
     |
     v
  Groq LLM (agent)  --picks tool(s)-->  Wikipedia / Arxiv / Web / RAG
     |                                        |
     |<-------------- tool results ------------
     v
 Final synthesized answer  -->  Streamlit UI (LaTeX rendered)
```

## Engineering highlights

These are the real problems solved while building it:

- **Invalid tool-call recovery.** The gpt-oss model occasionally emits a tool name Groq rejects with `tool call validation failed`. The app catches `groq.APIError` and retries once with streaming disabled.
- **Tool naming to reduce hallucination.** The DuckDuckGo tool is registered as `web_search` instead of the default `duckduckgo_search`. The shorter name lowers the chance the model invents an invalid name.
- **Arxiv HTTP 301 fix.** Uses `arxiv.Client()` over HTTPS directly instead of `ArxivAPIWrapper`, which defaulted to HTTP on older versions and failed with redirects. Requires arxiv >= 2.1.0.
- **LaTeX rendering.** The system prompt asks for `$...$` / `$$...$$` delimiters, and a `render_latex()` helper converts `\[...\]` / `\(...\)` to the `$` form so Streamlit renders math correctly.
- **Trustworthy math.** A safe AST-based calculator tool evaluates arithmetic (no `eval`/`exec`), so rendered LaTeX reflects correct numbers instead of the model's mental math.
- **Graceful optional tools.** The stock-price tool (yfinance) is added only if the package is installed; the app runs fine without it.
- **Iteration tuning.** AgentExecutor runs with `max_iterations=30` (default is 15) to handle multi-step questions.

## Tech stack

| Component | Library |
|-----------|---------|
| Agent framework | LangChain 1.3.x + langchain-classic 1.0.7 |
| LLM | Groq (langchain-groq) |
| Embeddings (RAG tool) | OpenAI (langchain-openai) |
| Vector store (RAG tool) | FAISS (faiss-cpu) |
| Tools | langchain-community, wikipedia, arxiv, ddgs |
| UI | Streamlit |
| Tracing | LangSmith |

## Prerequisites

- Python 3.10+
- A Groq API key (https://console.groq.com/)
- An OpenAI API key (https://platform.openai.com/), only for the notebook's RAG tool embeddings
- Optional: a LangSmith API key for tracing

## Setup

```bash
# 1. Navigate into the project folder
cd "Langchain-tool-and-agents "

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install langchain langchain-classic langchain-community langchain-core \
            langchain-groq langchain-openai langchain-text-splitters \
            langchainhub langsmith faiss-cpu streamlit wikipedia \
            arxiv ddgs python-dotenv openai yfinance
```

Create a `.env` file in this folder:

```env
GROQ_API_KEY="your_groq_api_key"
OPENAI_API_KEY="your_openai_api_key"
LANGCHAIN_API_KEY="your_langsmith_api_key"
LANGCHAIN_PROJECT="multi-tool-agent"
```

> Do not commit `.env`. Add it to `.gitignore` before pushing.

## Run

**Web app:**

```bash
streamlit run app.py
```

Then enter your Groq API key in the sidebar and start chatting.

**Notebook:**

```bash
jupyter notebook tools_agents.ipynb
```

Run cells top to bottom: build the Wikipedia and Arxiv tools, create the LangSmith RAG tool, set up the Groq LLM, pull a prompt from LangSmith Hub, then create and run the agent with `create_openai_tools_agent`.

## Project structure

```
app.py              Streamlit agent app
tools_agents.ipynb  Notebook walkthrough (includes RAG tool)
README.md           This file
```

## Notes

- langchain-classic is required for `create_openai_tools_agent` and `AgentExecutor`, which were removed from core langchain in v1.x.
- langchain-community is being sunset and emits deprecation warnings.
- Wikipedia requires a User-Agent header, set automatically via `wikipedia.set_user_agent()`.
# multi-tool-ai-agent
