import re
import ast
import operator
import math
from datetime import datetime
import os
import streamlit as st
import groq
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
import wikipedia
import arxiv
wikipedia.set_user_agent("LangchainApp/1.0 (learning project)")
from langchain_classic.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()

# yfinance is optional. If it is not installed, the finance tool is skipped
# and the rest of the app still runs.
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False


# ----------------------------------------------------------------------------
# Tool: Arxiv (HTTPS client avoids HTTP 301 from ArxivAPIWrapper)
# ----------------------------------------------------------------------------
def _arxiv_search(query: str) -> str:
    try:
        client = arxiv.Client()
        results = list(client.results(arxiv.Search(query=query, max_results=2)))
    except Exception as e:
        return f"Arxiv search failed: {e}"
    if not results:
        return "No results found."
    return "\n\n".join([f"Title: {r.title}\nSummary: {r.summary[:500]}" for r in results])

tool_arxiv = Tool(name="arxiv", func=_arxiv_search,
                  description="Search Arxiv for scientific papers on physics, math, CS, biology, finance, and economics.")

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
tool_wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

search = DuckDuckGoSearchRun(name="web_search",
                             api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=3))


# ----------------------------------------------------------------------------
# Tool: Calculator (safe AST evaluator, no eval/exec)
# Supports + - * / // % **, parentheses, and math.* functions/constants.
# ----------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_ALLOWED_NAMES = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}

def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError("operator not allowed")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_OPS.get(type(node.op))
        if op is None:
            raise ValueError("operator not allowed")
        return op(_safe_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES and not callable(_ALLOWED_NAMES[node.id]):
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_NAMES:
            raise ValueError("unknown function")
        return _ALLOWED_NAMES[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("unsupported expression")

def _calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_safe_eval(tree))
    except Exception as e:
        return f"Calculation error: {e}"

tool_calculator = Tool(
    name="calculator",
    func=_calculator,
    description=("Evaluate a math expression and return the exact result. "
                 "Use this for ANY arithmetic instead of computing it yourself. "
                 "Input is a single expression, e.g. '2**10 / 7' or 'sqrt(144) + log(e)'."),
)


# ----------------------------------------------------------------------------
# Tool: Current date / time
# ----------------------------------------------------------------------------
def _current_datetime(_: str = "") -> str:
    now = datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %H:%M:%S %Z")

tool_datetime = Tool(
    name="current_datetime",
    func=_current_datetime,
    description=("Return the current local date and time. Use this whenever the "
                 "question depends on 'today', 'now', the current year, or the day of week."),
)


# ----------------------------------------------------------------------------
# Tool: Stock price lookup (optional, needs yfinance)
# ----------------------------------------------------------------------------
def _stock_lookup(ticker: str) -> str:
    if not _HAS_YF:
        return "Finance tool unavailable: yfinance is not installed."
    try:
        symbol = ticker.strip().upper()
        hist = yf.Ticker(symbol).history(period="1d")
        if hist.empty:
            return f"No price data found for '{symbol}'. Check the ticker symbol."
        last = hist["Close"].iloc[-1]
        return f"{symbol} latest close: {last:.2f} (most recent trading day)."
    except Exception as e:
        return f"Stock lookup failed: {e}"

tool_finance = Tool(
    name="stock_price",
    func=_stock_lookup,
    description=("Look up the latest closing stock price for a ticker symbol "
                 "(e.g. 'AAPL', 'MSFT', 'TSLA'). Input is the ticker symbol only."),
)


def render_latex(text: str) -> str:
    """Normalize LaTeX so Streamlit's markdown renders it as math.

    Streamlit renders math only with $...$ (inline) and $$...$$ (block).
    LLMs often emit \\[...\\] / \\(...\\) delimiters instead, so convert them.
    """
    # Block math: \[ ... \]  ->  $$ ... $$
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    # Inline math: \( ... \)  ->  $ ... $
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text


def build_chat_history(messages):
    """Convert Streamlit session messages into LangChain message objects."""
    history = []
    for msg in messages:
        content = msg["content"]
        if msg["role"] == "user":
            history.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=content))
    return history


def _preset_groq_key() -> str:
    """Optional preset key for deployment.

    If the deployer sets GROQ_API_KEY in Streamlit secrets (or the environment),
    it pre-fills the sidebar so visitors don't need their own key. Visitors can
    still type their own to override it.
    """
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


st.title("Langchain Agents with Groq API")


# Siderbar for API key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password",
                                value=_preset_groq_key())

# Assemble the toolset. Finance is added only if yfinance is available.
tools = [tool_arxiv, tool_wiki, search, tool_calculator, tool_datetime]
if _HAS_YF:
    tools.append(tool_finance)

st.sidebar.markdown("**Available tools**")
st.sidebar.markdown("\n".join(f"- {t.name}" for t in tools))

# Greeting flag is kept separate so it never leaks into agent memory
GREETING = "I am a chatbot who can surf the web, do math, check the date, and look up stocks. How can I help? "

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": GREETING}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if not api_key:
    st.info("Please enter your Groq API key in the sidebar to continue.")
    st.stop()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You can search the web, Wikipedia, and Arxiv, "
               "evaluate math with the calculator tool, check the current date/time, and look "
               "up stock prices. Always use the calculator tool for arithmetic instead of doing "
               "it yourself, and use current_datetime for anything that depends on today's date. "
               "Use the prior conversation to resolve follow-up questions and references. "
               "When writing mathematical formulas, always use Markdown/LaTeX with $...$ for "
               "inline math and $$...$$ for block equations (never \\[ \\] or \\( \\) delimiters)."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def run_agent(prompt, api_key, callbacks, streaming, chat_history=None):
    """Build and run the agent. Returns the output string."""
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key, streaming=streaming)
    agent = create_openai_tools_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                                   handle_parsing_errors=True, max_iterations=30)
    return agent_executor.invoke(
        {"input": prompt, "chat_history": chat_history or []},
        config={"callbacks": callbacks},
    )["output"]


if prompt := st.chat_input(placeholder="Ask me anything..."):
    # Capture history from prior turns BEFORE adding the new user message,
    # and drop the initial canned greeting so it is not treated as context.
    prior = [m for m in st.session_state.messages if m["content"] != GREETING]
    chat_history = build_chat_history(prior)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = run_agent(prompt, api_key, [st_cb], streaming=True,
                                 chat_history=chat_history)
        except groq.APIError:
            # gpt-oss sometimes hallucinates a tool name while streaming, which
            # Groq rejects. Retry once without streaming, which is more reliable.
            response = run_agent(prompt, api_key, [st_cb], streaming=False,
                                 chat_history=chat_history)
        response = render_latex(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(response)
