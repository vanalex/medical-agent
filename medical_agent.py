"""
medical_agent.py
----------------
Decision agent using LangGraph to search medical information on PubMed or Tavily
depending on the user query type.
"""

from langchain.chat_models import init_chat_model
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# Optional checkpoint memory
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

# === Initialize model and tools ===
llm = init_chat_model("gpt-5-mini")  # or another OpenAI-compatible model
memory = MemorySaver()

pubmed = PubMedAPIWrapper(top_k_results=5)
tavily = TavilySearch(max_results=5)


# === Define state ===
class MessagesState(TypedDict):
    query: str
    classification: str
    source: str
    results: str
    needs_refine: bool
    answer: str


# === Define graph node functions ===
def classify_query(state: MessagesState):
    """Classify query as 'research' (PubMed) or 'general' (Tavily)."""
    query = state["query"]
    prompt = f"""
    You are deciding how to search for medical information.

    Classify this query:
    "{query}"

    Output exactly one label:
    - 'research' → if it asks about studies, evidence, molecules, or clinical trials.
    - 'general' → if it asks about treatment options, symptoms, or patient-friendly info.
    """
    resp = llm.invoke([{"role": "user", "content": prompt}])
    label = resp.content.strip().lower()
    state["classification"] = "research" if "research" in label else "general"
    return state


def perform_search(state: MessagesState):
    """Perform PubMed or Tavily search depending on classification."""
    query = state["query"]
    if state["classification"] == "research":
        print("[Agent] Using PubMed for scholarly search...")
        results = pubmed.run(query)
        state["source"] = "PubMed"
    else:
        print("[Agent] Using Tavily for general medical search...")
        results = tavily.invoke({"query": query})
        state["source"] = "Tavily"

    state["results"] = str(results)
    return state


def check_quality(state: MessagesState):
    """Check if retrieved results are sufficient."""
    results = state.get("results", "")
    if not results or len(results) < 100:  # small heuristic threshold
        state["needs_refine"] = True
    else:
        state["needs_refine"] = False
    return state


def refine_query(state: MessagesState):
    """Refine query if results insufficient."""
    if not state.get("needs_refine"):
        return state

    query = state["query"]
    prompt = f"""
    The previous search for "{query}" returned limited results.
    Suggest a more specific or alternative query that could yield better results.
    """
    resp = llm.invoke([{"role": "user", "content": prompt}])
    new_query = resp.content.strip()
    print(f"[Agent] Refining query → {new_query}")
    state["query"] = new_query
    return state


def summarize_results(state: MessagesState):
    """Summarize retrieved results with citations."""
    query = state["query"]
    results = state.get("results", "")
    source = state.get("source", "")

    prompt = f"""
    You are a medical information assistant.

    Summarize the following search results for the query:
    "{query}"

    Use professional, factual tone and include citations (PMID or URLs).
    Always include a disclaimer that this is not professional medical advice.

    Results ({source}):
    {results}
    """
    resp = llm.invoke([{"role": "user", "content": prompt}])
    state["answer"] = resp.content.strip()
    return state


# === Build LangGraph ===
def should_refine(state: MessagesState) -> str:
    """Conditional edge function."""
    return "refine" if state.get("needs_refine", False) else "summarize"


workflow = StateGraph(MessagesState)

workflow.add_node("classify", classify_query)
workflow.add_node("search", perform_search)
workflow.add_node("check_quality", check_quality)
workflow.add_node("refine", refine_query)
workflow.add_node("summarize", summarize_results)

# edges
workflow.set_entry_point("classify")
workflow.add_edge("classify", "search")
workflow.add_edge("search", "check_quality")
workflow.add_conditional_edges("check_quality", should_refine)
workflow.add_edge("refine", "search")  # loop back for refined query
workflow.add_edge("summarize", "__end__")

graph = workflow.compile(checkpointer=memory)

# === Entrypoint ===
def run_agent(user_query: str):
    initial_state = {"query": user_query}
    final_state = graph.invoke(initial_state, config={"configurable": {"thread_id": "1"}})
    return final_state["answer"]


if __name__ == "__main__":
    print("=== Medical Decision Agent ===")
    user_q = input("Enter your medical question: ").strip()
    answer = run_agent(user_q)
    print("\n--- Answer ---\n")
    print(answer)
