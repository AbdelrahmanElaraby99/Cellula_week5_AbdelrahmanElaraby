import os
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Dict
from huggingface_hub import InferenceClient
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
import streamlit as st

################################################## Setup ##################################################
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openrouter_api_key = os.getenv("LLM_ID")
embedding_model_id = "BAAI/bge-code-v1"
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

# ---- Embeddings ----
class HFInferenceEmbeddings(Embeddings):
    def __init__(self, model_name: str, api_key: str):
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model_name = model_name

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            emb = self.client.feature_extraction(text, model=self.model_name)
            if isinstance(emb, list) and isinstance(emb[0], list):
                emb = emb[0]
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text):
        emb = self.client.feature_extraction(text, model=self.model_name)
        if isinstance(emb, list) and isinstance(emb[0], list):
            emb = emb[0]
        return emb

# ---- Load FAISS ----
persist_directory = "./faiss_humaneval"
embedding_function = HFInferenceEmbeddings(model_name=embedding_model_id, api_key=hf_token)
vectorstore = FAISS.load_local(persist_directory, embeddings=embedding_function, allow_dangerous_deserialization=True)

# ---- Initialize LLM ----
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    model="mistralai/mistral-7b-instruct:free",
    temperature=0.25,
    max_tokens=2048
)

################################################## LangGraph ##################################################
class AgentState(TypedDict):
    user_input: str
    intent: Optional[str]
    retrieved_docs: Optional[List[Document]]
    generated_code: Optional[str]
    explanation: Optional[str]
    final_answer: Optional[str]
    memory: List[Dict[str, str]]

# def intent_classifier(state: AgentState) -> AgentState:
#     text = state["user_input"].lower()
#     if "explain" in text or "what does this code" in text:
#         state["intent"] = "explain"
#     elif "generate" in text or "write code" in text:
#         state["intent"] = "generate"
#     else:
#         state["intent"] = "out_of_scope"
#     return state

def intent_classifier(state: AgentState) -> AgentState:
    text = state["user_input"]

    prompt = f"""
Classify the user's intent into one of these three categories:

1. generate — if the user wants new code written or created.
2. explain — if the user wants to understand, summarize, or explain code.
3. out_of_scope — if it's unrelated to coding or neither of the above.

User message:
{text}

Answer with only one word: generate, explain, or out_of_scope.
do not add any extra text or special tokens such as <s>, [/s], [INST], etc.
    """.strip()

    intent = llm.invoke(prompt).content.strip().lower()
    print(f"inside intent_classifier, classified intent: {intent}")
    # Fallback to out_of_scope if model gives something else
    if intent not in ["generate", "explain", "out_of_scope"]:
        intent = "out_of_scope"

    state["intent"] = intent
    print(f"inside intent_classifier, classified intent: {intent}")
    return state


def retriever(state: AgentState) -> AgentState:
    docs = vectorstore.similarity_search(state["user_input"], k=3)
    state["retrieved_docs"] = docs
    return state

def code_generator(state: AgentState) -> AgentState:
    history_text = ""
    for m in state["memory"][-3:]:
        user = m.get("user_input", "")
        answer = m.get("final_answer", "")
        if user:
            history_text += f"\nUser: {user}"
        if answer:
            history_text += f"\nAssistant: {answer}"

    context = "A Retrieved Code Example:\n" + "\n\nA Retrieved Code Example:\n".join(
        d.page_content for d in state.get("retrieved_docs", [])
    )
    
    prompt = f"""
You are a helpful coding assistant.

Here is the recent conversation:
{history_text}

The user now asked:
{state['user_input']}

Use the following retrieved context if relevant:
{context}
Make sure that you always reply with clean text, do not leave answer empty, and don't include special tokens such as <s>, [INST], etc.
Write clear, correct code. make sure to include your code between three backticks like this:
```<your code here>```
""".strip()

    answer = llm.invoke(prompt)
    state["generated_code"] = answer.content
    return state

def code_explainer(state: AgentState) -> AgentState:
    history_text = "\n".join(
        f"User: {m['user_input']}\nAssistant: {m.get('final_answer', '')}"
        for m in state["memory"][-3:]
    )
    code = state["user_input"]
    if "previous code" in code.lower():
        for m in reversed(state["memory"]):
            if m.get("generated_code"):
                code = m["generated_code"]
                break
    prompt = f"""
You are an expert code explainer.
Make sure that you always reply with clean text, do not leave answer empty, and don't include special tokens such as <s>, [INST], etc.


Conversation history:
{history_text}

Explain this code:
{code}
""".strip()
    answer = llm.invoke(prompt)
    state["explanation"] = answer.content
    return state

def out_of_scope_responder(state: AgentState) -> AgentState:
    history_text = "\n".join(
        f"User: {m['user_input']}\nAssistant: {m.get('final_answer', '')}"
        for m in state["memory"][-3:]
    )
    prompt = f"""
You are a helpful coding assistant. but can also answer general questions.
Make sure that you always reply with clean text, do not leave answer empty (even if user said just thank you), and don't include special tokens such as <s>, [/s], [INST], etc.
Here is the recent conversation:
{history_text}

The user now asked:
{state['user_input']}
"""
    answer = llm.invoke(prompt)
    state["final_answer"] = answer.content
    return state

def final_answer_builder(state: AgentState) -> AgentState:
    if state["intent"] == "generate":
        state["final_answer"] = state["generated_code"]
    elif state["intent"] == "explain":
        state["final_answer"] = state["explanation"]
    elif state["intent"] == "out_of_scope":
        state["final_answer"] = state["final_answer"]
    return state

def update_memory(state: AgentState) -> AgentState:
    entry = {
        "user_input": state["user_input"],
        "generated_code": state.get("generated_code", ""),
        "explanation": state.get("explanation", ""),
        "final_answer": state.get("final_answer", "")
    }
    state["memory"].append(entry)
    return state

graph = StateGraph(AgentState)
graph.add_node("intent_classifier", intent_classifier)
graph.add_node("retriever", retriever)
graph.add_node("code_generator", code_generator)
graph.add_node("code_explainer", code_explainer)
graph.add_node("out_of_scope_responder", out_of_scope_responder)
graph.add_node("final_answer_builder", final_answer_builder)
graph.add_node("update_memory", update_memory)
graph.add_edge(START, "intent_classifier")
graph.add_conditional_edges(
    "intent_classifier",
    lambda state: state["intent"],
    {"generate": "retriever", "explain": "code_explainer", "out_of_scope": "out_of_scope_responder"}
)
graph.add_edge("retriever", "code_generator")
graph.add_edge("code_generator", "final_answer_builder")
graph.add_edge("code_explainer", "final_answer_builder")
graph.add_edge("out_of_scope_responder", "final_answer_builder")
graph.add_edge("final_answer_builder", "update_memory")
graph.add_edge("update_memory", END)

coding_assistant = graph.compile()

################################################## Streamlit App ##################################################
# --- Style Config ---
st.markdown(
    """
    <style>
    /* Add blue border around the entire page */
    .stApp {
        border: 10px solid #014BB6;  /* Cobalt blue border */
        border-radius: 0px;        /* Optional rounded corners */
        padding: 10px;              /* Space inside the border */
        margin: 15px;               /* Space outside the border */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page configuration
st.set_page_config(page_title="CodeGen Agent", page_icon="💻", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>💻 Coding Assistant 💻</h1>",
    unsafe_allow_html=True
)

logo = "./cellula.jpg"
# Center content with side columns
col1, col2, col3 = st.columns([1.25, 2, 1])
with col2:
    st.image(logo, width=500)

st.markdown("---")

# Initialize session state
if "state" not in st.session_state:
    st.session_state.state = {
        "user_input": "",
        "intent": None,
        "retrieved_docs": None,
        "generated_code": None,
        "explanation": None,
        "final_answer": None,
        "memory": []
    }

# Display chat history
for m in st.session_state.state["memory"]:
    with st.chat_message("user"):
        st.markdown(m["user_input"])
    with st.chat_message("assistant"):
        st.markdown(m["final_answer"])

# Input box
user_input = st.chat_input("Type your message or code request...")

if user_input:
    st.session_state.state["user_input"] = user_input
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = coding_assistant.invoke(st.session_state.state)
            answer = result["final_answer"].strip()
            st.markdown(answer)

    st.session_state.state = result
