import os
from bisect import bisect_right

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Youtube RAG Assistant", page_icon="🎥", layout="centered")
load_dotenv()


# ---------------------------------------------------------------------------
# Transcript + chunking
# ---------------------------------------------------------------------------
def extract_video_id(url):
    """Accept watch, youtu.be, Shorts and embed links, and ignore stray spaces."""
    url = url.strip()
    for marker in ("v=", "youtu.be/", "/shorts/", "/embed/"):
        if marker in url:
            return url.split(marker)[1].split("&")[0].split("?")[0].split("/")[0]
    return url


def fetch_transcript(video_id):
    """Return the transcript as one string, plus (character_position, start_seconds) for every caption line."""
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    except TranscriptsDisabled:
        raise Exception("No captions available for this video.")
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {e}")

    offsets, parts, pos = [], [], 0
    for snippet in transcript:
        offsets.append((pos, snippet.start))
        parts.append(snippet.text)
        pos += len(snippet.text) + 1  # +1 for the space added by join
    return " ".join(parts), offsets


def build_chunks(text, offsets):
    """Split the transcript into overlapping chunks and tag each one with its video timestamp."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    chunks = splitter.create_documents([text])

    starts = [pos for pos, _ in offsets]
    for chunk in chunks:
        i = bisect_right(starts, chunk.metadata["start_index"]) - 1
        chunk.metadata["start_sec"] = int(offsets[max(i, 0)][1])
    return chunks


# ---------------------------------------------------------------------------
# RAG chain
# ---------------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag(chunks):
    """Build the retriever + LLM chain. The chain returns {"docs", "question", "answer"}."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5},
    )

    # Model name comes from .env so a retired model never needs a code change
    llm = ChatGroq(model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0.3)

    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer only from the provided transcript context.
If the context is insufficient, just say you do not know the answer.
{context}
Question: {question}
""",
        input_variables=["context", "question"],
    )

    answer_chain = (
        RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )
    chain = (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnablePassthrough.assign(answer=answer_chain)
    )
    return chain, llm


# ---------------------------------------------------------------------------
# Extra features: study material, memory, sources
# ---------------------------------------------------------------------------
def generate_study_material(llm, text):
    """One-time LLM calls on the transcript: summary, study notes, suggested questions."""
    text = text[:8000]  # cap the size to stay inside token limits
    summary = llm.invoke(
        f"Summarize this transcript in 5 short bullet points for a learner:\n\n{text}"
    ).content
    notes = llm.invoke(
        f"Write concise study notes in markdown (headings + bullet points) from this transcript:\n\n{text}"
    ).content
    raw = llm.invoke(
        f"Suggest 4 short questions a learner could ask about this video. One per line, no numbering:\n\n{text}"
    ).content
    questions = [q.strip("-•*0123456789. ").strip() for q in raw.split("\n") if q.strip()][:4]
    return summary, notes, questions


def rewrite_query(query, history, llm):
    """Turn a follow-up like 'why is that?' into a standalone question using recent chat history."""
    if not history:
        return query
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    prompt = (
        "Rewrite the last question as a standalone question using the conversation for context. "
        "Return only the rewritten question.\n\n"
        f"Conversation:\n{convo}\n\nLast question: {query}"
    )
    return llm.invoke(prompt).content.strip()


def make_sources(docs, video_id):
    """Convert retrieved chunks into clickable timestamp links."""
    sources = []
    for doc in sorted(docs, key=lambda d: d.metadata["start_sec"]):
        sec = doc.metadata["start_sec"]
        sources.append({
            "label": f"{sec // 60}:{sec % 60:02d}",
            "link": f"https://youtu.be/{video_id}?t={sec}",
            "text": doc.page_content[:200],
        })
    return sources


def show_sources(sources):
    with st.expander("📄 Sources from the video"):
        for s in sources:
            st.markdown(f"**[▶ {s['label']}]({s['link']})** — {s['text']}...")


# ---------------------------------------------------------------------------
# Cached pipeline (runs once per video)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Processing video transcript and building index...")
def initialize_rag(video_id):
    text, offsets = fetch_transcript(video_id)
    chain, llm = build_rag(build_chunks(text, offsets))
    summary, notes, questions = generate_study_material(llm, text)
    return {
        "chain": chain,
        "llm": llm,
        "video_id": video_id,
        "summary": summary,
        "notes": notes,
        "questions": questions,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def ask(question):
    st.session_state.pending_question = question


def clear_chat():
    st.session_state.messages = []


st.title("🎥 Chat with YouTube Video")
st.caption("Paste a YouTube link, then ask questions, check the sources, and download study notes.")

with st.sidebar:
    st.header("🎬 Video")
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    process_button = st.button("Process Video", type="primary")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None

if process_button and youtube_url:
    try:
        st.session_state.rag = initialize_rag(extract_video_id(youtube_url))
        st.session_state.messages = []
        st.sidebar.success("Video processed! You can now chat.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

rag = st.session_state.rag
if rag is None:
    st.info("👈 Enter a YouTube URL and click 'Process Video' in the sidebar to get started.")
    st.stop()

with st.sidebar:
    st.video(f"https://www.youtube.com/watch?v={rag['video_id']}")
    st.download_button("⬇️ Download study notes", rag["notes"], file_name="study_notes.md")
    st.button("🗑️ Clear chat", on_click=clear_chat)

# A question comes either from the chat box or from a clicked suggestion
typed_query = st.chat_input("Ask something about the video...")
pending_query = st.session_state.pop("pending_question", None)
user_query = typed_query or pending_query

with st.expander("📌 Video summary", expanded=not (st.session_state.messages or user_query)):
    st.markdown(rag["summary"])

# Starter questions, shown only before the chat begins
if not (st.session_state.messages or user_query):
    st.caption("💡 Try asking:")
    columns = st.columns(2)
    for i, q in enumerate(rag["questions"]):
        columns[i % 2].button(q, key=f"suggested_{i}", on_click=ask, args=(q,))

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

# New question
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    history = st.session_state.messages[-4:]  # taken before adding the new question
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            standalone_query = rewrite_query(user_query, history, rag["llm"])
            result = rag["chain"].invoke(standalone_query)
            sources = make_sources(result["docs"], rag["video_id"])
        st.markdown(result["answer"])
        show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": sources}
    )
    