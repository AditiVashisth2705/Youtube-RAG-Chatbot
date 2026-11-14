"""
RAGAS evaluation for the YouTube RAG Assistant.

Run from your project folder (same place as YtChatbot.py and .env):
    python evaluate_rag.py          # generates answers, then scores them
    python evaluate_rag.py --regen  # throw away saved answers and generate again

Needs in .env:  GROQ_API_KEY=...
Optional .env:  GROQ_MODEL=openai/gpt-oss-120b   (model that answers)
                JUDGE_MODEL=openai/gpt-oss-20b   (model that scores)
"""
import json
import os
import sys
import time

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
)
from ragas.run_config import RunConfig

load_dotenv()

ANSWER_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b")
ANSWERS_FILE = "eval_answers.json"

ALTMAN = "TjPPIf-M6KE"    # Sam Altman x Raj Shamani (FO583)
SANDEEP = "3T3bR8sxnmo"   # Sandeep Jethwani x Raj Shamani (FO 50)

# ---------------------------------------------------------------------------
# STEP 1 - fill in every "reference" by hand, from what is actually said in
# the video (1-2 sentences). Leave reference empty ONLY for out_of_scope=True.
# ---------------------------------------------------------------------------
QUESTIONS = [
    # --- Video 1: Sam Altman ---
    {"video": ALTMAN, "out_of_scope": False, "reference": "India is OpenAI's biggest market by ChatGPT users, with more users than the US and fast growth. Altman points to strong builder energy there.",
     "question": "Why is India so important to OpenAI, according to Sam Altman?"},
    {"video": ALTMAN, "out_of_scope": False, "reference": "Altman calls himself a very nerdy kid who was not good at sports but always had friends. He says he always followed what he was curious about.",
     "question": "How does Altman describe himself as a child, and what did he say guided what he did?"},
    {"video": ALTMAN, "out_of_scope": False, "reference": "Altman says not holding equity is weird and people do not like weird, so they wonder why a founder is really doing it. He says he does it because he finds the work the most interesting and useful thing he could do.",
     "question": "What does Altman say about why people are suspicious of founders' motives, and how does that apply to him?"},
    {"video": ALTMAN, "out_of_scope": False, "reference": "Raj points out that Altman holds no equity in the company he built. Altman says not having equity in OpenAI has been a mistake, made for reasons he thought were good.",
     "question": "What did Raj ask about Altman not taking equity in OpenAI, and how did Altman answer?"},
    {"video": ALTMAN, "out_of_scope": False, "reference": "Raj lists the departures and the AI drama. Altman does not address each person; he says people only trust themselves to build this because the stakes feel so high, and that disputes mix concerns about someone's approach with concerns about power concentrating in them.",
     "question": "What does Altman say about the departures of people like Ilya Sutskever, Mira Murati, Dario Amodei and Elon Musk?"},
    {"video": ALTMAN, "out_of_scope": False, "reference": "Altman points to the equity question as hugely confusing to people, and says OpenAI's vision is still not very well understood.",
     "question": "What does Altman say is the biggest misconception people have about him?"},
    {"video": ALTMAN, "out_of_scope": False, "reference": "Altman says the US is unusually negative about AI while India is unusually positive. He explains that faster-growing economies with large unmet needs welcome AI more, and links US skepticism to distrust of powerful tech companies.",
     "question": "Why does Altman think people in India are more optimistic about AI than people in the US?"},
    {"video": ALTMAN, "out_of_scope": False, "reference": "Altman mentions AI solving a hard math problem, the Navier-Stokes result, and hopes AI can help discover deep insights in physics, new medical treatments and new sources of energy.",
     "question": "Which scientific breakthroughs does Altman expect AI to help with?"},
    {"video": ALTMAN, "out_of_scope": True, "reference": "",
     "question": "Which programming language does Altman recommend beginners learn first?"},
    {"video": ALTMAN, "out_of_scope": True, "reference": "",
     "question": "Which cricket team does Altman support?"},

    # --- Video 2: Sandeep Jethwani ---
    {"video": SANDEEP, "out_of_scope": False, "reference": "Wealth creation means earning enough to buy time and freedom to do what you love without spending time on tasks you dislike, and helping others do the same.",
     "question": "What does Sandeep say wealth creation actually means?"},
    {"video": SANDEEP, "out_of_scope": False, "reference": "The three phases are income management (making ends meet on a salary), wealth management (saving and putting money to work for returns), and wealth creation (building wealth to free up time, often through entrepreneurship).",
     "question": "What are the 3 phases of India's wealth journey?"},
    {"video": SANDEEP, "out_of_scope": False, "reference": "Wealthy people know what they want to do with their time. For their money they rely on experts, because the stakes are high, so they can spend their own time elsewhere.",
     "question": "Do wealthy people know what to do with their money?"},
    {"video": SANDEEP, "out_of_scope": False, "reference": "Judge the advisor's ethical framework and whether their intent to grow your capital is genuine. Focus on whether they act in your best interest, because trust is what lets you delegate.",
     "question": "How can you tell whether a wealth management expert is good?"},
    {"video": SANDEEP, "out_of_scope": False, "reference": "Trust is built over time through consistent actions that benefit stakeholders such as employees and customers. Trust is essential both to accumulating wealth and to keeping it.",
     "question": "Why do some people and brands come across as more trustworthy, and does trust attract wealth?"},
    {"video": SANDEEP, "out_of_scope": False, "reference": "Some people build wealth consistently because they use real expertise, either by becoming experts or finding ones to guide them. They also apply the power of compounding across all areas of life.",
     "question": "Why are some people wealthy and others not?"},
    {"video": SANDEEP, "out_of_scope": False, "reference": "Compounding applies to career, reputation and relationships as well as money. Progress may not feel visible at first but adds up, and breaking the thread in any field hurts long-term progress.",
     "question": "How does compounding help in life, according to Sandeep?"},
    {"video": SANDEEP, "out_of_scope": False, "reference": "The top three concepts are probability, dealing with uncertainty, and dealing with insecurity, treating gaps in knowledge or talent as things to learn from.",
     "question": "What are the top 3 concepts everyone should understand to create wealth?"},
    {"video": SANDEEP, "out_of_scope": True, "reference": "",
     "question": "Which cryptocurrency does Sandeep recommend buying?"},
    {"video": SANDEEP, "out_of_scope": True, "reference": "",
     "question": "What is dezerv's current number of customers?"},
]

# Same prompt as YtChatbot.py - keep these two in sync.
PROMPT = PromptTemplate(
    template=""" You are a helpful assistant. Answer only from the provided transcript context. 
If the context is insufficient, just say you do not know the answer.
{context}
Question: {question}
""",
    input_variables=["context", "question"],
)


def build_retriever(video_id, embeddings):
    """Same pipeline as the app: transcript -> chunks -> FAISS -> MMR retriever."""
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["en"])
    text = " ".join(snippet.text for snippet in transcript)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    store = FAISS.from_documents(chunks, embeddings)
    return store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.5})


def collect_answers():
    """Ask every question once and save question, contexts and answer to disk."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(model_name=ANSWER_MODEL, temperature=0.3)
    chain = PROMPT | llm | StrOutputParser()

    retrievers = {}
    rows = []
    for item in QUESTIONS:
        vid = item["video"]
        if vid not in retrievers:
            try:
                print(f"Building index for {vid} ...")
                retrievers[vid] = build_retriever(vid, embeddings)
            except Exception as e:
                print(f"  SKIPPING video {vid}: {type(e).__name__}: {e}")
                retrievers[vid] = None
        if retrievers[vid] is None:
            continue

        docs = retrievers[vid].invoke(item["question"])
        contexts = [d.page_content for d in docs]
        answer = chain.invoke({"context": "\n\n".join(contexts), "question": item["question"]})
        rows.append({**item, "contexts": contexts, "answer": answer})
        print(f"  answered: {item['question'][:70]}")
        time.sleep(2)  # be gentle with Groq's free-tier rate limit

    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def main():
    if "--regen" in sys.argv and os.path.exists(ANSWERS_FILE):
        os.remove(ANSWERS_FILE)

    missing = [q["question"] for q in QUESTIONS if not q["out_of_scope"] and not q["reference"].strip()]
    if missing:
        print(f"{len(missing)} answerable question(s) still have an empty 'reference'. Fill them in first, e.g.:")
        for m in missing[:3]:
            print("  -", m)
        sys.exit(1)

    if os.path.exists(ANSWERS_FILE):
        print(f"Using saved answers from {ANSWERS_FILE} (run with --regen to redo them).")
        with open(ANSWERS_FILE, encoding="utf-8") as f:
            rows = json.load(f)
    else:
        rows = collect_answers()

    answerable = [r for r in rows if not r["out_of_scope"]]
    out_of_scope = [r for r in rows if r["out_of_scope"]]

    # ---- Out-of-scope questions: did the model say "I don't know"? ----
    if out_of_scope:
        print("\n=== OUT-OF-SCOPE QUESTIONS (should be refused) ===")
        refused = 0
        for r in out_of_scope:
            low = r["answer"].lower()
            ok = any(p in low for p in ["do not know", "don't know", "not mentioned", "does not say", "doesn't say", "no information"])
            refused += ok
            print(f"[{'OK ' if ok else 'CHECK'}] {r['question']}\n        -> {r['answer'][:200]}")
        print(f"Refused correctly: {refused}/{len(out_of_scope)} (read the CHECK ones yourself)")

    # ---- Answerable questions: RAGAS ----
    if not answerable:
        print("No answerable rows to score.")
        return

    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "retrieved_contexts": r["contexts"],
            "response": r["answer"],
            "reference": r["reference"],
        }
        for r in answerable
    ])

    judge = LangchainLLMWrapper(ChatGroq(model_name=JUDGE_MODEL, temperature=0))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),   # Groq only supports n=1
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]

    print(f"\nScoring {len(answerable)} questions with judge model {JUDGE_MODEL} (this can take 10-20 minutes) ...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(timeout=180, max_retries=8, max_wait=60, max_workers=1),
        raise_exceptions=False,
    )

    df = result.to_pandas()
    df.to_csv("ragas_results.csv", index=False)
    scores = df.select_dtypes("number")
    print("\n=== RAGAS AVERAGES (answerable questions) ===")
    print(scores.mean().round(3).to_string())
    print("\nRows that failed to score (NaN) per metric:")
    print(scores.isna().sum().to_string())
    print("\nPer-question scores saved to ragas_results.csv")


if __name__ == "__main__":
    main()

    