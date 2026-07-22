import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st 

st.set_page_config(page_title="Youtube RAG Assistant", page_icon="🎥", layout= 'centered' )
load_dotenv()

def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return url


@st.cache_resource(show_spinner="Processing video transcript and building index...")


def initialize_rag_chain(video_url):
    video_id = extract_video_id(video_url)


    try: 
        api = YouTubeTranscriptApi() 
        transcript = api.fetch(video_id, languages=["en"]) 
        transcript_text = " ".join(snippet.text for snippet in transcript) 
    except TranscriptsDisabled: 
        raise Exception("No captions available for this video.")  
    except Exception as e:
        raise Exception(f"Failed to fetch transcript: {str(e)}")
    

    splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    chunks = splitter.create_documents([transcript_text])     


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store= FAISS.from_documents(chunks,embeddings) 

    retriever= vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs= {"k":3, "lambda_mult":0.5}
    ) 

    llm= ChatGroq(model= "llama-3.3-70b-versatile", temperature=0.2) 

    prompt= PromptTemplate(
        template= """ You are a helpful assistant. Answer only from the provided transcript context. 
If the context is insufficient, just say you do not know the answer.
{context}
Question: {question}
""", 
        input_variables=['context', 'question']
)

    def format_docs(retrieved_docs): 
         return "\n\n".join(doc.page_content for doc in retrieved_docs)
    

    parallel_chain= RunnableParallel({ 
         'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser= StrOutputParser()

    main_chain= parallel_chain | prompt | llm | parser 

    return main_chain



st.title("🎥 Chat with YouTube Video")
st.write("Enter a YouTube video link below, let us index the transcript, and ask any questions you have!")

with st.sidebar:
   st.header("Settings")
   youtube_url= st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")
   process_button = st.button("Process Video", type="primary")
   
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None


if process_button and youtube_url:
    try:
    
        st.session_state.chain = initialize_rag_chain(youtube_url)
        st.session_state.messages = [] 
        st.sidebar.success("Video processed successfully! You can now chat.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

if st.session_state.chain is not None:
   
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

   
    if user_query := st.chat_input("Ask something about the video..."):
     
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

       
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain.invoke(user_query)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("👈 Please enter a YouTube URL and click 'Process Video' in the sidebar to get started.")

