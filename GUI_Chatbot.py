import re
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
import streamlit as st 

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel, RunnableSequence
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

############################---Config---#######################################

load_dotenv()

st.set_page_config(
    page_title="Youtube Video Chatbot",
    layout="centered",
)

st.title("Youtube Video Chatbot")
st.caption("Ask Questions based on video and get satisfying Answers!!")

############################---Session_State---#######################################

if "message" not in st.session_state:
    st.session_state.messages = []
    
if "chain" not in st.session_state:
    st.session_state.chain = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "video_url" not in st.session_state:
    st.session_state.video_url = ""
    
############################---Video_ID_Extraction---#######################################

def video_id_extraction(url):
    try:
        parsed_url = urlparse(url.strip())
        host = parsed_url.netloc.lower().split(":")[0]
        
        if host == "youtu.be" or host.endswith(".youtu.be"):
            video_id = parsed_url.path.strip("/").split("/")[0]
        elif (
            host == "youtube.com" or host.endswith(".youtube.com") or host == "youtube-nocookie.com" or host.endswith("youtube-nocookie.com")
        ):
            if parsed_url.path == "/watch":
                video_id = parse_qs(parsed_url.query).get("v",[None])[0]
            elif re.match("^/(shorts|embed|live)/[^/]+", parsed_url.path,
            ):
                video_id = parsed_url.path.split("/")[2]
            else:
                return None
        else:
            return None
        
        if video_id and re.fullmatch("[A-Za-z0-9_-]{11}", video_id):
            return video_id
        return None
    except(ValueError, IndexError):
        return None

############################---Fetch_Transcript---#######################################

# def fetch_transcript (video_id):
#     transcript_api = YouTubeTranscriptApi()
#     trasncript_list = (transcript_api.fetch(video_id,languages="en").to_raw_data())
#     transcript = " ".join(chunk['text'] for chunk in trasncript_list)
#     if not transcript.strip():
#         raise ValueError("Transcript is not present!")
#     return transcript

def fetch_transcript(video_id):
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    try:
        transcript = transcript_list.find_transcript(["en"])
    except Exception:
        raise ValueError(
            "No English transcript track could be selected."
        )

    transcript_data = transcript.fetch().to_raw_data()

    return " ".join(
        chunk["text"] for chunk in transcript_data
    )

############################---Prepare DOC and Vector Store---#######################################      

def vec_store_creation(transcript):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 250
    )
    chunks = splitter.create_documents([transcript])
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

############################---Prepare context text---#######################################      

def prepare_docs(retriever_docs):
    context_text = "\n\n".join(doc.page_content for doc in retriever_docs)
    return context_text

############################---Chains_creation---####################################### 
    
def qa_chain_creation(vector_Store):
    retriever = vector_Store.as_retriever(search_type = "similarity", search_kwargs = {"k" : 4})
    llm = ChatOpenAI(model = "gpt-4.1-mini", temperature=0.2)
    
    prompt = PromptTemplate(
        template= """ 
        You are a helpful Assistant. Answer only from provided transcript context. If provided context is insufficient, then just say I don't have that much knowledge about the topic.
        You may use the conversation history to understand follow-up questions, but every factual answer must be supported by the transcript context.
        Transcript Context : {context}
        Coversation History : {chat_history}
        Question : {question}
        """,
        input_variables=['context','chat_history', 'question']
    )
    
    parallel_chain = RunnableParallel({
        'context' : RunnableLambda(lambda x : x['question'])| retriever | RunnableLambda(prepare_docs),
        'question' : RunnableLambda(lambda x : x['question']),
        'chat_history' : RunnableLambda(lambda x : x['chat_history']) 
    }
    )
    
    parser = StrOutputParser()
    
    chain = parallel_chain | prompt | llm | parser
    return chain

############################---Sidebar : Video Inputs and controls---####################################### 
    
with st.sidebar:
    st.header("video Settings")
    video_url = st.text_input(
        "Enter Youtube Url",
        placeholder=("https://www.youtube.com...."),
        key="url_input"
    )   
    
    load_button = st.button(
        "Load Video",
        type="primary",
        use_container_width=True
    ) 
    
    if load_button:
        video_id = video_id_extraction(video_url)
        if not video_id:
            st.error("Enter a valid Youtube Url!")
        else:
            try:
                with st.spinner("Fetching Video Transcript from provided url for vector store..."):
                    transcript = fetch_transcript(video_id)
                    vector_store = vec_store_creation(transcript)
                    chain = qa_chain_creation(vector_store)
                st.session_state.chain = chain
                st.session_state.video_id = video_id
                st.session_state.video_url = video_url.strip()
                st.session_state.messages = []
                
                st.success("Video Loaded Successfully!")
            except TranscriptsDisabled:
                st.error("Captions are disabled for this video!")
            except Exception as e:
                st. error(f"Could not load the video: {e}")
    st.divider()
    
    if st.session_state.chain is not None:
        st.success("Video Ready for QA!")
        st.link_button("Open Youtube Video", st.session_state.video_url , use_container_width=True)
    if st.button("clear_chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

############################---History Loading---####################################### 
        
if st.session_state.chain is None:
    st.info(
        "Enter a url to start the session"
    )
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

############################---Accept Questions and Answe---#######################################         

if question := st.chat_input("Ask Anything regarding the Video..."):
    if st.session_state.chain is None:
        st.warning("Please load a video to start the session")
    else:
        st.session_state.messages.append(
            {
                'role' : 'user',
                'content' : question
            }
        )
        
        with st.chat_message('user'):
            st.markdown(question)
            
        history = st.session_state.messages[:-1]
        
        chat_history = "\n".join(
            f"{'user' if msg['role'] == 'user' else 'assistant' }:"
            f"{msg['content']}"
            for msg in history  
        )
        
        with st.chat_message('assistant'):
            with st.spinner('Thinking'):
                
                try:
                    
                    answer = st.session_state.chain.invoke(
                        {
                        'question':question,
                        'chat_history': chat_history
                        }
                    )
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {
                            'role' : 'assistant',
                            'content': answer
                        }
                    )
                except Exception as e:
                    st.error(f"Error while Generating Answer: {e}")
    