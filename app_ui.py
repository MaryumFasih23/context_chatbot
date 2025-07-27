import os
import streamlit as st
import time
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Smart Document Assistant",
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Custom menu button */
        .menu-button {
            position: fixed;
            top: 1rem;
            left: 1rem;
            z-index: 999999;
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            color: white;
            font-size: 1.2rem;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
        }
        
        .menu-button:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .menu-button:active {
            transform: scale(0.95);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Custom header styling */
        .custom-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        .custom-header h1 {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .custom-header p {
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            font-weight: 300;
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            animation: fadeInUp 0.3s ease-out;
            position: relative;
        }
        
        .user-message {
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            margin-left: 2rem;
            color: white;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }
        
        .bot-message {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            margin-right: 2rem;
            color: white;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }
        
        .message-avatar {
            position: absolute;
            top: -10px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        
        .user-avatar {
            right: -20px;
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            color: white;
        }
        
        .bot-avatar {
            left: -20px;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            color: white;
        }
        
        .message-content {
            line-height: 1.6;
            font-size: 1rem;
        }
        
        .message-time {
            font-size: 0.8rem;
            opacity: 0.8;
            margin-top: 0.5rem;
            text-align: right;
        }
        
        /* Sidebar styling */
        .sidebar-content {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 1rem;
            border: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        /* Status indicators */
        .status-indicator {
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 500;
            display: inline-block;
            margin: 0.25rem;
        }
        
        .status-ready {
            background: #10b981;
            color: white;
        }
        
        .status-loading {
            background: #3b82f6;
            color: white;
        }
        
        .status-error {
            background: #ef4444;
            color: white;
        }
        
        /* Loading animation */
        .loading-dots {
            display: inline-block;
        }
        
        .loading-dots:after {
            content: '⠋';
            animation: loading-spinner 1s infinite;
        }
        
        @keyframes loading-spinner {
            0% { content: '⠋'; }
            10% { content: '⠙'; }
            20% { content: '⠹'; }
            30% { content: '⠸'; }
            40% { content: '⠼'; }
            50% { content: '⠴'; }
            60% { content: '⠦'; }
            70% { content: '⠧'; }
            80% { content: '⠇'; }
            90% { content: '⠏'; }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Input styling */
        .stChatInput > div {
            border-radius: 25px;
            border: 2px solid #e5e7eb;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .stChatInput > div:focus-within {
            border-color: #3b82f6;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        }
        
        /* Metrics styling */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #3b82f6;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZATION AND CACHING
# ============================================================================
@st.cache_resource
def load_models():
    """Load and cache the AI models"""
    with st.spinner("Loading AI models..."):
        model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        llm = HuggingFacePipeline(pipeline=pipe)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embedding_model

@st.cache_resource
def setup_vectorstore(_embedding_model):
    """Setup and cache the vectorstore"""
    with st.spinner("Processing documents..."):
        if not os.path.exists("data/my_notes.txt"):
            st.error("Document file 'data/my_notes.txt' not found!")
            st.stop()
        
        loader = TextLoader("data/my_notes.txt", encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        db = FAISS.from_documents(docs, _embedding_model)
    return db

@st.cache_resource
def setup_qa_chain(_llm, _db):
    """Setup and cache the QA chain"""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=_db.as_retriever(),
        memory=memory
    )
    return qa_chain

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("""
        <div class="sidebar-content">
            <h2 style="margin-top: 0;">AI Assistant</h2>
            <p>Your intelligent document companion</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model status
    st.subheader("System Status")
    
    if not st.session_state.models_loaded:
        st.markdown('<span class="status-indicator status-loading">Loading Models</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-ready">Models Ready</span>', unsafe_allow_html=True)
    
    # Document info
    if os.path.exists("data/my_notes.txt"):
        with open("data/my_notes.txt", "r", encoding="utf-8") as f:
            content = f.read()
            word_count = len(content.split())
            char_count = len(content)
        
        st.markdown('<span class="status-indicator status-ready">Document Loaded</span>', unsafe_allow_html=True)
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{word_count:,}</div>
                    <div class="metric-label">Words</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.message_count}</div>
                    <div class="metric-label">Messages</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-indicator status-error">No Document</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Controls
    st.subheader("Controls")
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.message_count = 0
        st.rerun()
    
    if st.button("Refresh Models", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.models_loaded = False
        st.rerun()
    
    # Tips
    with st.expander("Tips for Better Results"):
        st.markdown("""
        - **Be specific**: Ask detailed questions about the document content
        - **Use context**: Reference previous parts of our conversation
        - **Try different angles**: Rephrase questions if needed
        - **Ask follow-ups**: Build on previous answers for deeper insights
        """)

# ============================================================================
# MAIN INTERFACE
# ============================================================================

# Custom header
st.markdown("""
    <div class="custom-header">
        <h1>🤖 Smart Document Assistant</h1>
        <p>Ask intelligent questions about your documents using advanced AI</p>
    </div>
""", unsafe_allow_html=True)

# Load models (cached)
try:
    llm, embedding_model = load_models()
    db = setup_vectorstore(embedding_model)
    qa_chain = setup_qa_chain(llm, db)
    st.session_state.models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Chat interface
st.subheader("Chat with Your Document")

# Display chat history
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #6b7280;">
                <h3>Welcome! How can I help you today?</h3>
                <p>Ask me anything about your document and I'll provide detailed answers.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        for i, (user_q, bot_a) in enumerate(st.session_state.chat_history):
            timestamp = datetime.now().strftime("%H:%M")
            
            # User message
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-avatar user-avatar">👤</div>
                    <div class="message-content">{user_q}</div>
                    <div class="message-time">{timestamp}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-avatar bot-avatar">🤖</div>
                    <div class="message-content">{bot_a}</div>
                    <div class="message-time">{timestamp}</div>
                </div>
            """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask me anything about your document...", key="chat_input")

if user_input:
    # Show loading state
    with st.spinner("Processing your question..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        try:
            result = qa_chain({"question": user_input})
            answer = result["answer"]
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, answer))
            st.session_state.message_count += 1
            
            progress_bar.empty()
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            st.error(f"Error processing question: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 1rem;">
        <p>Powered by Flan-T5 and Streamlit | Built with ❤️ for intelligent document interaction</p>
    </div>
""", unsafe_allow_html=True)