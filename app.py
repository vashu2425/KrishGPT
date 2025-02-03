import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import time
import translators as ts
from huggingface_hub import hf_hub_download

# Set the page layout to wide
st.set_page_config(layout="wide")

# ConfigurationNo module named 'langchain_huggingface'
HF_TOKEN = os.getenv("HF_TOKEN") # From .env huggingface token
VECTORSTORE_REPO_ID = "vectorstore/db_faiss"  
MODEL_REPO_ID = "mistralai/Mistral-Nemo-Instruct-2407"

# Constants
CUSTOM_PROMPT_TEMPLATE = """
Use The Pieces Of Information Provided In The Context To Answer User's Question.
If You Don't Know The Answer, Just Say "I Don't Have Information",except this do not say anything. 
Don't Try To Make Up An Answer. Don't Provide Anything Out Of The Given Context.

Context: {context}
Question: {question}

Start The Answer Directly. No Small Talk, Please. The Answer Should Contain All 3 Contexts.
Consider Yourself As God Krishna And Answer The Question Result Should Not Start With "Answer"
"""

def translate_text(text, dest_language="hi"):
    try:
        return ts.google(text, to_language=dest_language)
    except Exception as e:
        st.error(f"Translation failed: {str(e)}")
        return text

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Create directory structure
    os.makedirs("vectorstore/db_faiss", exist_ok=True)
    
    # List of required FAISS files
    faiss_files = ["index.faiss", "index.pkl"]
    
    # Download files from Hugging Face Hub
    for filename in faiss_files:
        if not os.path.exists(f"vectorstore/db_faiss/{filename}"):
            try:
                hf_hub_download(
                    repo_id=VECTORSTORE_REPO_ID,
                    filename=filename,
                    local_dir="vectorstore/db_faiss",
                    token=HF_TOKEN,
                    repo_type="dataset"
                )
            except Exception as e:
                st.error(f"Failed to download {filename}: {str(e)}")
                raise

    return FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, hf_token):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )

def format_source_docs(source_documents):
    formatted_docs = []
    for idx, doc in enumerate(source_documents, start=1):
        content = doc.page_content.replace('\t', ' ').replace('\n', ' ').strip()
        formatted_doc = f"**Source {idx}** (Page {doc.metadata['page']}):\n\n{content[:500]}..." 
        formatted_docs.append(formatted_doc)
    return "\n\n".join(formatted_docs)

def render_predefined_questions():
    predefined_questions = [
        "Meaning of Dharma?",
        "What is the purpose of life?",
        "How to find inner peace?",
        "How can I be a better person?",
        "What is the meaning of life?",
        "How can I be a better friend?"
    ]
    st.markdown("### Or, try one of these:")
    buttons = st.columns(len(predefined_questions))
    for idx, question in enumerate(predefined_questions):
        if buttons[idx].button(question, key=f"predefined_{idx}"):
            st.session_state.selected_question = question
            st.session_state.show_predefined = False

def initialize_session_states():
    session_defaults = {
        "messages": [],
        "selected_question": None,
        "show_predefined": True,
        "last_response": None,
        "translation_done": False
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def animated_response(response_text, is_hindi=False):
    response_placeholder = st.empty()
    accumulated_text = ""
    for char in response_text:
        accumulated_text += char
        style_class = "hindi-text" if is_hindi else "english-text"
        response_placeholder.markdown(f'<div class="{style_class}">{accumulated_text}</div>', unsafe_allow_html=True)
        time.sleep(0.01)
    return accumulated_text

def render_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🐿" if message["role"] == "user" else "🪈"):
            content = message["content"]
            if "hindi-text" in content:
                st.markdown(content, unsafe_allow_html=True)
            else:
                st.markdown(content)

def handle_user_input(prompt, qa_chain):
    if prompt:
        with st.chat_message("user", avatar="🐿"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            animated_response(result)
            if "don't have information" not in result.lower():
                with st.expander("Source Documents"):
                    st.markdown(format_source_docs(source_documents))

            st.session_state.messages.append({"role": "assistant", "content": result})
            st.session_state.last_response = result
            st.session_state.show_predefined = False
            st.session_state.translation_done = False

        except Exception as e:
            st.error(f"Error: {str(e)}")

def handle_translation():
    if st.session_state.last_response and not st.session_state.translation_done:
        try:
            translated_text = translate_text(st.session_state.last_response, "hi")
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    msg["content"] = f'<div class="hindi-text">{translated_text}</div>'
                    break
            st.session_state.translation_done = True
            st.rerun()
        except Exception as e:
            st.error(f"Translation error: {str(e)}")

def main():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&display=swap');
        .hindi-text { font-family: 'Noto Sans Devanagari', sans-serif; font-size: 16px; line-height: 1.8; }
        .english-text { font-family: Arial, sans-serif; font-size: 16px; line-height: 1.6; }
        .translate-btn { background-color: #4CAF50!important; color: white!important; border-radius: 20px; padding: 6px 20px; }
        .top-left-button { background-color: #e0162e; color: white!important; border-radius: 50px; padding: 10px 20px; }
        body, .stApp { background-color: #1e1e30; }
    </style>
    <a href="https://iskconmangaluru.com/wp-content/uploads/2021/04/English-Bhagavad-gita-His-Divine-Grace-AC-Bhaktivedanta-Swami-Prabhupada.pdf" 
       target="_blank" class="top-left-button">Source Bhagavad Gita PDF</a>
    """, unsafe_allow_html=True)
    
    st.title("Ask Krishna! 🦚")
    st.markdown('<p class="hindi-text" style="color:#666666; font-size:20px;">शांति स्वीकृति से शुरू होती है</p>', 
                unsafe_allow_html=True)

    initialize_session_states()
    render_chat_messages()

    if st.session_state.show_predefined:
        render_predefined_questions()

    prompt = st.chat_input("What's your curiosity?") or st.session_state.selected_question
    st.session_state.selected_question = None

    try:
        vectorstore = get_vectorstore()
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(MODEL_REPO_ID, HF_TOKEN),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        if prompt:
            handle_user_input(prompt, qa_chain)

        if st.session_state.get("last_response"):
            if st.button("🌐 Translate to Hindi", key="translate_btn"):
                handle_translation()

    except Exception as e:
        st.error(f"Initialization error: {str(e)}")

if __name__ == "__main__":
    main()
