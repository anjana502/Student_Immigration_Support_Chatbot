from llama_index.core import PromptTemplate
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import load_index_from_storage
import os
import streamlit as st

def load_vector_store():
    faiss_db_path = "faiss_db"
    embeddings = HuggingFaceEmbedding(model_name = "BAAI/bge-base-en-v1.5")
    storage_context = StorageContext.from_defaults(persist_dir = faiss_db_path)
    index = load_index_from_storage(storage_context, embed_model = embeddings)

    return index

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name = repo_id,
    token = HUGGINGFACEHUB_API_TOKEN,
    temperature = 0.3,
    max_tokens = 256,
)

index = load_vector_store()

template = """
You are an academic assistant.
Based on the provided context, answer clearly and concisely.
Question: {query_str}
Context:
{context_str}
Answer:
"""
prompt = PromptTemplate(template)
query_engine = index.as_query_engine(
    llm = llm,
    similarity_top_k = 4,
    text_qa_template = prompt,
)

if __name__ == "__main__":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask questions about OPT, CPT, and student visa rules"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        response = query_engine.query(prompt)
        
        with st.chat_message("assistant"):
            st.markdown(str(response))
    
        st.session_state.messages.append({"role": "assistant", "content": response})