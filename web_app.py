"""
Numobel RAG Web App
====================
Streamlit Web UI layer for our RAG Chatbot.
"""

import streamlit as st

import rag_chatbot as rag

# Set page settings
st.set_page_config(
    page_title="Numobel Assistant",
    page_icon="🏭",
    layout="centered"
)

# Initialize vector database (cached so it doesn't reload heavily on every turn)
@st.cache_resource(show_spinner="Loading Knowledge Base...")
def load_db():
    collection = rag.get_collection()
    count = collection.count()
    if count == 0:
        with st.spinner("Ingesting product data into ChromaDB..."):
            rag.ingest_data(collection)
    return collection

collection = load_db()

# Application Title
st.title("🏭 Numobel Product Assistant")
st.markdown(
    f"""
    Welcome! I can answer questions about Numobel products. 
    \n**Memory limit:** {rag.MEMORY_LIMIT} turns | **Query Rewrite:** {'Enabled' if rag.REWRITE_QUERY else 'Disabled'}
    """
)

# Initialize Chat Memory in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display full chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if prompt := st.chat_input("Ask about products..."):
    # Render user query
    with st.chat_message("user"):
        st.markdown(prompt)
        
    rag.log.debug("WEBUI_INPUT | %s", prompt)

    # Bot thinking state
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Rewrite Query (if memory active)
            search_query = rag.rewrite_query(prompt, st.session_state.messages)
            
            # 2. Retrieve Context
            hits = rag.retrieve(collection, search_query)
            
            # Log retrieved context
            for i, h in enumerate(hits, 1):
                rag.log.debug("WEB_RETRIEVED_%d | dist=%.4f | %s", i, h['distance'], h['metadata'].get('product_name', '?'))
            
            # 3. Generate Answer
            if not hits:
                answer = "I don't have that information."
            else:
                answer = rag.generate_answer(prompt, hits, st.session_state.messages)
            
            rag.log.debug("WEBUI_OUTPUT | %s", answer.replace('\n', '\\n'))
            
            # Streamlit output display
            st.markdown(answer)
    
    # Store history for memory
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": answer})
