import streamlit as st
import time

from src.rag import ComplaintRAG

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CrediTrust Complaint Assistant",
    page_icon="💬",
    layout="wide"
)

# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = ComplaintRAG()

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("💬 CrediTrust Complaint Assistant")
st.success("NEW APP VERSION LOADED")

st.markdown("""
Ask questions about customer complaints related to:

- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers
""")

st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------

question = st.text_area(
    "Enter your question",
    placeholder="Why are customers unhappy with credit cards?",
    height=100
)

col1, col2 = st.columns([1, 1])

ask_button = col1.button(
    "🔍 Ask",
    use_container_width=True
)

clear_button = col2.button(
    "🗑️ Clear Conversation",
    use_container_width=True
)

# --------------------------------------------------
# Clear Chat
# --------------------------------------------------

if clear_button:
    st.session_state.history = []
    st.rerun()

# --------------------------------------------------
# Ask Question
# --------------------------------------------------

if ask_button:

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Analyzing complaints..."):

        result = st.session_state.rag.ask(question)

    st.session_state.history.append(
        {
            "question": question,
            "answer": result["answer"],
            "documents": result["documents"],
            "sources": result["sources"]
        }
    )

# --------------------------------------------------
# Display Conversation History
# --------------------------------------------------

for chat in reversed(st.session_state.history):

    st.divider()

    # ---------------------------
    # Question
    # ---------------------------

    st.subheader("Question")

    st.info(chat["question"])

    # ---------------------------
    # Answer
    # ---------------------------

    st.subheader("Answer")

    answer_placeholder = st.empty()

    streamed_text = ""

    for word in chat["answer"].split():

        streamed_text += word + " "

        answer_placeholder.markdown(streamed_text)

        time.sleep(0.01)

    # ---------------------------
    # Sources
    # ---------------------------

    st.subheader("Retrieved Sources")

    documents = chat["documents"]
    sources = chat["sources"]

    for i in range(min(2, len(documents))):

        metadata = sources[i]

        with st.expander(f"Source {i+1}"):

            st.markdown(
                f"""
                **Product Category:** {metadata.get('product_category', 'N/A')}

                **Issue:** {metadata.get('issue', 'N/A')}

                **Company:** {metadata.get('company', 'N/A')}

                **Complaint ID:** {metadata.get('complaint_id', 'N/A')}
                """
            )

            st.markdown("**Complaint Excerpt**")

            st.write(documents[i])

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Retrieval-Augmented Generation (RAG) system for complaint analysis."
)