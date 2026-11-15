import streamlit as st
from pipeline import answer_question, load_vector_store

st.set_page_config(page_title="Employee Help Desk AI", page_icon="🧑‍💼")
st.title("🧑‍💼 Zyro Dynamics HR Help Desk")
st.caption("Ask questions about company HR policies. Answers are grounded in the available policy PDFs.")

if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = load_vector_store()
    except Exception as exc:
        st.error(f"Knowledge base is not ready: {exc}")
        st.info("Place the 11 HR policy PDFs in data/hr_policies, run ingestion.ipynb, and create the FAISS index.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View PDF sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['file']} — page {source['page']}**")
                    st.caption(source["excerpt"])

if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

question = st.chat_input("Ask an HR policy question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            try:
                result = answer_question(question, st.session_state.vector_store)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("View PDF sources", expanded=True):
                        for source in result["sources"]:
                            st.markdown(f"**{source['file']} — page {source['page']}**")
                            st.caption(source["excerpt"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.info("Check that Ollama is running and the FAISS index exists.")
