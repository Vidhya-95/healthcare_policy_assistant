import streamlit as st
import tempfile
from healthcare_policy_assistant_backend import load_and_chunk_document,vector_store,get_qa_chain

st.title("Healthcare Policy Assistant")
uploaded = st.file_uploader("Upload a Policy PDF", type="pdf")

if uploaded:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        pdf_path = tmp.name

    # Save, chunk, index
    chunks = load_and_chunk_document(pdf_path)
    vectorstore = vector_store(chunks)
    chain = get_qa_chain(vectorstore)

    tab1, tab2, tab3 = st.tabs(["Q&A", "Summarize", "Extract Rules"])

    with tab1:
        question = st.text_input("Ask a question about this policy:")
        if question:
            result = chain.invoke({"input": question})
            st.write(result["answer"])

    with tab2:
        st.markdown("Summarizes the policy document")
        if st.button("Summarize"):
            result = chain.invoke({"input": "Summarize this policy document."})
            st.write(result["answer"])

    with tab3:
        st.markdown("Lists all rules and requirements in this policy.")
        if st.button("Extract Rules"):
            result = chain.invoke({"input": "List all rules and requirements in this policy."})
            st.write(result["answer"])