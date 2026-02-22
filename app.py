import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain

st.title("📄 PDF Intelligence Bot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # save and load PDF
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    # break text into manageable bites
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embeddings & Vector Store
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # RAG Chain Setup
    llm = ChatOllama(model="llama3")
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 5. User Input
    user_input = st.text_input("Ask a question about your document:")
    if user_input:
        response = retrieval_chain.invoke({"input": user_input})
        st.write(response["answer"])