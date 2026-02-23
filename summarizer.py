import streamlit as st 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq  
from langchain_classic.chains.summarize import load_summarize_chain 

st.set_page_config(page_title="Document Summarizer", page_icon="📄") 
st.title("PDF Summarizer Bot")

llm= ChatGroq(
    temperature=0,
    groq_api_key="your-api-key",
    model_name="llama-3.1-8b-instant"
)

uploaded_file = st.file_uploader("Upload Your PDF to Summarize", type="pdf")

if uploaded_file:
    with st.spinner("Processing...."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue()) 
        loader= PyPDFLoader("temp.pdf")
        docs = loader.load()
        # chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs) 

        chain = load_summarize_chain(llm, chain_type="map_reduce")
        st.subheader("Summary")
        summary = chain.run(splits)
        st.write(summary) 

        st.success("Summary generated successfully!")