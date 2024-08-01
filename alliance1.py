import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader

import os

import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversational_chain():
    prompt_template = """
    Hi there! I'm here to help you with any questions you have about Alliance University.
    I'll provide the most detailed and accurate answers based on the information I have.

    If I don't have the exact context, I'll still try my best to give you a helpful answer.
    So, let's get started!

    Context:\n{context}\n
    Question:\n{question}\n

    Answer :  

    """
    model=ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-70b-versatile",temperature=0.3)
  
    # model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Alliance PDF")
    st.header("Alliance University Chatbot 💁")

    user_question = st.text_input("Ask a question related to Alliance University")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on the Submit & Process button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
