__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

# Title
st.title("Chat PDF")
st.write("---")

# Upload File
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# If Uploaded
if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)

  # Split
  text_splitter = RecursiveCharacterTextSplitter(
      # Set a really small chunk size, just to show.
      chunk_size = 100,
      chunk_overlap  = 20,
      length_function = len,
      is_separator_regex = False,
  )

  texts = text_splitter.split_documents(pages)

  # Embedding
  embeddings_model = OpenAIEmbeddings()

  # Load it into Chroma
  db = Chroma.from_documents(texts, embeddings_model)

  # Question
  st.header("Chat with your PDF now!!")
  question = st.text_input('Question', 'input your prompt')

  if st.button('Generate'):
      with st.spinner('Loading...'):
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        answer = qa_chain({"query": question})
        st.write(answer["result"])