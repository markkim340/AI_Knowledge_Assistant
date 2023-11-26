# from dotenv import load_dotenv
# load_dotenv()

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
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler


# App title
st.set_page_config(page_title="A.I Knowledge Assistant")

# Title
st.title("📄 A.I Knowledge Assistant")
st.write("---")

# Upload File
uploaded_file = st.file_uploader("Choose a file", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# Prompt
st.header("무엇을 도와드릴까요?")
question = st.text_input('Enter your question:', placeholder = 'EX) Please provide a short summary.', disabled=not uploaded_file)

# If Uploaded
if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)

  # Split
  text_splitter = RecursiveCharacterTextSplitter(
      # Set a really small chunk size, just to show.
      chunk_size = 200,
      chunk_overlap  = 10,
      length_function = len,
      is_separator_regex = False,
  )

  texts = text_splitter.split_documents(pages)
  print(texts)

  # Embedding
  embeddings_model = OpenAIEmbeddings()

  # Load it into Chroma
  db = Chroma.from_documents(texts, embeddings_model)

  # Stream Handler
  class StreamHandler(BaseCallbackHandler):
      def __init__(self, container, initial_texts=""):
         self.container = container
         self.text = initial_texts

      def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
         self.text+=token
         self.container.markdown(self.text)

  # Question

  if st.button('Generate'):
      with st.spinner('Loading...'):
        chat_box = st.empty()
        stream_handler = StreamHandler(chat_box)

        llm = ChatOpenAI(temperature=0, streaming=True, callbacks=[stream_handler])
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        qa_chain({"query": question})