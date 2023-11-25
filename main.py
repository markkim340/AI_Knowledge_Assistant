from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA


# Loader
loader = PyPDFLoader("youth_support.pdf")
pages = loader.load_and_split()


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
question = "무엇을 위한 사업이며, 사업에 선정되기 위해선 어떻게 해야 하나요?"
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
answer = qa_chain({"query": question})

print(answer)