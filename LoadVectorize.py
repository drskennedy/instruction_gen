# LoadVectorize.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# load an online pdf and split it
def load_doc() -> 'List[Document]':
    loader = OnlinePDFLoader("https://support.riverbed.com/bin/support/download?did=7q6behe7hotvnpqd9a03h1dji&version=9.15.0")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

# vectorize, commit to disk and create a BM25 retriever
def vectorize(embeddings) -> tuple[FAISS,BM25Retriever]:
    docs = load_doc()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("./opdf_index")
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k=5
    return db,bm25_retriever

# attempts to load vectorstore from disk
def load_db() -> tuple[FAISS,BM25Retriever]:
    embeddings_model = HuggingFaceEmbeddings()
    try:
        db = FAISS.load_local("./opdf_index", embeddings_model)
        bm25_retriever = BM25Retriever.from_documents(load_doc())
        bm25_retriever.k=5
    except Exception as e:
        print(f'Exception: {e}\nno index on disk, creating new...')
        db,bm25_retriever = vectorize(embeddings_model)
    return db,bm25_retriever
