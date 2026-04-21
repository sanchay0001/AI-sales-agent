import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Path to knowledge base
KB_PATH = os.path.join(os.path.dirname(__file__), "../knowledge_base/autostream_kb.md")
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "../vectorstore")

def build_vectorstore():
    """Load knowledge base, embed it and save FAISS vectorstore."""

    print("Building vectorstore from knowledge base...")

    # Load the markdown file
    loader = TextLoader(KB_PATH, encoding="utf-8")
    documents = loader.load()

    # Split into chunks
    splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create embeddings using local sentence-transformers (free, no API needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save to disk so we don't rebuild every time
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vectorstore built and saved. Total chunks: {len(chunks)}")

    return vectorstore


def load_vectorstore():
    """Load existing vectorstore from disk, or build if not found."""

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    if os.path.exists(VECTORSTORE_PATH):
        print("Loading existing vectorstore...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = build_vectorstore()

    return vectorstore


def get_retriever():
    """Return a retriever from the vectorstore."""
    vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def retrieve_context(query: str) -> str:
    """Given a user query, return relevant context from knowledge base."""
    retriever = get_retriever()
    docs = retriever.invoke(query)

    # Combine all retrieved chunks into one context string
    context = "\n\n".join([doc.page_content for doc in docs])
    return context