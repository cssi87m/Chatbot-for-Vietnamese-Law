import argparse
import os
import shutil
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.vectorstores.chroma import Chroma
from utils import DATA_PATH, EMBEDDING, VECTORDATABASE_PATH
import pandas as pd




def load_documents(data_path: str):
    """
    Load the documents from the data file. Here, the data file is a text file. Data should be preprocess before loading in this text file
    """
    loader = DirectoryLoader(
            data_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def create_chunks(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=VECTORDATABASE_PATH, embedding_function=EMBEDDING
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add chunk content and cid in chunks.csv 
    chunks_df = pd.DataFrame([chunk.page_content for chunk in chunks_with_ids], columns=["content"])
    chunks_df["cid"] = [chunk.metadata["id"] for chunk in chunks_with_ids]
    chunks_df.to_csv("chunks.csv", index=False)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    """
    Generate unique IDs for each chunk by combining source, page, and chunk index.
    """
    # Dictionary to keep track of chunk indices for each page
    page_chunk_indices = {}
    
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "0")
        current_page_id = f"{source}:{page}"
        
        # Initialize or increment the chunk index for this page
        if current_page_id not in page_chunk_indices:
            page_chunk_indices[current_page_id] = 0
        else:
            page_chunk_indices[current_page_id] += 1
        
        # Create a unique ID that includes source, page, and chunk index
        # Removing the file path and replacing special characters to make clean IDs
        clean_source = os.path.basename(source).replace(".", "_").replace(" ", "_")
        chunk_id = f"{clean_source}_p{page}_c{page_chunk_indices[current_page_id]}"
        
        # Add the unique ID to the chunk's metadata
        chunk.metadata["id"] = chunk_id
    return chunks


def clear_database():
    if os.path.exists(VECTORDATABASE_PATH):
        shutil.rmtree(VECTORDATABASE_PATH)

def main():
    # documents = load_documents()
    # print(documents[2].page_content)
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data file.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents(args.data_path)
    chunks = create_chunks(documents)
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()