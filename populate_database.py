import argparse
import os
import shutil
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.vectorstores.chroma import Chroma
from utils import CHROMA_PATH, DATA_PATH, EMBEDDING, VECTORDATABASE_PATH
import pandas as pd




def load_documents(data_path=DATA_PATH):
    """
    Load the documents from the data file. Here, the data file is a text file. Data should be preprocess before loading in this text file
    """
    if not data_path.endswith(".txt"):
        raise ValueError("Data path must be a text file.")
    with open(data_path, 'r') as f:
        data = f.read()
    return data


def create_chunks(documents: list[str]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([documents])
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
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def main():
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