import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Cassandra
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cassandra.cluster
import cassio
import streamlit as st

def populate_vector_store(doc_dir, db_token, db_id):
    raw_texts = []

    # Read text from all PDFs in the specified directory
    for filename in os.listdir(doc_dir):
        if filename.endswith('.pdf'):
            pdfreader = PdfReader(os.path.join(doc_dir, filename))
            raw_text = ''
            if pdfreader.is_encrypted:
                try:
                    pdfreader.decrypt("")  # Attempt to decrypt with an empty password
                except Exception as e:
                    print(f"Failed to decrypt PDF '{filename}': {e}")
                    continue 
            print(f"PDF '{filename}' is being read")
            for page in pdfreader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content
            raw_texts.append((filename, raw_text))
            print(f"PDF '{filename}' has been saved with length {len(raw_text)}.")  # Print the message


    cassio.init(token=db_token, database_id=db_id)
    embedding = OpenAIEmbeddings()

    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="document_embeddings",
        session=None,
        keyspace=None,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )

    # Split text and generate embeddings for each document
    for filename, raw_text in raw_texts:
        texts = text_splitter.split_text(raw_text)
        astra_vector_store.add_texts(texts)
        print(f"PDF '{filename}' has been embedded with {len(texts)} chunks.")  # Print the message

def initialize_vector_store(db_token, db_id):
    cassio.init(token=db_token, database_id=db_id)
    embedding = OpenAIEmbeddings()

    vector_store = Cassandra(
        embedding=embedding,
        table_name="document_embeddings",
        session=None,
        keyspace=None,
    )
    
    return vector_store

if __name__ == "__main__":
    doc_dir = "docs"  # Directory containing your PDF documents
    db_token = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    db_id = st.secrets["ASTRA_DB_ID"]

    populate_vector_store(doc_dir, db_token, db_id)
