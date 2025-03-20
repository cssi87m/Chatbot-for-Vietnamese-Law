import torch
import numpy as np
import random
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings

# Set a fixed random seed
SEED = 42

# Configuration settings
CHROMA_PATH = 'chroma'
DATA_PATH = 'luat.txt'
VECTORDATABASE_PATH = 'chroma'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# HuggingFace Embeddings with fixed seed
EMBEDDING = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': DEVICE},
    encode_kwargs={'normalize_embeddings': False}
)
