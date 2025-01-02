import os
import re
from datetime import datetime
import chromadb
import fitz
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from config import (EMD_MODEL_PATH, EMD_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, PRE_PROCESS,
                    REMOVE_FOOTER, REMOVE_TABLE, PROCESS_TABLE, REMOVE_PAGES, 
                    PDF_1_PAGES, PDF_2_PAGES, PDF_3_PAGES)


def persist_dir(model: str, chunk: int, pre_process: bool = False):
    """create directory"""
    if not isinstance(chunk, str):
        chunk = str (chunk)
    if pre_process:
        db = f"db_preprocess_{model)_{chunk}"
    else:
        db = f"db_{model}_{chunk}"
    return db



