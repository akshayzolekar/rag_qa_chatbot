from pathlib import Path


MODEL_CD = r"\models"  # TODO local model directory

# EMD MODEL
MINILM = "sentence-transformers_all-MiniLM-L6-v2"
MPNET = "sentence-transformers_all-mpnet-base-v2"
E5_BASE_V2 = "intfloat_e5-base-v2"
BGE_LARGE_EN = "bge-large-en-v1.5"
EMD_MODEL = BGE_LARGE_EN  # TODO local embedding model selection
EMD_MODEL_PATH_INT = str(Path(MODEL_CD, EMD_MODEL))

EMD_MODEL_PATH_EXT = "bge/large-en-v1.5"  # TODO model path from huggingface

EMD_MODEL_PATH = EMD_MODEL_PATH_INT  # TODO decide internal or external path

CHUNK_OVERLAP: int = 150
CHUNK_SIZE: int = 500  # TODO can try 700 to check performance
PRE_PROCESS: bool = True  # TODO True in case of any pre processing
REMOVE_FOOTER: bool = True  # TODO True to remove footers
REMOVE_TABLE: bool = False  # TODO True to remove tables
PROCESS_TABLE: bool = True  # TODO True to process table to text
REMOVE_PAGES: bool = False  # TODO True to ignore unnecessary pages

PDF_1_PAGES = []

PDF_2_PAGES = []

PDF_3_PAGES = []

# LLM MODEL
LAMINI_738 = "MBZUAI-LaMini-T5-738M"
FLAN_T5_BASE = "google-flan-t5-base"
LAMINI_783 = "MBZUAI-LaMini-Flan-T5-783M"
CHAR_IN_TOKEN = 4
PERCENT_CONTEXT_LLM = 0.75
TEMP_LLM = 0.1
MAX_CONTEXT_LENGTHS = {
    LAMINI_738: 512,
    FLAN_T5_BASE: 512,
    LAMINI_783: 512,
}
LLM_MODEL = LAMINI_738
LLM_MODEL_PATH_INT = str(Path(MODEL_CD, LLM_MODEL))

LLM_MODEL_PATH_EXT = "MBZUAI/LaMini-T5-738M"  # TODO model path from huggingface

LLM_MODEL_PATH = LLM_MODEL_PATH_INT  # TODO decide internal or external path
