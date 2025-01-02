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
        chunk = str(chunk)
    if pre_process:
        db = f"db_preprocess_{model}_{chunk}"
    else:
        db = f"db_{model}_{chunk}"

    return db


def custom_extract_text(page, df_toc:pd.DataFrame, pre_process: bool = False, remove_footer: bool = False,
                        remove_table: bool = False, process_table: bool = False):
    """Extract text from a page and returns a list of string"""

    text = ""

    if pre_process:
        # The footers are not same in different pages, so clipping text below a certain page length
        footer_height = 40.0
        portrait_rect = fitz.Rect(0.0, 0.0, 612.0, 792.0)
        landscape_rect = fitz.Rect(0.0, 0.0, 792.0, 612.0)
        if page.rect == portrait_rect:
            footer_rect = fitz.Rect(0.0, 792.0 - footer_height, 612.0, 792.0)
        else:
            footer_rect = fitz.Rect(0.0, 612.0 - footer_height, 792.0, 612.0)

        # remove footer
        if remove_footer:
            page.add_redact_annot(footer_rect)

        tabs = []
        try:
            tabs = page.find_tables()
            if tabs.tables:
                for tab in tabs:
                    if remove_table:
                        tab_rect = tab.bbox
                        page.add_redact_annot(tab_rect)

                    if process_table and pre_process:
                        tab_rect = tab.bbox
                        page.add_redact_annot(tab_rect)
                        table_text = ""
                        lines = tab.extract()
                        headers = [text.replace("\n", "-") for text in lines[0]]
                        values = [text.replace("\n", ",") for text in lines[1]]
                        for no, line in enumerate(lines):
                            # ignore headers
                            if no != 0:
                                for i in range(len(headers)):
                                    if line[i] is not None:
                                        values[i] = line[i].replace("\n", ", ")
                                    table_text = table_text + f"{headers[i]} is {values[i]}"

                                    if i == len(headers) - 1:
                                        table_text = table_text + ".\n"
                                    else:
                                        table_text = table_text + " and "

                        tab.generated_text = table_text

        except:
            pass

        # apply the removal of annot
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # update block text
        text = text + process_page_blocks(page, tabs, df_toc)

    else:
        text = page.get_text(sort=True)

    return text


def preprocess_pdf_data(document, pre_process, remove_footer, remove_table, process_table, remove_pages):
    """Pre-process document"""

    if remove_pages:
        remove_pages_no = []
        if "Vol_I.pdf" in str(document):
            remove_pages_no = PDF_1_PAGES
        elif "Vol_II.pdf" in str(document):
            remove_pages_no = PDF_2_PAGES
        elif "Vol_III.pdf" in str(document):
            remove_pages_no = PDF_3_PAGES

        print(f"Removing pages: {remove_pages_no}")

        remove_pages_no = [no - 1 for no in remove_pages_no]

        document.delete_pages(remove_pages_no)

    # extract table of content from document
    df_toc = pd.DataFrame(document.get_toc(), columns=['#', 'original_heading', 'destinationpage'])
    df_toc['original_heading'] = df_toc['original_heading'].str.strip()
    df_toc['heading_text'] = ''

    pages_ext = [custom_extract_text(page, df_toc, pre_process, remove_footer, remove_table, process_table) for page in
                 document]

    return pages_ext


def process_page_blocks(page, tables, df_toc):
    """
    Return paragraph text of a page.
    This function will also convert sentences with bullet to sentence with full stop.
    """
    text = ''
    for block in page.get_text("dict", sort=True)['blocks']:
        block_text = ''
        table_text1 = ''
        table_text2 = ''
        for table in tables:

            # if table comes first
            if ((table.bbox[3] - block["bbox"][1]) < 30) and hasattr(table, "generated_text"):
                table_text1 = table.generated_text
                table_text1 = table_text1 + '\n'
                # once assigned, delete attribute
                delattr(table, "generated_text")

            # if table comes after block
            elif ((table.bbox[1] - block["bbox"][3]) < 30) and hasattr(table, "generated_text"):
                table_text2 = table.generated_text
                table_text2 = '\n' + table_text2
                # once assigned, delete attribute
                delattr(table, "generated_text")

        if 'lines' in block.keys():
            for line in range(len(block['lines'])):
                line_text = ''
                for i in range(len(block['lines'][line]['spans'])):
                    span_text = block['lines'][line]['spans'][i]['text']
                    line_text = line_text + span_text
                if '•' == line_text:
                    line_text = '.'
                block_text = block_text + ' ' + line_text
        text = text + table_text1 + '\n' + block_text + "\n" + table_text2

    # add unassigned table text to the page
    for table in tables:
        if hasattr(table, "generated_text"):
            text = text + "\n" + table.generated_text
            delattr(table, "generated_text")

    return text


def remove_excessive_newlines(text_list):
    """
    Removes excessive new lines from the list of texts
    """
    cleaned_pages = []
    for text in text_list:
        cleaned_text = re.sub(r'\n{2,}', '\n', text)
        cleaned_pages.append(cleaned_text)
    return cleaned_pages


def main(model: str, chunk: int = 500, overlap: int = 50, pre_process: bool = False,
         remove_footer: bool = False, remove_table: bool = False, process_table: bool = False,
         remove_pages: bool = False):
    """Main calculation"""
    # load the document and split it into chunks
    loaders = [fitz.open(os.path.join("docs", fn)) for fn in os.listdir("docs")]

    print(f"Total documents loaded: {len(loaders)}.")

    for document in loaders:

        print(f"Creating embeddings for {str(document)}.")

        pages = preprocess_pdf_data(document, pre_process, remove_footer, remove_table, process_table, remove_pages)
        pages = remove_excessive_newlines(pages)
        # split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk,
                                                       chunk_overlap=overlap)
        texts = text_splitter.create_documents(pages)

        # create the open-source embedding function
        embeddings = SentenceTransformerEmbeddings(
            model_name=EMD_MODEL_PATH
        )

        # create vector store here
        client = chromadb.PersistentClient(path=persist_dir(
            model=model, chunk=chunk, pre_process=pre_process))

        Chroma.from_documents(
            client=client,
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_dir(model=model, chunk=chunk, pre_process=pre_process)
        )


if __name__ == "__main__":
    print("Creating embeddings....")
    main(
        model=EMD_MODEL,
        chunk=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
        pre_process=PRE_PROCESS,
        remove_footer=REMOVE_FOOTER,
        remove_table=REMOVE_TABLE,
        process_table=PROCESS_TABLE,
        remove_pages=REMOVE_PAGES
    )
    print("Completed embeddings....")
