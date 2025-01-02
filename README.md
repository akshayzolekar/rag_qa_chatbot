# Q/A Chatbot
This is a generative AI powered Chatbot that responds to the questions about PDF files.

## Prerequisites
Create an environment from the requirements.txt

## Embeddings
To create embeddings run ingest.py file.

## Testing
- To run locally, open a command line at the App's top-level directory and run the command:

`streamlit run pdf_chatbot.py`

## Assumptions
- Tables from the PDF document are converted to text data.
- Images are removed from the PDF documents as part of data cleaning.
