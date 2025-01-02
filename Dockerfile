FROM python:3.8.12

WORKDIR /rag_qa_chatbot

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "ingest.py"]
CMD ["streamlit", "run", "pdf_chatbot.py"]