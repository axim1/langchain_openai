
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,SimpleSequentialChain
# from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import PyPDF2
import os

# Function to load PDF files from a specified directory
def load_pdfs_from_directory(directory):
    pdf_texts = {}
    text = ""
    # List all files in the directory
    for filename in os.listdir(directory):
        # Construct full file path
        full_path = os.path.join(directory, filename)

        # Check if the file is a PDF
        if filename.lower().endswith('.pdf'):
            if os.path.isfile(full_path):
                with open(full_path, 'rb') as file:
                    # Create a PDF reader object
                    pdf_reader = PyPDF2.PdfReader(file)

                    # Initialize a variable to store the text of the PDF


                    # Iterate through each page in the PDF
                    for page in pdf_reader.pages:
                        # Extract text from the page and add it to the text variable
                        page_text = page.extract_text()
                        if page_text:  # Check if text was found
                            text += page_text + "\n"
                            text +="-" * 40 +"\n"
                    # Store the collected text in the dictionary with the filename as key
                    pdf_texts[filename] = text
            else:
                print(f"Skipping {filename}, not a file.")
        else:
            print(f"Skipping {filename}, not a PDF.")



    return text


# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="-" * 40,
        chunk_size=2500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

