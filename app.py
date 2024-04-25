
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,SimpleSequentialChain
# from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain

from utils import load_pdfs_from_directory, get_text_chunks
from flask import Flask, request, jsonify
import PyPDF2
import os
from dotenv import load_dotenv
load_dotenv()  # This method will read the .env file and set the variables


app = Flask(__name__)
@app.route('/')
def hello():
    return 'Hello, World!'

# Route to handle file uploads
@app.route('/upload-pdfs', methods=['POST'])
def upload_pdfs():
    if 'pdf_files' not in request.files:
        return "No pdf_files key in request.files", 400

    files = request.files.getlist('pdf_files')
    query = request.form.get('query', '')  # Get the query from form-data

    print(query)
    if not files:
        return "No files uploaded", 400

    extracted_texts = {}
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Read PDF file
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from each page
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:  # Check if text was found
                        text += page_text + "\n"
                        text +="-" * 40 +"\n"

                extracted_texts[file.filename] = text
            except Exception as e:
                extracted_texts[file.filename] = str(e)
#     chunks=get_text_chunks(text)
#     # Create the embeddings of the chunks using openAIEmbeddings
#     embeddings = OpenAIEmbeddings()

#     # Pass the documents and embeddings inorder to create FAISS vector index
#     vectorindex_openai = FAISS.from_texts(chunks, embeddings)
#     # retriever = vectorindex_openai.as_retriever()
#     retriever = vectorindex_openai.as_retriever(search_type="similarity", search_kwargs={"k":2})
#     llm = OpenAI(temperature=0.9, max_tokens=2000)

# # from langchain.chains.question_answering import load_qa_chain

#     qa = ConversationalRetrievalChain.from_llm(llm=llm,
#                                                 retriever=retriever,
#                                                 #  condense_question_prompt=condense_question_prompt,
#                                                 return_source_documents=True,
#                                                 verbose=False)
#     result = qa({"question": "I am sending you documents of resumes of different candidates based on these answer: Seeking a candidate with 10 years of Java experience.", "chat_history": []})   # return jsonify(extracted_texts)



#     return jsonify(result)
    # Assuming the `get_text_chunks` function and others return serializable data
    try:
        chunks = get_text_chunks(text)
        embeddings = OpenAIEmbeddings()  # This needs to be adjusted if it doesn't return serializable data
        vectorindex_openai = FAISS.from_texts(chunks, embeddings)
        retriever = vectorindex_openai.as_retriever(search_type="similarity")
        llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.9, max_tokens=2000)

        qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                                   retriever=retriever,
                                                   return_source_documents=True,
                                                   verbose=False)
        result = qa({"question": f"{query}", "chat_history": []})
        # print(result.answer)
        # Assume you process the text here to get 'result'
        # Serialize Document objects if present
        if 'source_documents' in result:
            result['source_documents'] = [document_to_dict(doc) for doc in result['source_documents']]
        print(f"result::::::::::::::::::::{result}")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'
def document_to_dict(doc):
    return {
        "page_content": doc.page_content
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)







# os.environ['OPENAI_API_KEY'] = 'sk-proj-Q0qXxqHBzIU1ZENj2qCXT3BlbkFJtYZG1q7XLLvHVGfj8Z2A'

# # Specify the directory containing PDFs
# pdf_directory = "./"

# # Load all PDFs from the directory
# pdf_texts = load_pdfs_from_directory(pdf_directory)
# print(pdf_texts)
