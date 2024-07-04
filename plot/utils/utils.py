import json
from langchain_community.document_loaders import PyMuPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import JSONLoader
import os
import subprocess
from langchain_community.document_loaders import UnstructuredExcelLoader

def load_pdf(file_path):
    print(f"Loading PDF file from {file_path}")
    loader = PyMuPDFLoader(file_path=file_path)
    data = loader.load()
    print(f"Loaded {len(data)} pages from PDF")
    return data

def load_csv(file_path, delimiter=",", quotechar='"', fieldnames=None):
    print(f"Loading CSV file from {file_path} with delimiter '{delimiter}' and quotechar '{quotechar}'")
    loader = CSVLoader(
        file_path=file_path,
        csv_args={
            "delimiter": delimiter,
            "quotechar": quotechar,
            "fieldnames": fieldnames,
        }
    )
    data = loader.load()
    print(f"Loaded {len(data)} rows from CSV")
    return data

def load_excel(file_path):
    print(f"Loading Excel file from {file_path}")
    loader = UnstructuredExcelLoader(file_path, mode="elements")
    data = loader.load()
    print(f"Loaded {len(data)} elements from Excel file")
    return data

def load_json(file_path):
    print(f"Loading JSON file from {file_path}")
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.messages[].content',
        text_content=False
    )
    data = loader.load()
    print(f"Loaded {len(data)} elements from JSON file")
    return data

def split_docs(documents, chunk_size=1000, chunk_overlap=0):
    print(f"Splitting documents into chunks of size {chunk_size} with overlap {chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents=documents)
    print(f"Split documents into {len(chunks)} chunks")
    
    # returning the document chunks
    return chunks

def load_embedding_model(model_path, normalize_embedding=True):
    print(f"Loading embedding model from {model_path}")
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cpu'}, # here we will run the model with CPU only
        encode_kwargs = {
            'normalize_embeddings': normalize_embedding # keep True to compute cosine similarity
        }
    )

# Function for creating embeddings using FAISS
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    print(f"Creating embeddings and saving to {storing_path}")
    # Creating the embeddings using FAISS
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    
    # Saving the model in current directory
    vectorstore.save_local(storing_path)
    print(f"Saved vectorstore to {storing_path}")
    
    # returning the vectorstore
    return vectorstore

def load_qa_chain(retriever, llm, prompt):
    print("Loading QA chain with provided prompt")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever, # here we are using the vectorstore as a retriever
        chain_type="stuff",
        return_source_documents=True, # including source documents in output
        chain_type_kwargs={'prompt': prompt} # customizing the prompt
    )
    
def get_response(query, chain):
    print(f"Getting response for query: {query}")
    # Getting response from chain
    response = chain.invoke({'query': query})
    print(f"Got response: {response}")
    result = response['result']
    result = json.loads(result)
    script = result.get('code')
    print(script)
    
    if script is None:
        key = input("Key CODE is not present, Enter the new key: ")
        script = result.get(key)
        print(script)
    return script

def get_incremental_filename(base_name):
    """Generate an incremental filename."""
    i = 1
    while os.path.exists(f"output/scripts/{base_name}_{i}.py"):
        i += 1
    print(f"{base_name}_{i}")
    return f"{base_name}_{i}"

def create_py_file(script, file_name):
    print(f"Creating Python file {file_name}.py")
    with open(f"output/scripts/{file_name}.py", "w") as file:
        file.write(script)
    print(f"Python file {file_name}.py created successfully")
        
def execute_py_file(file_name):
    print(f"Executing Python file {file_name}.py")
    command = f"python output/scripts/{file_name}.py"
    subprocess.run(command, shell=True, check=True)
    print(f"Executed Python file {file_name}.py successfully")
