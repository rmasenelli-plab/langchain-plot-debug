from utils.utils import load_embedding_model, create_embeddings, get_incremental_filename, execute_py_file, create_py_file, load_json, load_pdf, load_excel, load_csv, split_docs, load_qa_chain, get_response
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import os

MODEL = 'qwen2'

llm = Ollama(model=MODEL, format="json", temperature=0)
embed = load_embedding_model(model_path="all-miniLM-L6-v2")

doc = load_csv("../data/test.csv")
documents = split_docs(documents=doc)

vectorstore = create_embeddings(documents, embed)
retriever = vectorstore.as_retriever()
template = """
### System:
You are a plot generator. Generate a Python script that creates plots for any type of data requested by the user. Respond exclusively with a complete python script, without comment and written on a single line.

### Context:
{context}

### User:
{question}

### Response:
"""
prompt = PromptTemplate.from_template(template)

# Creating the chain
chain = load_qa_chain(retriever, llm, prompt)
question = """
Il grafico dovrebbe avere:

L'asse X rappresenta tutti i mesi che hai a disposizione.
L'asse Y rappresenta i valori numerici corrispondenti a ciascun mese.
Un titolo appropriato per il grafico, come "Valori Mensili durante l'Anno".
Etichette per gli assi X e Y, rispettivamente "Mesi" e "Valori".

Presenta il grafico in modo chiaro e leggibile, con una linea continua che connette i punti per ogni mese.
"""
script = get_response(question, chain)

filename = get_incremental_filename("sales_plot")
create_py_file(script, filename)
execute_py_file(filename)