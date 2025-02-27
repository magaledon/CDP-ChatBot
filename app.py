from flask import Flask, render_template, request
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

app = Flask(__name__,template_folder='templates')

# Load data, create embeddings, and initialize the QA chain (as before)
# Documentation URLs from Assignment 2
urls = [
    "https://segment.com/docs/?ref=nav",
    "https://docs.mparticle.com/",
    "https://docs.lytics.com/",
    "https://docs.zeotap.com/home/en-us/"
]

# Load data from URLs
loaders = [WebBaseLoader(url) for url in urls]
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(documents)

# Initialize Sentence Transformer embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
db = FAISS.from_documents(documents, embeddings)

# Save the FAISS index to disk (optional)
db.save_local("faiss_index")

# Load FAISS index from disk
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize a local language model (e.g., T5)
model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a text generation pipeline
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=500)

# Wrap the pipeline with HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(), return_source_documents=True)

def generate_cross_cdp_prompt(query):
    prompt_template = f"Compare and contrast the following Customer Data Platforms based on the user query: Zeotap, Lytics, mParticle, Segment. User query: {query}"
    return prompt_template

@app.route("/", methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        query = request.form['query']
        # prompt = generate_cross_cdp_prompt(query)
        result = qa({"query": query})
        return render_template("index.html", question=query, answer=result["result"],
                               source_documents=result["source_documents"])
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
