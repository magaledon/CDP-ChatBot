# CDP Support Agent Chatbot

## Overview

This project is a support agent chatbot designed to answer "how-to" questions related to four Customer Data Platforms (CDPs): Segment, mParticle, Lytics, and Zeotap. The chatbot uses official documentation from these CDPs to provide accurate and helpful responses. It leverages web scraping, document embedding with FAISS, and a T5 text generation model to extract relevant information and generate answers. A Flask-based web interface provides a user-friendly way to interact with the chatbot.

## Key Features

- **"How-to" Question Answering:** Answers user questions about how to perform specific tasks or use features within each of the supported CDPs.
- **Documentation Extraction:** Retrieves relevant information from the official documentation of Segment, mParticle, Lytics, and Zeotap.
- **Flask Web Interface:** Provides a simple and intuitive web-based interface for interacting with the chatbot.
- **Text Generation:** The LLM model generates the text using the documentation that was extracted.

## Technologies Used

- **Python:** The primary programming language for the chatbot and Flask web application.
- **Langchain:** A framework for building language model-powered applications.
- **Transformers (Hugging Face):** Provides pre-trained models and tools for natural language processing.
- **Flask:** A micro web framework for building the chatbot's web interface.
- **FAISS (Facebook AI Similarity Search):** A library for efficient similarity search and clustering of high-dimensional vectors, used for document embedding and retrieval.

## Libraries Used (with brief explanations)

- **flask:** For creating the web application.
- **langchain:** For building the chatbot.
- **langchain\_community.document\_loaders:** For loading data from URLs.
- **langchain\_community.embeddings:** For creating embeddings.
- **langchain.text\_splitter:** For splitting documents into smaller chunks.
- **langchain\_community.vectorstores:** For using FAISS as a vector store.
- **langchain.chains:** For creating the QA chain.
- **transformers:** For using pre-trained models.
- **transformers.AutoModelForSeq2SeqLM, transformers.AutoTokenizer, transformers.pipeline:** For initializing and using the T5 model.
- **langchain\_community.llms:** For wrapping the Hugging Face pipeline as a Langchain language model.

## Setup and Usage

1.  **Clone the repository:**
       git clone https://github.com/magaledon/CDP-ChatBot.git
2.  **Navigate to the project directory:**
   cd CDP-ChatBot
3.  **Install the required packages:**
   pip install -r requirements.txt
4. **Run the Flask application:**
   python app.py
5.  **Access the chatbot:** Open your web browser and go to `http://127.0.0.1:5000/`

## How to Use

1.  Enter your question related to Segment, mParticle, Lytics, or Zeotap in the text box.
2.  Click the "Submit" button.
3.  The chatbot will display the answer based on the information extracted from the official documentation.

## Project Structure
![image](https://github.com/user-attachments/assets/a257c42c-e01b-444c-a9d9-035ca533183e)



## Evaluation Criteria

The assignment was evaluated based on the following criteria:

-   Accuracy and completeness of the chatbot's responses.
-   Code quality and build.
-   Handling of variations in question phrasing and terminology.
-   Implementation of bonus features (cross-CDP comparisons, advanced questions).
-   Overall user experience and chatbot interaction.

## Additional Notes

-   This project demonstrates the use of web scraping, document indexing, and natural language processing techniques to build a support agent chatbot.
-   The chatbot is designed to provide accurate and helpful answers based on the official documentation of the supported CDPs.







