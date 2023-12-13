import streamlit as st
from huggingface_hub import notebook_login
from PyPDF2 import PdfReader
import io
from langchain.embeddings import HuggingFaceInstructEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setting up the Streamlit page
st.set_page_config(page_title="PDF Insights Explorer", page_icon="")

# Define a class to hold the text and metadata with the expected attributes
class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

# Define the function to read and extract text from a PDF byte stream
def read_pdf(file_stream):
    reader = PdfReader(file_stream)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ""  # Adding a fallback for pages with no text
    return text

# Initialize session state variables
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

# Sidebar for Hugging Face Login and PDF Upload
with st.sidebar:
    st.subheader("Hugging Face Login")
    hf_token = st.text_input("Enter your Hugging Face Access Token", type="password")
    submit_button = st.button("Authenticate")

    if submit_button:
        try:
            notebook_login(hf_token)
            st.success("Connected successfully to Hugging Face Hub.")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

    st.subheader("Your Documents")
    uploaded_files = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type='pdf')
    process_button = st.button("Process PDFs")

# Main Page Interface
st.header("Chat with Your PDFs 📄")

# Handling the PDF upload and processing
if process_button and uploaded_files:
    documents = []   # List to store document objects
    for uploaded_file in uploaded_files:
        # Read file as bytes
        bytes_data = uploaded_file.getvalue()
        file_name = uploaded_file.name
        st.write(f"Reading {file_name}...")
        text = read_pdf(io.BytesIO(bytes_data))
        documents.append(Document(text))  # Add each document as an instance of Document class

    st.session_state.documents_processed = True
    st.success("PDFs processed successfully!")

    # Combine all texts and split into chunks
    combined_text = " ".join([doc.page_content for doc in documents])
    DEVICE = "cuda"  # Use "cuda" for GPU
    embeddings = SentenceTransformerEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_text = text_splitter.split_documents(documents)
    st.write(f"Number of text chunks: {len(split_text)}")

    # Creating database of embeddings
    db = Chroma.from_documents(split_text, embeddings, persist_directory="db")
    st.success("Embeddings processed and database created.")

    # Initialize the model and tokenizer for conversational AI
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Function to generate a response from the model
    def generate_response(prompt_text):
        inputs = tokenizer.encode(prompt_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response_text

    # Set up the conversational retrieval chain
    retriever = db.as_retriever(search_kwargs={'k': 2})
    st.session_state.chain = ConversationalRetrievalChain.from_llm(model, retriever, return_source_documents=True)

# Chat interface
if st.session_state.documents_processed:
    st.subheader("Chat with AI")
    user_query = st.text_input("Ask a question about your documents:", key="user_query")
    if st.button("Submit"):
        if user_query:
            # Generating a response
            response = generate_response(user_query)
            st.write('Answer:', response)
        else:
            st.warning("Please enter a question.")
else:
    st.write("Please upload and process PDFs to enable the chat feature.")

#%%
