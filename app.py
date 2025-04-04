import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY in your .env file.")
else:
    genai.configure(api_key=api_key)

def read_pdf(docs):
    """Read PDF files and extract text."""
    text = ""
    for pdf in docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text = text + page.extract_text() or ""  # Handle None return from extract_text()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

def create_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def convert_to_vectors(text_chunks):
    """Convert text chunks into vector embeddings and save them."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Use FAISS.from_texts instead of FAISS.from_text
        vectors = FAISS.from_texts(text_chunks, embedding=embeddings)
        vectors.save_local("faiss_index")
        st.success("Vector database created successfully!")
    except Exception as e:
        st.error(f"Error creating vector database: {e}")


def create_conversational_chain():
    """Create a conversational chain for Q&A."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in 
    the provided context, just say, "The answer is not available in the context."
    
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """ 
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {e}")
        return None

def process_question(question):
    """Process user input and generate a response."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


        # Perform similarity search to find relevant documents
        docs = new_db.similarity_search(question)

        chain = create_conversational_chain()
        if chain:
            response = chain(
                {
                    "input_documents": docs,
                    "question": question
                }, return_only_outputs=True
            )
            st.write("Reply: ", response["output_text"])
        else:
            st.error("Failed to generate a response.")
    except Exception as e:
        st.error(f"Error processing question: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF using Gemini")

    question = st.text_input("Ask a Question from the PDF Files")

    if question:
        process_question(question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on Submit Button", accept_multiple_files=True)
        
        if st.button("Submit"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = read_pdf(pdf_docs)
                    if raw_text.strip():  # Check if raw_text is not empty
                        text_chunks = create_chunks(raw_text)
                        convert_to_vectors(text_chunks)
                    else:
                        st.error("No text extracted from the PDFs.")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
