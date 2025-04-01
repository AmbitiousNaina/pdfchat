# pdfchat
PDF Document QnA With Langchain And Google Gemini 

model used : gemini-1.5-flash

# Working :

PDF Upload and Text Extraction: Users upload PDF files via a Streamlit interface. The code extracts text from the PDFs using the PdfReader library, combining all pages into a single text string.

Text Chunking: The extracted text is split into smaller, manageable chunks using LangChain's RecursiveCharacterTextSplitter. These chunks allow for efficient processing and retrieval.

Embedding Generation: Each chunk is converted into vector embeddings using Google's Gemini Embedding Model (embedding-001), which captures the semantic meaning of the text.

FAISS Index Creation: The embeddings are stored in a FAISS vector database, enabling fast similarity searches to find relevant chunks when a user asks a question.

Question Processing: When a user submits a question, the FAISS database performs a similarity search to retrieve the most relevant text chunks based on their semantic similarity to the query.

Answer Generation: The retrieved chunks and user query are passed to Google's Gemini 1.5 Flash model, which generates a detailed, context-aware answer using LangChain's load_qa_chain.

Response Display: The generated answer is displayed back to the user in the Streamlit interface.


