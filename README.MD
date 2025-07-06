#TASK1 
- Performed EDA on financial product complaints
- Filtered for 5 key product types (credit card, personal loan, etc.)
- Removed empty complaint narratives
- Cleaned text (lowercase, removed special characters)
- Saved cleaned data to data/filtered_complaints.csv
Task 2: Chunking, Embedding, and Vector Indexing
 chunk complaint narratives and embed with MiniLM

- Split long narratives using LangChain's RecursiveCharacterTextSplitter
- Embedded chunks with all-MiniLM-L6-v2 from SentenceTransformers
- Stored embeddings and metadata in ChromaDB persistent vector store
task 3
- Created prompt template for context-aware answering
- Integrated ChromaDB to retrieve top-k relevant chunks
- Used Hugging Face DistilGPT2 for generation
- Function `ask_question_rag()` returns answer + source chunks
