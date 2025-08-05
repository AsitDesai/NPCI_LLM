# NPCI_LLM - RAG System

A Retrieval-Augmented Generation (RAG) system built with LlamaIndex for document processing, embedding generation, and intelligent text retrieval.

## 🏗️ Architecture

This RAG system is built entirely using **LlamaIndex** for all data operations:

### ✅ **LlamaIndex Integration**
- **Document Loading**: LlamaIndex SimpleDirectoryReader for TXT, PDF, DOCX files
- **Text Chunking**: LlamaIndex SentenceSplitter and TokenTextSplitter
- **Embedding Generation**: LlamaIndex HuggingFace embedding integration
- **Vector Storage**: LlamaIndex Qdrant vector store integration
- **Text Processing**: LlamaIndex Document and TextNode objects

### 📁 **Repository Structure**
```
NPCI_LLM/
├── config/                 # Configuration and settings
├── data/                   # Data ingestion and processing
│   └── ingestion/         # Document loading, preprocessing, chunking
├── embeddings/            # Embedding generation and management
├── retrieval/             # Vector search and retrieval
├── generation/            # LLM response generation
├── api/                   # FastAPI endpoints
├── tests/                 # Test suites
├── scripts/               # Utility scripts
├── reference_documents/   # Source documents (.txt, .pdf, .docx)
└── storage/              # Local storage and cache
```

## 🚀 **Current Status**

### ✅ **Phase 2: Data Pipeline - COMPLETED**
- Document collection from `reference_documents/`
- Text preprocessing and cleaning
- Intelligent chunking using LlamaIndex
- Integration pipeline validation

### 🔄 **Next Phase: Embeddings & Storage**
- LlamaIndex HuggingFace embedding generation
- Qdrant vector database integration
- Embedding storage and indexing

## 🛠️ **Setup**

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   cp env.example .env
   # Update .env with your API keys and settings
   ```

3. **Test Data Pipeline**:
   ```bash
   python test_post_data.py
   ```

## 📊 **Test Results**

The data pipeline successfully:
- ✅ Loads documents from `reference_documents/`
- ✅ Preprocesses and cleans text content
- ✅ Creates intelligent chunks using LlamaIndex
- ✅ Validates all components work together

## 🔧 **Configuration**

All settings are managed through environment variables in `.env`:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (via LlamaIndex)
- **Vector Database**: Qdrant (localhost:6333)
- **Chunking**: LlamaIndex SentenceSplitter (1024 chars, 200 overlap)
- **LLM**: Mistral (for generation)

## 📝 **Usage**

1. Place your documents in `reference_documents/`
2. Run the data pipeline tests
3. Generate embeddings (Phase 3)
4. Start the API server (Phase 4)

---

**Built with LlamaIndex for robust, production-ready RAG capabilities.**