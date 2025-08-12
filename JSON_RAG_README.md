# JSON RAG Pipeline

A complete Retrieval-Augmented Generation (RAG) pipeline specifically designed for processing JSON files containing structured chunk objects. This pipeline uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and integrates with Mistral 24B for answer generation.

## 🚀 Features

- **JSON-Specific Processing**: Designed to handle JSON files with arrays of chunk objects
- **Advanced Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for high-quality embeddings
- **Vector Storage**: Stores embeddings in Qdrant with payload filtering capabilities
- **Smart Retrieval**: Semantic search with optional filtering by category and type
- **Answer Generation**: Integrates with Mistral 24B for intelligent answer generation
- **Token Management**: Ensures chunks stay within 200-token limit
- **Batch Processing**: Efficient batch processing for large datasets

## 📋 Requirements

### Dependencies
- Python 3.8+
- sentence-transformers
- qdrant-client
- mistralai
- torch
- transformers

### Environment Variables
Create a `.env` file with the following variables:

```env
# Qdrant Configuration
QDRANT_HOST=0.0.0.0
QDRANT_PORT=6333
QDRANT_API_KEY=dhsuhdujhisduygh
VECTOR_DB_NAME=answers_collection
VECTOR_DB_DIMENSION=384
VECTOR_DB_METRIC=cosine

# Mistral Configuration
MISTRAL_API_KEY=htsiRa57UO5unjCb3vBAHk3HS0oP1s0l
MISTRAL_MODEL=mistral-small-latest

# Embedding Configuration
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_DIMENSION=384
EMBEDDING_BATCH_SIZE=32

# Retrieval Configuration
RETRIEVAL_TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.3

# Document Configuration
REFERENCE_DOCUMENTS_DIR=./reference_documents
```

## 📁 Project Structure

```
├── scripts/
│   ├── json_data_ingestion.py      # JSON data processing pipeline
│   └── build_json_embeddings.py    # Embedding generation pipeline
├── retrieval/
│   └── json_retriever.py           # JSON-specific semantic retriever
├── generation/
│   └── json_rag_pipeline.py        # Complete RAG pipeline
├── reference_documents/
│   └── upi_decline.json            # Example JSON data file
├── run_json_rag_pipeline.py        # Complete pipeline runner
├── JSON_RAG_CLI.py                 # Interactive CLI interface
└── JSON_RAG_README.md              # This file
```

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Start Qdrant** (local instance should be running on 0.0.0.0:6333):
   ```bash
   # Qdrant should already be running on the server
   # If not, contact the system administrator
   ```

## 🚀 Quick Start

### 1. Run Complete Pipeline

To run the entire pipeline from data ingestion to testing:

```bash
python run_json_rag_pipeline.py
```

This will:
- Process JSON files in `reference_documents/`
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Store vectors in Qdrant with payload filtering
- Test the retrieval and generation system

### 2. Interactive CLI

For interactive querying:

```bash
python JSON_RAG_CLI.py
```

Available commands:
- `query <text>` - Ask a question
- `category <cat> <text>` - Ask filtered by category
- `type <type> <text>` - Ask filtered by type
- `filters` - Show available categories and types
- `help` - Show help
- `quit` - Exit

### 3. Individual Components

#### Data Ingestion
```bash
python scripts/json_data_ingestion.py
```

#### Embedding Generation
```bash
python scripts/build_json_embeddings.py
```

#### Test Retrieval
```bash
python retrieval/json_retriever.py
```

#### Test RAG Pipeline
```bash
python generation/json_rag_pipeline.py
```

## 📊 JSON Data Format

The pipeline expects JSON files with the following structure:

```json
[
  {
    "category": "Balance-related decline",
    "scenario": "Denial / Confusion",
    "user_statement": "But I have enough balance, why did it say insufficient balance?",
    "agent_response": "There may be a temporary issue in balance validation...",
    "system_behavior": "Transaction gets declined when account balance is less than the requested debit amount.",
    "agent_guideline": "Never challenge the user's belief about having balance...",
    "type": "scenario"
  },
  {
    "category": "Balance-related decline",
    "scenario": "FAQ",
    "user_statement": "Can I retry the same transaction?",
    "agent_response": "Yes, you can retry the transaction once you've confirmed...",
    "type": "faq"
  }
]
```

### Field Processing

The pipeline concatenates the following fields for embedding:
- `category`
- `scenario`
- `user_statement`
- `agent_response`
- `system_behavior`
- `agent_guideline`

All original fields are preserved in the payload for filtering and retrieval.

## 🔍 Retrieval Features

### Semantic Search
- Uses cosine similarity for vector matching
- Configurable top-k results
- Score threshold filtering

### Payload Filtering
- Filter by `category` field
- Filter by `type` field
- Combined filtering support

### Example Queries

```python
# Basic retrieval
results = retriever.retrieve("insufficient balance", top_k=5)

# Category filtering
results = retriever.retrieve_by_category("balance check", "Balance-related decline")

# Type filtering
results = retriever.retrieve_by_type("PIN reset", "scenario")

# Combined filtering
results = retriever.retrieve(
    "transaction failed",
    category_filter="Balance-related decline",
    type_filter="faq"
)
```

## 🤖 Answer Generation

The RAG pipeline:

1. **Retrieves** relevant documents using semantic search
2. **Builds** context from retrieved documents with metadata
3. **Generates** answers using Mistral 24B
4. **Returns** structured response with timing information

### Response Format

```python
RAGResponse(
    answer="Generated answer text...",
    query="Original user query",
    retrieved_documents=[...],  # List of JSONRetrievalResult
    generation_time=1.234,
    retrieval_time=0.567,
    total_time=1.801,
    model_used="mistral-large-latest"
)
```

## ⚙️ Configuration

### Token Limits
- Maximum tokens per chunk: 200 (configurable)
- Token estimation: ~4 characters per token

### Batch Processing
- Embedding batch size: 32 (configurable)
- Qdrant upload batch size: 100

### Model Settings
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector dimension: 384
- Distance metric: Cosine similarity

## 📈 Performance

### Typical Performance Metrics
- **Embedding Generation**: ~100-500 embeddings/second (CPU)
- **Retrieval**: ~10-50ms per query
- **Answer Generation**: ~1-5 seconds (depends on Mistral API)

### Optimization Tips
- Use GPU for embedding generation if available
- Adjust batch sizes based on available memory
- Use payload filtering to reduce search space
- Cache frequently used embeddings

## 🐛 Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   - Check if Qdrant is running on 0.0.0.0:6333
   - Verify host/port configuration
   - Check API key (dhsuhdujhisduygh)

2. **Mistral API Error**
   - Verify API key is set correctly (htsiRa57UO5unjCb3vBAHk3HS0oP1s0l)
   - Check API quota and limits
   - Ensure model name is correct (mistral-small-latest)

3. **Memory Issues**
   - Reduce batch size
   - Process smaller files
   - Use CPU instead of GPU

4. **No Results Found**
   - Check if embeddings were generated
   - Verify collection exists in Qdrant
   - Lower score threshold

### Debug Mode

Enable debug logging by setting:
```env
LOG_LEVEL=DEBUG
```

## 🔧 Customization

### Adding New Fields
To add new fields to the concatenation process, modify the `_concatenate_fields` method in `JSONDataIngestion`.

### Changing Embedding Model
Update the model name in settings:
```env
EMBEDDING_MODEL_NAME=your-model-name
EMBEDDING_MODEL_DIMENSION=your-model-dimension
```

### Custom Filtering
Add new filter conditions in the `_create_filter` method in `JSONSemanticRetriever`.

## 📝 Examples

### Example Usage in Code

```python
from generation.json_rag_pipeline import JSONRAGPipeline

# Initialize pipeline
pipeline = JSONRAGPipeline(top_k=5)

# Answer a query
response = pipeline.answer_query("What should I do if I have insufficient balance?")

print(f"Answer: {response.answer}")
print(f"Retrieved {len(response.retrieved_documents)} documents")
print(f"Total time: {response.total_time:.3f}s")

# Answer with category filter
response = pipeline.answer_query_by_category(
    "How do I check my balance?",
    "Balance-related decline"
)
```

### Example CLI Session

```
🚀 JSON RAG PIPELINE - INTERACTIVE CLI
================================================================================

🤖 JSON RAG > query What should I do if I have insufficient balance?

🔍 Processing query: 'What should I do if I have insufficient balance?'

================================================================================
📋 RAG RESPONSE
================================================================================
🤔 Query: What should I do if I have insufficient balance?
💡 Answer: If you're experiencing insufficient balance issues, you should first verify your account balance by checking your bank statement or UPI app. If you're certain you have sufficient funds, there may be a temporary issue with balance validation. In such cases, contact your bank directly for assistance. You can also try selecting a different linked account if you have multiple accounts. Remember that failed transactions due to insufficient balance do not debit your account, so you can safely retry once balance is available.
⏱️  Total time: 2.345s
🔍 Retrieval time: 0.123s
🤖 Generation time: 2.222s
📊 Retrieved documents: 3
🤖 Model used: mistral-large-latest
================================================================================
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the configuration examples

---

**Note**: This pipeline is specifically designed for JSON data with structured chunk objects. For other data formats, consider using the general RAG pipeline components.

