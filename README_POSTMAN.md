# Postman Collection for BajajxHackRX Metamorphosis API

## Overview
This Postman collection provides a complete testing suite for the BajajxHackRX Metamorphosis API, which offers LLM-powered contextual document QA with explainability and optimized performance.

## Performance Features
- **Document Caching**: Documents are cached after first processing to avoid re-vectorization
- **Batch Processing**: Embeddings are generated in optimized batches for faster processing
- **Parallel Processing**: Questions are processed concurrently for lightning-fast responses
- **Smart Retrieval**: Cached retrieval results reduce repeated vector searches

## Import Instructions
1. Open Postman
2. Click "Import" button
3. Select the `postman_collection.json` file
4. The collection will be imported with all requests and environment variables

## Environment Variables
The collection includes the following variables that you can customize:

- `base_url`: API base URL (default: `http://localhost:8000`)
- `auth_token`: Authorization bearer token
- `document_url`: URL to the document for analysis

## API Endpoint

### POST /api/v1/hackrx/run
Analyzes a document and answers questions about it with AI-powered explanations.

**Performance Notes:**
- First-time document processing: ~30-60 seconds (depending on document size)
- Subsequent requests with same document: ~5-15 seconds (cached)
- Questions processed in parallel for maximum speed

**Headers:**
- `Content-Type: application/json`
- `Accept: application/json`
- `Authorization: Bearer {token}`

**Request Body:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "Question 1 about the document?",
        "Question 2 about the document?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "AI-generated answer 1",
        "AI-generated answer 2"
    ]
}
```

## Optimization Features
1. **Document Caching**: Previously processed documents are cached locally
2. **Batch Embeddings**: Text chunks are embedded in optimized batches
3. **Retrieval Caching**: Search results are cached to avoid repeated vector operations
4. **Parallel Processing**: Multiple questions processed simultaneously

## Sample Data
The collection includes a complete example using a policy document with 10 insurance-related questions.

## Testing
The collection includes automated tests that verify:
- Response status is 200
- Response contains answers array
- Processing time is logged for performance monitoring

## Usage
1. Update environment variables if needed
2. Run the "Document QA Query" request
3. Check the test results tab for validation
4. Monitor response times - subsequent requests should be much faster

## Error Handling
The API returns appropriate HTTP status codes:
- `200`: Success
- `500`: Internal server error (e.g., invalid document URL, processing errors)

## Performance Tips
- Documents are automatically cached after first processing
- Re-running the same document will be significantly faster
- Clear the `document_cache` folder to force re-processing if needed
