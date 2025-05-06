from pleias_rag_interface import RAGWithCitations

# Initialize with your preferred model
rag = RAGWithCitations("PleIAs/Pleias-RAG-1B", backend="remote_vllm")

# Define query and sources
query = "What is the capital of France?"
sources = [
    {
        "text": "Paris is the capital and most populous city of France.",
        "metadata": {"source": "Geographic Encyclopedia", "reliability": "high"}
    },
    {
        "text": "The Eiffel Tower is located in Paris, France.",
        "metadata": {"source": "Travel Guide", "year": 2020}
    }
]

# Generate a response
response = rag.generate(query, sources)

# Print the final answer with citations
#print(response["processed"]["clean_answer"])
print(response["processed"])
