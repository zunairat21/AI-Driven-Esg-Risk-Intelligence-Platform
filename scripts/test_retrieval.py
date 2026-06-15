import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
    )

from src.rag.embeddings import ESGEmbeddingGenerator
from src.rag.retriever import ESGRetriever

generator = ESGEmbeddingGenerator()
retriever = ESGRetriever()


documents = generator.load_documents(
    "data/ESG_daily_news.csv"
)
documents = documents[:500]

embeddings = generator.generate_embeddings(
    documents
)

retriever.add_documents(
    documents,
    embeddings
)


query = "Why was Tesla removed from the ESG index?"

results = retriever.retrieve_documents(
    query,
    top_k=3
)

print(results.keys())

print("\nQuery:")
print(query)

print("\nRetrieve Documents")

for doc in results["documents"][0]:
    print(doc[:500])
    print("-" * 80)




