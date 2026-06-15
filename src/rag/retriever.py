import chromadb
from sentence_transformers import SentenceTransformer

class ESGRetriever:
    def __init__(self):

        self.model =  SentenceTransformer (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path = "chromadb"
        )

        self.collection = self.client.get_or_create_collection(
            name = "esgnews"
        )
    
    def add_documents(self, documents, embeddings):

        ids = [str(i) for i in range(len(documents))]

      
        self.collection.add(
            ids = ids,
            documents = documents, 
            embeddings = embeddings.tolist()
        )
    
    def retrieve_documents(self, query , top_k = 3):

        query_embedding = self.model.encode(query)

        results = self.collection.query(
            query_embeddings = [query_embedding.tolist()],
            n_results = top_k
        )
        

        return results
    
    