from sentence_transformers import SentenceTransformer
import pandas as pd 

class ESGEmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer (
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_documents(self, csv_path):
        df = pd.read_csv(csv_path)
        documents = []

        for _, row in df.iterrows():
            document = (
                str(row["headline"])
                + "\n\n"
                + str(row["text"])
            )
            
            documents.append(document)

        return documents
            
    def generate_embeddings(self, documents):
        embeddings = self.model.encode(
            documents,
            show_progress_bar = True
        )

        return embeddings