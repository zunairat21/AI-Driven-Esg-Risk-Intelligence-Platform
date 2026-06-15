from transformers import(
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

from src.rag.retriever import ESGRetriever

class ESGRAGPipeline:

    def __init__(self):
        self.retriever = ESGRetriever()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-base"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base"
        )

    def generate_answer(self, question):
        results = self.retriever.retrieve_documents(
            question, 
            top_k=3
        )
        
        documents = results["documents"][0]

        if not documents:
          return "No relevant ESG documents found."

        context = "\n\n".join(documents)

        prompt = f"""
        You are an ESG analyst.

        Based only on the context below, identify the ESG-related risks, concerns, or controversies associated with the company.

        Provide a concise summary in 3-5 sentences.

        Context:
        {context}

        Question:
        {question}
        
        Answer:
        """

    

        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation=True,
            max_length=1024
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200

        )

        answer=self.tokenizer.decode(
        outputs[0],
            skip_special_tokens = True
        )
        return answer