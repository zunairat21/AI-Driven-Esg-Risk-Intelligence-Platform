import sys
from pathlib import Path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from src.rag.rag_pipeline import ESGRAGPipeline

rag = ESGRAGPipeline()

question = "What ESG risks are associated with Tesla?"

answer = rag.generate_answer(
    question
)

print("\nQuestion:")
print(question)

print("\nGenerated Answer:")
print(answer)