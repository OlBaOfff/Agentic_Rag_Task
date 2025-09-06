from src.agentic_rag.loader import load_sample
from src.agentic_rag.chunker import simple_chunk
from src.agentic_rag.embedding import embed_texts
from src.agentic_rag.retriever import add_documents, query_documents
from src.agentic_rag.llm import generate_answer

#Teljes pipeline, ami egy str fog visszatérni
def rag_pipeline(question:str,max_words:int = 10,top_k:int = 2):
    text = load_sample()
    chunks = simple_chunk(text,max_words=max_words)
    vectors = embed_texts(chunks)
    add_documents(chunks,vectors)
    results = query_documents(question,embed_texts,top_k=top_k)
    context = results["documents"][0]
    answer = generate_answer(question,context)
    return answer

if __name__ == "__main__":
    question = "what is your opinion on dogs?"
    print(f"Question:{question}")
    print(rag_pipeline(question))