from transformers import pipeline
from typing import List

#kisméretű modell base
#nagyobb méretű modell sajnos kissé pontatlan a válasz vagy lemarad egy két betű vagy többet generál
generator = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_answer(question:str,context_chunks:List[str]):
    #str alakítjuk mert listát nem tud kezelni az LLM inputként
    context_text = " ".join(context_chunks)
    prompt = f"{context_text}\n\nQuestion: {question}\nAnswer in one full English sentence:"
    result = generator(prompt,max_new_tokens=100,do_sample=True,temperature=0.6)
    answer = result[0]["generated_text"].strip()
    return answer

if __name__ == "__main__":
    from src.agentic_rag.loader import load_sample
    from src.agentic_rag.chunker import simple_chunk
    from src.agentic_rag.embedding import embed_texts
    from src.agentic_rag.retriever import add_documents, query_documents

    #Szöveg betöltése
    text = load_sample()
    chunks = simple_chunk(text,max_words=10)
    vectors = embed_texts(chunks)
    add_documents(chunks,vectors)

    #Kérdés
    question = "Melyik állat tud repülni?"
    results = query_documents(question,embed_texts,top_k=2)
    context = results["documents"][0]
    print(f"Kérdés:{question}")
    print(f"Kontextus:{context}")

    #Válasz generálása
    answer = generate_answer(question,context)
    print(f"\nLLM válasza: {answer}")