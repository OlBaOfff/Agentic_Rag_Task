from   typing import List
import chromadb
from chromadb.utils import embedding_functions

#Létrehoz egy ChromaDB kliens objektumot inmemoryban
chroma_client = chromadb.Client()

#létrehozzuk az adatbázis táblánkat
#tartalmazza a chunkokat és a vektoraikat
collection = chroma_client.create_collection(name="rag_collection")

#táblázat feltöltése egyedi azonosítóval
def add_documents(chunks:List[str],embeddings: List[List[float]]):
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

#kérdésből csinalunk egy embedding vektort
#resultsban pedig benne lesz két chunk amik a legjobbna illettek
def query_documents(query:str,embed_fn,top_k:int = 2):
    query_vec = embed_fn([query])
    results = collection.query(query_embeddings=query_vec,n_results=top_k)
    return results

if __name__ == "__main__":
    from src.agentic_rag.loader import load_sample
    from src.agentic_rag.chunker import simple_chunk
    from src.agentic_rag.embedding import embed_texts

    text = load_sample()
    chunks = simple_chunk(text,max_words=10)
    vectors = embed_texts(chunks)
    
    add_documents(chunks,vectors)
    question = "Melyik állat tud repülni?"
    print(f"\n Kérdés:{question}")
    #dictionary lesz a típusa a resultsnak
    results = query_documents(question,embed_texts,top_k=2)
    print("Talált eredményke: ")
    for doc in results["documents"][0]:
        print("-",doc)
