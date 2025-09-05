from sentence_transformers import SentenceTransformer
from typing import List

#kis méretű modellt betöltünk
_model = SentenceTransformer("all-MiniLM-L6-V2")

#feladata, hogy a chunkokból vektort állítson elő, hogy kereshetőek legyenek
def embed_texts(texts:List[str]):
    #az gesz texts átalakítja vektorrá majd a numpy tömbként adja vissza és végül listaként kapjuk meg
    embeddings = _model.encode(texts,convert_to_numpy=True).tolist()
    return embeddings

if __name__ == "__main__":
    
    #teszt
    from src.agentic_rag.loader import load_sample
    from src.agentic_rag.chunker import simple_chunk

    text = load_sample()
    chunks = simple_chunk(text, max_words=10)

    print("Feldarabolt szövegek:", chunks)

    vectors = embed_texts(chunks)
    print(vectors[0][:5])