#Feldarabolja a szöveget kisebb részekre
#maximum max_words szavas blokkban
#az érték egy lista lesz amiben lesznek a feldarabolt szövegrészek

def simple_chunk(text:str,max_words:int = 20):
    words = text.split()
    chunks = []

    for i in range(0,len(words),max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

if __name__ == "__main__":
    
    from src.agentic_rag.loader import load_sample
    text = load_sample()
    print("Eredti szöveg:")
    print(text)
    print("Feldarabolva:")

    for i, chunk in enumerate(simple_chunk(text, max_words=10), start=1):
        print(f"\nChunk {i}:")
        print(chunk)