# Agentic_Rag_Task

A rendszer egy szöveges dokumentumból képes releváns információt kikeresni, majd válaszólni is egy nyelvi modell segítségével.

Fő elemei:
1)Loader-szöveg betöltése('data/sample/transformer_intro.txt')
2)Chunker-feladarabolja a szövegeket kisebb részekre
3)Embeddings-vektor-reprezentáció('HuggingFaceEmbeddings')
4)Vector Store-ChromaDB tárolja az embeddingeket
5)Retriever-kikeresi a legjobban illő 2 db chunkot
6)LLM-válasz generálása 'flan-t5-base' segítségével
7)RetrievalQA(LangChain)-a retriever + LLM összekapcsolása

# Virtuális környezet létrehozása
python -m venv .venv
.venv\Scripts\activate

# Követelmények telepítése
pip install -r requirements.txt

# RAG pipeline futtatása
python -m src.agentic_rag.rag_langchain "Which animal can swim?"
