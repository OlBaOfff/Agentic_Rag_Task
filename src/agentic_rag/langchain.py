from src.agentic_rag.loader import load_sample
from src.agentic_rag.chunker import simple_chunk
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA

text = load_sample()
chunks = simple_chunk(text,max_words=10)
#embedding fg kivaltja (vektorra alakit)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="rag_collection",
    embedding_function=embedding_model
)
#feltöltjük chunkokkal és a vektoraikkal
vectorstore.add_texts(chunks)
# 2 legjobb találatott adja vissza
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
#HF-nek a saját fg-je
generator = pipeline("text2text-generation", model="google/flan-t5-base")
llm = HuggingFacePipeline(pipeline=generator)

quan = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    #talált chunkokat odaadja az LLM-nek
    chain_type="stuff"
)

if __name__ == "__main__":
    question = "What do you think about dogs?"
    print(question)
    answer = quan.run(question)
    print(answer)
