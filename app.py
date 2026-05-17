from groq import Groq
from dotenv import load_dotenv
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import get_embedding_model
from utils.retriever import create_vector_store


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")


client = Groq(api_key=api_key)


documents = load_pdf("data/sample.pdf")


chunks = split_documents(documents)


embedding_model = get_embedding_model()


vector_store = create_vector_store(
    chunks,
    embedding_model
)

print("AI Financial Assistant Ready!")
print("Type 'exit' to stop.\n")

while True:

    
    user_question = input("You: ")

    if user_question.lower() == "exit":
        print("Goodbye!")
        break

    
    retrieved_docs = vector_store.similarity_search(
        user_question,
        k=3
    )

   
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    
    prompt = f"""
    Answer the question based only on the provided context.

    Context:
    {context}

    Question:
    {user_question}
    """

    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    
    answer = response.choices[0].message.content

    print("\nAI Answer:\n")

    print(answer)

    print("\n" + "="*60 + "\n")