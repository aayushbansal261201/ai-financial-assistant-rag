from groq import Groq
from dotenv import load_dotenv
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import get_embedding_model
from utils.retriever import create_vector_store

# Load environment variables
load_dotenv()

# Load Groq API key
api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)

# Load PDF
documents = load_pdf("data/sample.pdf")

# Split into chunks
chunks = split_documents(documents)

# Load embedding model
embedding_model = get_embedding_model()

# Create vector store
vector_store = create_vector_store(
    chunks,
    embedding_model
)

print("AI Financial Assistant Ready!")
print("Type 'exit' to stop.\n")

while True:

    # User input
    user_question = input("You: ")

    if user_question.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(
        user_question,
        k=3
    )

    # Combine retrieved text
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # Create prompt
    prompt = f"""
    Answer the question based only on the provided context.

    Context:
    {context}

    Question:
    {user_question}
    """

    # Send to LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Extract answer
    answer = response.choices[0].message.content

    print("\nAI Answer:\n")

    print(answer)

    print("\n" + "="*60 + "\n")